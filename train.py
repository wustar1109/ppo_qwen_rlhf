"""
PPO RLHF Training for Qwen Vision-Language Model
Main entry point (supports adaptive training).
"""

import logging
import os

from torch.utils.data import DataLoader

from trainer import AdaptivePPOTrainer
from dataset import RLHFDataset, collate_fn
from arguments import parse_args, get_config_from_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    config = get_config_from_args(args)

    logger.info("=" * 50)
    logger.info("PPO Qwen RLHF training config:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

    use_adaptive = args.resume is None
    trainer = AdaptivePPOTrainer(config, use_adaptive=use_adaptive)

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info(f"Loading dataset from {config.train_data_path}")

    if not os.path.exists(config.train_data_path):
        alt_path = os.path.join(os.path.dirname(__file__), config.train_data_path)
        if os.path.exists(alt_path):
            config.train_data_path = alt_path
        else:
            logger.error(f"Training data not found: {config.train_data_path}")
            return

    dataset = RLHFDataset(config.train_data_path, getattr(trainer, "processor", None))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    logger.info(f"Dataset size: {len(dataset)}, batches: {len(dataloader)}")
    eval_loader = None
    if config.eval_data_path:
        if not os.path.exists(config.eval_data_path):
            alt_eval = os.path.join(os.path.dirname(__file__), config.eval_data_path)
            if os.path.exists(alt_eval):
                config.eval_data_path = alt_eval
            else:
                logger.warning(f"Eval data not found: {config.eval_data_path}")
        if os.path.exists(config.eval_data_path):
            eval_dataset = RLHFDataset(config.eval_data_path, getattr(trainer, "processor", None))
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            logger.info(f"Eval dataset size: {len(eval_dataset)}, batches: {len(eval_loader)}")

    logger.info("Starting adaptive training...")

    for epoch in range(config.num_epochs):
        trainer.epoch = epoch
        epoch_rewards = []
        epoch_losses = []

        for step, batch in enumerate(dataloader):
            try:
                log_dict = trainer.train_step(batch)

                if log_dict.get('should_stop'):
                    logger.info(f"Early stop triggered: {log_dict.get('stop_reason')}")
                    break

                epoch_rewards.append(log_dict.get('mean_reward', 0.0))
                epoch_losses.append(log_dict.get('total_loss', 0.0))

                if step % config.log_interval == 0:
                    log_msg = (
                        f"Epoch {epoch+1}/{config.num_epochs} | "
                        f"Step {step}/{len(dataloader)} | "
                        f"Global Step {trainer.global_step} | "
                        f"Reward: {log_dict.get('mean_reward', 0.0):.4f} | "
                        f"Policy Loss: {log_dict.get('policy_loss', 0.0):.4f} | "
                        f"Value Loss: {log_dict.get('value_loss', 0.0):.4f} | "
                        f"KL: {log_dict.get('kl_penalty', 0.0):.4f}"
                    )

                    if use_adaptive:
                        log_msg += (
                            f" | LR: {log_dict.get('adaptive_lr', 0):.2e} | "
                            f"Adaptive KL: {log_dict.get('adaptive_kl', 0):.4f}"
                        )

                    logger.info(log_msg)

                if trainer.global_step > 0 and trainer.global_step % config.save_interval == 0:
                    trainer.save_checkpoint(epoch, step)
                    logger.info(f"Checkpoint saved: epoch={epoch+1}, step={step}")

            except Exception as e:
                logger.error(f"Error at epoch {epoch}, step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

        status_msg = f"========== Epoch {epoch+1} completed =========="
        status_msg += f"\nAvg Reward: {avg_reward:.4f}"
        status_msg += f"\nAvg Loss: {avg_loss:.4f}"

        if use_adaptive:
            status = trainer.adaptive_controller.get_status()
            status_msg += f"\nBest Reward: {status['best_reward']:.4f}"
            status_msg += f"\nCurrent LR: {status['current_lr']:.2e}"
            status_msg += f"\nCurrent KL: {status['kl_coef']:.4f}"
            status_msg += f"\nCurrent Clip: {status['clip_epsilon']:.4f}"

        logger.info(status_msg)

        if eval_loader is not None:
            try:
                trainer.run_fixed_eval(eval_loader, tag=f"fixed_epoch{epoch+1}")
            except Exception as exc:
                logger.warning(f"Fixed eval failed: {exc}")

        trainer.save_checkpoint(epoch, 0)

        if step > 0 and len(dataloader) > 0:
            if step < len(dataloader) - 1:
                logger.info(f"Training stopped early at epoch {epoch+1}")
                break

    final_path = f"{config.output_dir}/final_model"
    if getattr(trainer, "actor_type", "qwen-vl") == "z-image":
        os.makedirs(final_path, exist_ok=True)
        save_final_full = os.getenv("Z_IMAGE_SAVE_FINAL_FULL", "").strip().lower() in ("1", "true", "yes", "y", "on")
        if save_final_full:
            try:
                trainer.z_image.save_checkpoint(final_path)
                logger.info(f"Training completed! Final model saved to {final_path}")
            except OSError as exc:
                logger.warning(f"Final full z-image save failed: {exc}")
                logger.info(f"Training completed! Final state kept under {final_path}")
        else:
            with open(os.path.join(final_path, "README.txt"), "w", encoding="utf-8") as f:
                f.write("Skipped full z-image save. Set Z_IMAGE_SAVE_FINAL_FULL=1 to enable.\n")
            logger.info(f"Training completed! Final state kept under {final_path}")
    else:  
        trainer.policy_model.save_pretrained(final_path)
        logger.info(f"Training completed! Final model saved to {final_path}")

    if use_adaptive:
        report_path = f"{config.output_dir}/training_report.txt"
        with open(report_path, 'w') as f:
            f.write("Adaptive training report\n")
            f.write("=" * 50 + "\n\n")

            status = trainer.adaptive_controller.get_status()
            f.write(f"Best reward: {status['best_reward']:.4f}\n")
            f.write(f"Total steps: {trainer.global_step}\n")
            f.write(f"Final LR: {status['current_lr']:.2e}\n")
            f.write(f"Final KL: {status['kl_coef']:.4f}\n")
            f.write(f"Final Clip: {status['clip_epsilon']:.4f}\n")
            f.write(f"Final Entropy: {status['entropy_coef']:.4f}\n")

            if hasattr(trainer, 'reward_optimizer') and trainer.reward_optimizer:
                f.write("\nReward weights:\n")
                f.write(f"  Human: {trainer.reward_optimizer.human_weight:.4f}\n")
                f.write(f"  Aesthetic: {trainer.reward_optimizer.aesthetic_weight:.4f}\n")

        logger.info(f"Training report saved to: {report_path}")


if __name__ == "__main__":
    main()






