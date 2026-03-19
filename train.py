"""
PPO RLHF Training for Qwen Vision-Language Model.
Main entry point with explicit mode split:
- task execution mode (A-route)
- training mode (B-route PPO path, optional)
"""

import json
import logging
import os
from typing import Dict, Any

from torch.utils.data import DataLoader

from arguments import parse_args, get_config_from_args
from dataset import RLHFDataset, collate_fn
from learning_data import export_learning_datasets
from task_loader import load_prompt_tasks
from task_runner import TaskExecutionRunner
from trainer import AdaptivePPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return path
    alt_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(alt_path):
        return alt_path
    return path


def _log_config(config) -> None:
    logger.info("=" * 50)
    logger.info("PPO Qwen RLHF training config:")
    for key, value in vars(config).items():
        if not key.startswith("_"):
            logger.info("  %s: %s", key, value)
    logger.info("=" * 50)


def _run_task_execution_mode(args, config) -> Dict[str, Any]:
    if config.task_output_dir:
        config.eval_output_dir = config.task_output_dir

    config.task_mode_only = True
    logger.info("Task mode selected. Using async-decoupled A-route execution loop.")
    if not getattr(config, "task_disable_training", True):
        logger.warning(
            "task_disable_training=false is ignored in task mode; synchronous PPO training remains disabled by design."
        )

    if str(getattr(config, "actor_type", "")).lower() != "z-image":
        raise RuntimeError("Task execution mode currently requires --actor_type z-image")

    trainer = AdaptivePPOTrainer(config, use_adaptive=False)

    if args.resume:
        logger.info("Resuming z-image state from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    task_path = _resolve_path(config.task_excel_path)
    if not task_path or not os.path.exists(task_path):
        raise RuntimeError(f"Task file not found: {config.task_excel_path}")

    logger.info("Loading task queue from %s", task_path)
    tasks = load_prompt_tasks(
        task_path=task_path,
        sheet_name=config.task_sheet_name,
        start_row=config.task_start_row,
        end_row=config.task_end_row,
        default_max_retry=config.task_max_retry,
        default_target_score=config.task_pass_threshold,
    )
    logger.info("Task queue loaded. tasks=%d", len(tasks))

    runner = TaskExecutionRunner(trainer, config)
    result = runner.run(tasks)

    task_results = result.get("task_results", [])
    for item in task_results:
        logger.info(
            "Task summary prompt_id=%s final_score=%.4f execution_count=%s status=%s",
            item.get("prompt_id"),
            float(item.get("final_score") or 0.0),
            item.get("execution_count"),
            item.get("final_status"),
        )

    result_path = os.path.join(config.eval_output_dir, "task_queue_result.json")
    os.makedirs(config.eval_output_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Task queue result saved to %s", result_path)

    if getattr(config, "task_export_learning_data", False):
        export_summary = export_learning_datasets(
            eval_output_dir=config.eval_output_dir,
            output_dir=os.path.join(config.eval_output_dir, "learning_data"),
            run_id=trainer.run_id,
        )
        export_path = os.path.join(config.eval_output_dir, "learning_data", "export_summary.json")
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_summary, f, ensure_ascii=False, indent=2)
        logger.info("Offline learning data export saved to %s", export_path)

    return result


def _run_training_mode(args, config) -> None:
    use_adaptive = args.resume is None
    trainer = AdaptivePPOTrainer(config, use_adaptive=use_adaptive)

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    logger.info("Loading dataset from %s", config.train_data_path)

    train_path = _resolve_path(config.train_data_path)
    if not train_path or not os.path.exists(train_path):
        logger.error("Training data not found: %s", config.train_data_path)
        return
    config.train_data_path = train_path

    dataset = RLHFDataset(config.train_data_path, getattr(trainer, "processor", None))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    logger.info("Dataset size: %d, batches: %d", len(dataset), len(dataloader))

    eval_loader = None
    if config.eval_data_path:
        eval_path = _resolve_path(config.eval_data_path)
        if eval_path and os.path.exists(eval_path):
            config.eval_data_path = eval_path
            eval_dataset = RLHFDataset(config.eval_data_path, getattr(trainer, "processor", None))
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
            )
            logger.info("Eval dataset size: %d, batches: %d", len(eval_dataset), len(eval_loader))
        else:
            logger.warning("Eval data not found: %s", config.eval_data_path)

    logger.info("Starting adaptive training...")

    for epoch in range(config.num_epochs):
        trainer.epoch = epoch
        epoch_rewards = []
        epoch_losses = []

        for step, batch in enumerate(dataloader):
            try:
                log_dict = trainer.train_step(batch)

                if log_dict.get("should_stop"):
                    logger.info("Early stop triggered: %s", log_dict.get("stop_reason"))
                    break

                epoch_rewards.append(log_dict.get("mean_reward", 0.0))
                epoch_losses.append(log_dict.get("total_loss", 0.0))

                if step % config.log_interval == 0:
                    log_msg = (
                        f"Epoch {epoch + 1}/{config.num_epochs} | "
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
                    logger.info("Checkpoint saved: epoch=%s, step=%s", epoch + 1, step)

            except Exception as exc:
                logger.error("Error at epoch %s, step %s: %s", epoch, step, exc)
                import traceback

                traceback.print_exc()
                continue

        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        status_msg = f"========== Epoch {epoch + 1} completed =========="
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
                trainer.run_fixed_eval(eval_loader, tag=f"fixed_epoch{epoch + 1}")
            except Exception as exc:
                logger.warning("Fixed eval failed: %s", exc)

        trainer.save_checkpoint(epoch, 0)

        if step > 0 and len(dataloader) > 0 and step < len(dataloader) - 1:
            logger.info("Training stopped early at epoch %d", epoch + 1)
            break

    final_path = f"{config.output_dir}/final_model"
    if getattr(trainer, "actor_type", "qwen-vl") == "z-image":
        os.makedirs(final_path, exist_ok=True)
        save_final_full = os.getenv("Z_IMAGE_SAVE_FINAL_FULL", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if save_final_full:
            try:
                trainer.z_image.save_checkpoint(final_path)
                logger.info("Training completed! Final model saved to %s", final_path)
            except OSError as exc:
                logger.warning("Final full z-image save failed: %s", exc)
                logger.info("Training completed! Final state kept under %s", final_path)
        else:
            with open(os.path.join(final_path, "README.txt"), "w", encoding="utf-8") as f:
                f.write("Skipped full z-image save. Set Z_IMAGE_SAVE_FINAL_FULL=1 to enable.\n")
            logger.info("Training completed! Final state kept under %s", final_path)
    else:
        trainer.policy_model.save_pretrained(final_path)
        logger.info("Training completed! Final model saved to %s", final_path)

    if use_adaptive:
        report_path = f"{config.output_dir}/training_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Adaptive training report\n")
            f.write("=" * 50 + "\n\n")

            status = trainer.adaptive_controller.get_status()
            f.write(f"Best reward: {status['best_reward']:.4f}\n")
            f.write(f"Total steps: {trainer.global_step}\n")
            f.write(f"Final LR: {status['current_lr']:.2e}\n")
            f.write(f"Final KL: {status['kl_coef']:.4f}\n")
            f.write(f"Final Clip: {status['clip_epsilon']:.4f}\n")
            f.write(f"Final Entropy: {status['entropy_coef']:.4f}\n")

            if hasattr(trainer, "reward_optimizer") and trainer.reward_optimizer:
                f.write("\nReward weights:\n")
                f.write(f"  Human: {trainer.reward_optimizer.human_weight:.4f}\n")
                f.write(f"  Aesthetic: {trainer.reward_optimizer.aesthetic_weight:.4f}\n")

        logger.info("Training report saved to: %s", report_path)


def main():
    args = parse_args()
    config = get_config_from_args(args)

    _log_config(config)

    if config.task_excel_path:
        _run_task_execution_mode(args, config)
        return

    if getattr(config, "task_mode_only", False):
        logger.warning("task_mode_only=true but task_excel_path is empty; falling back to training mode.")

    _run_training_mode(args, config)


if __name__ == "__main__":
    main()

