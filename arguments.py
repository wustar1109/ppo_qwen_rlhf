"""
Command line arguments.
"""

import argparse
from models import PPOConfig


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_tristate_bool(value):
    text = str(value).strip().lower()
    if text == "auto":
        return None
    return str2bool(text)


DEFAULT_QWEN_SYSTEM_PROMPT = (
    "You are a senior AIGC image visual designer and prompt engineer. "
    "Judge generated images with strict technical and aesthetic standards. "
    "Scoring semantics: noise_artifact means artifact severity where 1 is clean/artifact-free and 10 is severe artifacts/noise; higher is worse. "
    "You must diagnose defects and provide repair advice while preserving subject identity and style identity. "
    "Do NOT introduce new subjects, scenes, color themes, camera angles, or art styles. "
    "Return JSON only with fields: "
    "scores {aesthetic, gray_smoothness, noise_artifact, prompt_alignment} (1-10), "
    "confidence (0-1), labels (list), critique, prompt_optimization {"
    "protected_subject_tokens, protected_style_tokens, must_keep_phrases, "
    "rewrite_prompt_preserve_subject_style, append_constraints, forbidden_new_subject_tokens, reason"
    "}. "
    "Use labels from: noise, dirty_edge, broken_line, gray_band, local_collapse, prompt_mismatch, structure_collapse, severe_artifact."
)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Qwen RLHF Training")

    # Model config
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--use_aesthetic_reward", type=str2bool, default=True)
    parser.add_argument("--aesthetic_model_name", type=str, default="cafeai/cafe_aesthetic")
    parser.add_argument("--actor_type", type=str, default="z-image")
    parser.add_argument("--z_image_model_path", type=str, default=None)
    parser.add_argument("--allow_zimage_stub", type=str2bool, default=False)
    parser.add_argument("--reward_baseline_alpha", type=float, default=0.1)

    # Enhanced reward config
    parser.add_argument(
        "--use_enhanced_reward",
        type=str2bool,
        default=True,
        help="Enable enhanced reward model (image quality metrics)",
    )
    parser.add_argument("--gray_smoothness_weight", type=float, default=0.15)
    parser.add_argument("--line_smoothness_weight", type=float, default=0.15)
    parser.add_argument("--noise_artifact_weight", type=float, default=0.15)
    parser.add_argument("--human_weight", type=float, default=0.35)
    parser.add_argument("--aesthetic_weight", type=float, default=0.20)

    # Qwen judge config
    parser.add_argument("--use_qwen_judge", type=str2bool, default=False)
    parser.add_argument("--qwen_judge_model_name", type=str, default="Qwen/Qwen3-VL-7B-Instruct")
    parser.add_argument("--qwen_judge_device", type=str, default=None)
    parser.add_argument("--qwen_judge_local_files_only", type=parse_tristate_bool, default=None)
    parser.add_argument("--qwen_judge_trust_remote_code", type=str2bool, default=True)
    parser.add_argument("--qwen_judge_log_raw_output", type=str2bool, default=True)
    parser.add_argument("--qwen_judge_strict_schema", type=str2bool, default=False)
    parser.add_argument("--qwen_judge_retry_on_invalid", type=str2bool, default=True)
    parser.add_argument("--qwen_judge_max_retry", type=int, default=1)
    parser.add_argument("--qwen_judge_system_prompt", type=str, default=DEFAULT_QWEN_SYSTEM_PROMPT)
    parser.add_argument("--qwen_judge_prompt_version", type=str, default="v2_designer_guarded")
    parser.add_argument("--qwen_judge_max_new_tokens", type=int, default=512)
    parser.add_argument("--qwen_judge_temperature", type=float, default=0.2)
    parser.add_argument("--qwen_judge_top_p", type=float, default=0.9)
    parser.add_argument("--qwen_eval_every", type=int, default=50)
    parser.add_argument("--qwen_reward_weight", type=float, default=1.0)
    parser.add_argument("--qwen_aesthetic_weight", type=float, default=0.4)
    parser.add_argument("--qwen_gray_weight", type=float, default=0.4)
    parser.add_argument("--qwen_noise_weight", type=float, default=0.2)
    parser.add_argument("--qwen_prompt_alignment_weight", type=float, default=0.2)
    parser.add_argument("--qwen_confidence_low", type=float, default=0.3)
    parser.add_argument("--qwen_confidence_high", type=float, default=0.7)
    parser.add_argument("--qwen_gate_min_aesthetic", type=float, default=3.0)
    parser.add_argument("--qwen_gate_min_gray", type=float, default=3.0)
    parser.add_argument("--qwen_gate_min_prompt_alignment", type=float, default=3.0)
    parser.add_argument("--qwen_gate_max_noise", type=float, default=7.0)
    parser.add_argument("--qwen_gate_penalty", type=float, default=-1.0)
    parser.add_argument("--qwen_fatal_labels", type=str, default="structure_collapse,prompt_mismatch,severe_artifact")
    parser.add_argument("--qwen_pairwise_samples", type=int, default=1)
    parser.add_argument("--qwen_pairwise_every", type=int, default=50)
    parser.add_argument("--qwen_noise_higher_is_worse", type=str2bool, default=True)

    # Logging / outputs
    parser.add_argument("--enable_wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="z-image-rl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--eval_output_dir", type=str, default="./eval_output")
    parser.add_argument("--save_eval_every", type=int, default=50)
    parser.add_argument("--schema_version", type=str, default="v1")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--human_review_every", type=int, default=0)
    parser.add_argument("--human_disagreement_threshold", type=float, default=2.0)

    # Training config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # PPO config
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--value_clip", type=float, default=0.4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # Data config
    parser.add_argument("--train_data_path", type=str, default="data/train.json")
    parser.add_argument("--eval_data_path", type=str, default=None)

    # Task queue config (Excel/CSV closed-loop mode)
    parser.add_argument("--task_excel_path", type=str, default=None)
    parser.add_argument("--task_sheet_name", type=str, default=None)
    parser.add_argument("--task_max_retry", type=int, default=3)
    parser.add_argument("--task_pass_threshold", type=float, default=0.5)
    parser.add_argument("--task_use_row_threshold", type=str2bool, default=True)
    parser.add_argument("--task_start_row", type=int, default=1)
    parser.add_argument("--task_end_row", type=int, default=0)
    parser.add_argument("--task_save_best_only", type=str2bool, default=False)
    parser.add_argument("--task_enable_auto_repair", type=str2bool, default=True)
    parser.add_argument("--task_enable_negative_repair", type=str2bool, default=True)
    parser.add_argument("--task_enable_sampling_repair", type=str2bool, default=True)
    parser.add_argument("--task_continue_on_fail", type=str2bool, default=True)
    parser.add_argument("--task_mode_only", type=str2bool, default=False)
    parser.add_argument("--task_output_dir", type=str, default=None)
    parser.add_argument("--task_export_learning_data", type=str2bool, default=False)
    parser.add_argument("--task_disable_training", type=str2bool, default=True)
    parser.add_argument("--task_repair_low_score_max", type=float, default=3.0)
    parser.add_argument("--task_repair_mid_score_max", type=float, default=6.0)
    parser.add_argument("--task_prompt_identity_guard", type=str2bool, default=True)
    parser.add_argument("--task_prompt_identity_min_keep_ratio", type=float, default=0.6)
    parser.add_argument("--task_repair_use_original_anchor", type=str2bool, default=True)

    # Output config
    parser.add_argument("--output_dir", type=str, default="./ppo_qwen_output")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)

    # Device config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str2bool, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser.parse_args()


def get_config_from_args(args):
    """Create config from argparse namespace."""
    config = PPOConfig(
        model_name=args.model_name,
        reward_model_path=args.reward_model_path,
        use_aesthetic_reward=args.use_aesthetic_reward,
        aesthetic_model_name=args.aesthetic_model_name,

        actor_type=args.actor_type,
        z_image_model_path=args.z_image_model_path,
        allow_zimage_stub=args.allow_zimage_stub,
        reward_baseline_alpha=args.reward_baseline_alpha,

        use_enhanced_reward=args.use_enhanced_reward,
        gray_smoothness_weight=args.gray_smoothness_weight,
        line_smoothness_weight=args.line_smoothness_weight,
        noise_artifact_weight=args.noise_artifact_weight,
        human_weight=args.human_weight,
        aesthetic_weight=args.aesthetic_weight,

        use_qwen_judge=args.use_qwen_judge,
        qwen_judge_model_name=args.qwen_judge_model_name,
        qwen_judge_device=args.qwen_judge_device,
        qwen_judge_local_files_only=args.qwen_judge_local_files_only,
        qwen_judge_trust_remote_code=args.qwen_judge_trust_remote_code,
        qwen_judge_log_raw_output=args.qwen_judge_log_raw_output,
        qwen_judge_strict_schema=args.qwen_judge_strict_schema,
        qwen_judge_retry_on_invalid=args.qwen_judge_retry_on_invalid,
        qwen_judge_max_retry=args.qwen_judge_max_retry,
        qwen_judge_system_prompt=args.qwen_judge_system_prompt,
        qwen_judge_prompt_version=args.qwen_judge_prompt_version,
        qwen_judge_max_new_tokens=args.qwen_judge_max_new_tokens,
        qwen_judge_temperature=args.qwen_judge_temperature,
        qwen_judge_top_p=args.qwen_judge_top_p,
        qwen_eval_every=args.qwen_eval_every,
        qwen_reward_weight=args.qwen_reward_weight,
        qwen_aesthetic_weight=args.qwen_aesthetic_weight,
        qwen_gray_weight=args.qwen_gray_weight,
        qwen_noise_weight=args.qwen_noise_weight,
        qwen_prompt_alignment_weight=args.qwen_prompt_alignment_weight,
        qwen_confidence_low=args.qwen_confidence_low,
        qwen_confidence_high=args.qwen_confidence_high,
        qwen_gate_min_aesthetic=args.qwen_gate_min_aesthetic,
        qwen_gate_min_gray=args.qwen_gate_min_gray,
        qwen_gate_min_prompt_alignment=args.qwen_gate_min_prompt_alignment,
        qwen_gate_max_noise=args.qwen_gate_max_noise,
        qwen_gate_penalty=args.qwen_gate_penalty,
        qwen_fatal_labels=args.qwen_fatal_labels,
        qwen_pairwise_samples=args.qwen_pairwise_samples,
        qwen_pairwise_every=args.qwen_pairwise_every,
        qwen_noise_higher_is_worse=args.qwen_noise_higher_is_worse,

        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        eval_output_dir=args.eval_output_dir,
        save_eval_every=args.save_eval_every,
        schema_version=args.schema_version,
        run_id=args.run_id,
        human_review_every=args.human_review_every,
        human_disagreement_threshold=args.human_disagreement_threshold,

        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,

        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        value_clip=args.value_clip,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,

        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,

        task_excel_path=args.task_excel_path,
        task_sheet_name=args.task_sheet_name,
        task_max_retry=args.task_max_retry,
        task_pass_threshold=args.task_pass_threshold,
        task_use_row_threshold=args.task_use_row_threshold,
        task_start_row=args.task_start_row,
        task_end_row=args.task_end_row,
        task_save_best_only=args.task_save_best_only,
        task_enable_auto_repair=args.task_enable_auto_repair,
        task_enable_negative_repair=args.task_enable_negative_repair,
        task_enable_sampling_repair=args.task_enable_sampling_repair,
        task_continue_on_fail=args.task_continue_on_fail,
        task_mode_only=args.task_mode_only,
        task_output_dir=args.task_output_dir,
        task_export_learning_data=args.task_export_learning_data,
        task_disable_training=args.task_disable_training,
        task_repair_low_score_max=args.task_repair_low_score_max,
        task_repair_mid_score_max=args.task_repair_mid_score_max,
        task_prompt_identity_guard=args.task_prompt_identity_guard,
        task_prompt_identity_min_keep_ratio=args.task_prompt_identity_min_keep_ratio,
        task_repair_use_original_anchor=args.task_repair_use_original_anchor,

        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,

        device=args.device,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    return config
