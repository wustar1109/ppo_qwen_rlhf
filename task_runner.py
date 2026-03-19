"""
Task execution runner for the async-decoupled online loop.

A-route responsibilities:
- execute prompt tasks (generate -> judge -> repair -> retry)
- record trajectories
- avoid synchronous PPO optimizer updates
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class TaskExecutionRunner:
    """Runs task queue execution in inference/no-grad mode."""

    def __init__(self, trainer, config):
        self.trainer = trainer
        self.config = config

    @contextmanager
    def _inference_mode(self):
        prev_grad_enabled = torch.is_grad_enabled()
        backend = getattr(getattr(self.trainer, "z_image", None), "backend", None)
        old_enable_grad = None   

        try:
            torch.set_grad_enabled(False)

            if backend is not None and hasattr(backend, "enable_grad"):
                old_enable_grad = bool(getattr(backend, "enable_grad"))
                if old_enable_grad:
                    logger.info("Task mode forcing z-image backend into no-grad inference path.")
                backend.enable_grad = False

            pipe = getattr(backend, "pipe", None)
            if pipe is not None:
                for module_name in ("transformer", "text_encoder", "vae"):
                    module = getattr(pipe, module_name, None)
                    if module is not None and hasattr(module, "eval"):
                        module.eval()

            yield
        finally:
            if backend is not None and hasattr(backend, "enable_grad") and old_enable_grad is not None:
                backend.enable_grad = old_enable_grad
            torch.set_grad_enabled(prev_grad_enabled)

    def run(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not tasks:
            logger.warning("Task runner received empty task list.")
            return {
                "schema_version": getattr(self.trainer, "schema_version", "v1"),
                "run_id": getattr(self.trainer, "run_id", None),
                "total_tasks": 0,
                "passed_tasks": 0,
                "failed_tasks": 0,
                "task_results": [],
            }

        logger.info("Task execution mode started. total_tasks=%d", len(tasks))
        logger.info("A-route online loop is active; PPO train loop is disabled for this run.")

        with self._inference_mode():
            result = self.trainer.run_prompt_task_queue(tasks)

        logger.info(
            "Task execution mode finished. total=%s passed=%s failed=%s",
            result.get("total_tasks"),
            result.get("passed_tasks"),
            result.get("failed_tasks"),
        )
        return result
