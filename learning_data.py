"""
Offline learning-data export utilities (B-route).

B-route responsibilities:
- consume A-route trajectory artifacts
- produce datasets for prompt optimizer / reward model / preference model
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _filter_run(records: Iterable[Dict[str, Any]], run_id: Optional[str]) -> List[Dict[str, Any]]:
    if not run_id:
        return list(records)
    return [r for r in records if str(r.get("run_id") or "") == str(run_id)]


def _build_prompt_optimizer_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        repair_action = rec.get("repair_action") if isinstance(rec.get("repair_action"), dict) else {}
        if not repair_action:
            continue

        old_prompt = str(repair_action.get("old_prompt") or rec.get("used_prompt") or rec.get("current_prompt") or "")
        new_prompt = str(repair_action.get("new_prompt") or old_prompt)
        old_negative = str(repair_action.get("old_negative_prompt") or rec.get("used_negative_prompt") or "")
        new_negative = str(repair_action.get("new_negative_prompt") or old_negative)
        old_sampling = repair_action.get("old_sampling") if isinstance(repair_action.get("old_sampling"), dict) else {}
        new_sampling = repair_action.get("new_sampling") if isinstance(repair_action.get("new_sampling"), dict) else {}

        changed = (new_prompt != old_prompt) or (new_negative != old_negative) or (new_sampling != old_sampling)
        if not changed:
            continue

        rows.append({
            "schema_version": rec.get("schema_version"),
            "run_id": rec.get("run_id"),
            "task_id": rec.get("task_id"),
            "prompt_id": rec.get("prompt_id"),
            "row_index": rec.get("row_index"),
            "attempt_index": rec.get("attempt_index"),
            "input_prompt": old_prompt,
            "input_negative_prompt": old_negative,
            "labels": rec.get("labels", []),
            "critique": rec.get("critique"),
            "fail_reasons": rec.get("fail_reasons", []),
            "reward_before": rec.get("qwen_reward"),
            "target_prompt": new_prompt,
            "target_negative_prompt": new_negative,
            "target_sampling": new_sampling,
            "repair_action": repair_action,
            "timestamp": rec.get("timestamp"),
        })
    return rows


def _build_reward_model_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append({
            "schema_version": rec.get("schema_version"),
            "run_id": rec.get("run_id"),
            "task_id": rec.get("task_id"),
            "prompt_id": rec.get("prompt_id"),
            "attempt_index": rec.get("attempt_index"),
            "prompt": rec.get("used_prompt") or rec.get("current_prompt"),
            "negative_prompt": rec.get("used_negative_prompt") or rec.get("current_negative_prompt"),
            "image_path": rec.get("image_path"),
            "scores": rec.get("qwen_scores", {}),
            "target_reward": rec.get("qwen_reward"),
            "confidence": rec.get("confidence"),
            "labels": rec.get("labels", []),
            "passed": rec.get("passed"),
            "fail_reasons": rec.get("fail_reasons", []),
            "timestamp": rec.get("timestamp"),
        })
    return rows


def _build_preference_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = str(rec.get("task_id") or rec.get("prompt_id") or "")
        if not key:
            continue
        grouped[key].append(rec)

    rows: List[Dict[str, Any]] = []
    for key, attempts in grouped.items():
        if len(attempts) < 2:
            continue
        for left, right in combinations(attempts, 2):
            l_reward = float(left.get("qwen_reward") or 0.0)
            r_reward = float(right.get("qwen_reward") or 0.0)
            if l_reward == r_reward:
                continue

            if l_reward > r_reward:
                winner, loser = left, right
                reward_gap = l_reward - r_reward
            else:
                winner, loser = right, left
                reward_gap = r_reward - l_reward

            rows.append({
                "schema_version": winner.get("schema_version") or loser.get("schema_version"),
                "run_id": winner.get("run_id") or loser.get("run_id"),
                "task_id": key,
                "prompt_id": winner.get("prompt_id") or loser.get("prompt_id"),
                "winner_attempt": winner.get("attempt_index"),
                "loser_attempt": loser.get("attempt_index"),
                "prompt": winner.get("used_prompt") or winner.get("current_prompt"),
                "winner_image_path": winner.get("image_path"),
                "loser_image_path": loser.get("image_path"),
                "winner_reward": winner.get("qwen_reward"),
                "loser_reward": loser.get("qwen_reward"),
                "reward_gap": reward_gap,
                "winner_labels": winner.get("labels", []),
                "loser_labels": loser.get("labels", []),
                "winner_scores": winner.get("qwen_scores", {}),
                "loser_scores": loser.get("qwen_scores", {}),
            })
    return rows


def export_learning_datasets(
    eval_output_dir: str,
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export offline-learning datasets from task trajectory records."""
    if not eval_output_dir:
        raise RuntimeError("eval_output_dir is required")

    source_path = os.path.join(eval_output_dir, "task_run_records.jsonl")
    records = _filter_run(_iter_jsonl(source_path), run_id)

    if output_dir is None:
        output_dir = os.path.join(eval_output_dir, "learning_data")
    os.makedirs(output_dir, exist_ok=True)

    prompt_rows = _build_prompt_optimizer_rows(records)
    reward_rows = _build_reward_model_rows(records)
    pref_rows = _build_preference_rows(records)

    prompt_path = os.path.join(output_dir, "prompt_optimizer_train.jsonl")
    reward_path = os.path.join(output_dir, "reward_model_train.jsonl")
    pref_path = os.path.join(output_dir, "preference_train.jsonl")

    prompt_count = _write_jsonl(prompt_path, prompt_rows)
    reward_count = _write_jsonl(reward_path, reward_rows)
    pref_count = _write_jsonl(pref_path, pref_rows)

    summary = {
        "run_id": run_id,
        "source_path": source_path,
        "output_dir": output_dir,
        "prompt_optimizer_samples": prompt_count,
        "reward_model_samples": reward_count,
        "preference_samples": pref_count,
        "prompt_optimizer_path": prompt_path,
        "reward_model_path": reward_path,
        "preference_path": pref_path,
    }

    logger.info(
        "Exported learning datasets: prompt_optimizer=%d reward=%d preference=%d",
        prompt_count,
        reward_count,
        pref_count,
    )
    return summary
