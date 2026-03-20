"""
Qwen-VL judge adapter for image quality evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _coerce_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _dedupe_list(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _ensure_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _dedupe_list([str(v).strip() for v in value if v is not None])
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return _dedupe_list([part.strip() for part in text.split(",") if part.strip()])
        return [text]
    if isinstance(value, dict):
        out: List[str] = []
        for k in ("tokens", "items", "values", "list", "subject", "style"):
            if k in value:
                out.extend(_ensure_text_list(value.get(k)))
        return _dedupe_list(out)
    return []


def _first_text(value: Any, keys: List[str]) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in keys:
        v = value.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            nested = _first_text(v, ["text", "prompt", "value"])
            if nested:
                return nested
    return None


def _normalize_prompt_optimization(value: Any) -> Dict[str, Any]:
    normalized = {
        "protected_subject_tokens": [],
        "protected_style_tokens": [],
        "must_keep_phrases": [],
        "rewrite_prompt_preserve_subject_style": "",
        "append_constraints": "",
        "forbidden_new_subject_tokens": [],
        "reason": "",
    }

    if value is None:
        return normalized

    if isinstance(value, str):
        normalized["rewrite_prompt_preserve_subject_style"] = value.strip()
        return normalized

    if isinstance(value, list):
        rewrite_candidate = ""
        append_parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                item_norm = _normalize_prompt_optimization(item)
                if not rewrite_candidate and item_norm["rewrite_prompt_preserve_subject_style"]:
                    rewrite_candidate = item_norm["rewrite_prompt_preserve_subject_style"]
                if item_norm["append_constraints"]:
                    append_parts.append(item_norm["append_constraints"])
                normalized["protected_subject_tokens"].extend(item_norm["protected_subject_tokens"])
                normalized["protected_style_tokens"].extend(item_norm["protected_style_tokens"])
                normalized["must_keep_phrases"].extend(item_norm["must_keep_phrases"])
                normalized["forbidden_new_subject_tokens"].extend(item_norm["forbidden_new_subject_tokens"])
                if item_norm["reason"] and not normalized["reason"]:
                    normalized["reason"] = item_norm["reason"]
            elif isinstance(item, str) and item.strip():
                if not rewrite_candidate:
                    rewrite_candidate = item.strip()
                else:
                    append_parts.append(item.strip())

        normalized["rewrite_prompt_preserve_subject_style"] = rewrite_candidate
        normalized["append_constraints"] = "; ".join([p for p in append_parts if p]).strip()
        normalized["protected_subject_tokens"] = _dedupe_list(normalized["protected_subject_tokens"])
        normalized["protected_style_tokens"] = _dedupe_list(normalized["protected_style_tokens"])
        normalized["must_keep_phrases"] = _dedupe_list(normalized["must_keep_phrases"])
        normalized["forbidden_new_subject_tokens"] = _dedupe_list(normalized["forbidden_new_subject_tokens"])
        return normalized

    if not isinstance(value, dict):
        return normalized

    source = dict(value)
    if "prompt_optimization" in source and isinstance(source.get("prompt_optimization"), (dict, list, str)):
        nested = _normalize_prompt_optimization(source.get("prompt_optimization"))
        for k in normalized.keys():
            if isinstance(nested.get(k), list):
                normalized[k].extend(nested[k])
            elif isinstance(nested.get(k), str) and nested[k]:
                normalized[k] = nested[k]

    normalized["protected_subject_tokens"].extend(_ensure_text_list(
        source.get("protected_subject_tokens")
        or source.get("subject_tokens")
        or source.get("subject_keywords")
        or source.get("must_keep_subject_tokens")
    ))
    normalized["protected_style_tokens"].extend(_ensure_text_list(
        source.get("protected_style_tokens")
        or source.get("style_tokens")
        or source.get("style_keywords")
        or source.get("must_keep_style_tokens")
    ))
    normalized["must_keep_phrases"].extend(_ensure_text_list(
        source.get("must_keep_phrases")
        or source.get("keep_phrases")
        or source.get("must_keep")
        or source.get("protected_phrases")
    ))
    normalized["forbidden_new_subject_tokens"].extend(_ensure_text_list(
        source.get("forbidden_new_subject_tokens")
        or source.get("forbidden_tokens")
        or source.get("forbidden_new_subjects")
    ))

    rewrite = _first_text(source, [
        "rewrite_prompt_preserve_subject_style",
        "rewrite_prompt",
        "new_prompt",
        "optimized_prompt",
        "prompt",
        "text",
    ])
    append_constraints = _first_text(source, [
        "append_constraints",
        "constraints",
        "additional_constraints",
        "append",
    ])
    reason = _first_text(source, ["reason", "rationale", "analysis", "why"])

    if rewrite:
        normalized["rewrite_prompt_preserve_subject_style"] = rewrite
    if append_constraints:
        normalized["append_constraints"] = append_constraints
    if reason:
        normalized["reason"] = reason

    normalized["protected_subject_tokens"] = _dedupe_list(normalized["protected_subject_tokens"])
    normalized["protected_style_tokens"] = _dedupe_list(normalized["protected_style_tokens"])
    normalized["must_keep_phrases"] = _dedupe_list(normalized["must_keep_phrases"])
    normalized["forbidden_new_subject_tokens"] = _dedupe_list(normalized["forbidden_new_subject_tokens"])

    return normalized


def _extract_json_block(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    text = str(raw_text).strip()
    if not text:
        return None

    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fence_matches:
        block = block.strip()
        if block.startswith("{") and block.endswith("}"):
            return block

    if text.startswith("{") and text.endswith("}"):
        return text

    in_string = False
    escaped = False
    depth = 0
    start_idx = None

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                return text[start_idx: idx + 1]

    return None


def _safe_parse_judge_output(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str], Optional[str]]:
    if raw_text is None or str(raw_text).strip() == "":
        return None, False, "empty_response", None

    text = str(raw_text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, True, None, text
        return None, False, "json_root_not_object", text
    except Exception as exc:
        direct_error = str(exc)

    extracted = _extract_json_block(text)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed, True, None, extracted
            return None, False, "json_root_not_object", extracted
        except Exception as exc:
            return None, False, f"json_parse_failed: {exc}", extracted

    return None, False, f"json_parse_failed: {direct_error}", None


def _validate_schema(parsed: Optional[Dict[str, Any]], strict_schema: bool = False) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    normalized = {
        "scores": {
            "aesthetic": None,
            "gray_smoothness": None,
            "noise_artifact": None,
            "prompt_alignment": None,
        },
        "confidence": None,
        "labels": [],
        "critique": None,
        "prompt_optimization": _normalize_prompt_optimization(None),
    }

    if not isinstance(parsed, dict):
        return False, "parsed_payload_not_dict", normalized

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return False, "scores_missing_or_not_dict", normalized

    # Alias compatibility: artifact_severity -> noise_artifact
    if scores.get("noise_artifact") is None and scores.get("artifact_severity") is not None:
        scores = dict(scores)
        scores["noise_artifact"] = scores.get("artifact_severity")

    core_fields = ["aesthetic", "gray_smoothness", "noise_artifact", "prompt_alignment"]
    missing_or_invalid = []
    for field in core_fields:
        value = _coerce_score(scores.get(field))
        normalized["scores"][field] = value
        if value is None:
            missing_or_invalid.append(field)

    confidence = _coerce_score(parsed.get("confidence"))
    normalized["confidence"] = confidence
    if confidence is None:
        missing_or_invalid.append("confidence")

    normalized["labels"] = _coerce_labels(parsed.get("labels"))
    normalized["critique"] = parsed.get("critique") if parsed.get("critique") is not None else None
    normalized["prompt_optimization"] = _normalize_prompt_optimization(parsed.get("prompt_optimization"))

    if missing_or_invalid:
        return False, "missing_or_invalid_fields:" + ",".join(missing_or_invalid), normalized

    if strict_schema:
        score_range_invalid = []
        for field in core_fields:
            val = normalized["scores"][field]
            if val is None or val < 1 or val > 10:
                score_range_invalid.append(field)
        if score_range_invalid:
            return False, "score_out_of_range:" + ",".join(score_range_invalid), normalized
        if confidence is None or confidence < 0 or confidence > 1:
            return False, "confidence_out_of_range", normalized

    return True, None, normalized


class QwenVLJudge:
    def __init__(
        self,
        model_name: str,
        device: str,
        system_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        local_files_only: Optional[bool] = None,
        trust_remote_code: bool = True,
        log_raw_output: bool = True,
        strict_schema: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.local_files_only = bool(local_files_only) if local_files_only is not None else os.path.isdir(model_name)
        self.trust_remote_code = bool(trust_remote_code)
        self.log_raw_output = bool(log_raw_output)
        self.strict_schema = bool(strict_schema)

        self.model, self.processor = self._load_model()
        self.model.eval()

    def _processor_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda"):
            kwargs["device_map"] = "auto"
        if self.local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    def _load_model(self):
        processor_kwargs = self._processor_kwargs()
        model_kwargs = self._model_kwargs()

        prefer_auto = False
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name, **processor_kwargs)
            model_type = str(getattr(config, "model_type", "")).lower()
            if "qwen3" in model_type:
                prefer_auto = True
        except Exception:
            pass

        if not prefer_auto:
            try:
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

                processor = Qwen2VLProcessor.from_pretrained(self.model_name, **processor_kwargs)
                model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
                if not self.device.startswith("cuda"):
                    model.to(self.device)
                return model, processor
            except Exception as exc:
                logger.warning("Falling back to AutoModel for judge: %s", exc)

        from transformers import AutoProcessor, AutoModelForCausalLM

        model_classes = []
        try:
            from transformers import AutoModelForImageTextToText
            model_classes.append(AutoModelForImageTextToText)
        except Exception:
            pass

        try:
            from transformers import AutoModelForVision2Seq
            model_classes.append(AutoModelForVision2Seq)
        except Exception:
            pass

        model_classes.append(AutoModelForCausalLM)

        processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
        model = None
        last_exc = None
        for model_cls in model_classes:
            try:
                model = model_cls.from_pretrained(self.model_name, **model_kwargs)
                break
            except Exception as model_exc:
                last_exc = model_exc

        if model is None:
            raise RuntimeError(f"Failed to load judge model from {self.model_name}: {last_exc}")

        if not self.device.startswith("cuda"):
            model.to(self.device)
        return model, processor

    def _build_prompt(self, prompt_text: str) -> str:
        user_prompt = (
            "Prompt used to generate the image:\n"
            f"{prompt_text}\n\n"
            "Evaluate this image as a senior AIGC visual designer and prompt engineer. "
            "Important scoring semantics: noise_artifact means artifact severity where 1 is clean/artifact-free and 10 is severe artifacts/noise (higher is worse). "
            "Provide strict diagnostics and prompt repair suggestions while preserving subject/style identity. "
            "Do NOT introduce new subjects, scenes, color themes, camera angles, or art styles.\n\n"
            "Return JSON only with fields:\n"
            "scores {aesthetic, gray_smoothness, noise_artifact, prompt_alignment} (1-10),\n"
            "confidence (0-1), labels (list), critique,\n"
            "prompt_optimization {protected_subject_tokens, protected_style_tokens, must_keep_phrases, rewrite_prompt_preserve_subject_style, append_constraints, forbidden_new_subject_tokens, reason}."
        )
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.system_prompt + "\n\n" + user_prompt

    def _build_compare_prompt(self, prompt_text: str) -> str:
        user_prompt = (
            "Two images A and B were generated from the same prompt:\n"
            f"{prompt_text}\n\n"
            "Return JSON only with fields: better ('A'|'B'|'tie'), confidence (0-1), reason, labels."
        )
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.system_prompt + "\n\n" + user_prompt

    def _decode_generated_text(self, outputs, input_ids) -> str:
        try:
            if input_ids is not None and hasattr(input_ids, "shape"):
                in_len = int(input_ids.shape[-1])
                if outputs is not None and hasattr(outputs, "shape") and outputs.shape[-1] > in_len:
                    generated_ids = outputs[:, in_len:]
                    decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    if decoded:
                        return str(decoded[0])
        except Exception:
            pass

        decoded_full = self.processor.batch_decode(outputs, skip_special_tokens=True)
        if decoded_full:
            return str(decoded_full[0])
        return ""

    def _build_runtime_error_result(self, error: str) -> Dict[str, Any]:
        return {
            "raw_text": None,
            "raw_response_text": None,
            "scores": {
                "aesthetic": None,
                "gray_smoothness": None,
                "noise_artifact": None,
                "prompt_alignment": None,
            },
            "confidence": None,
            "labels": [],
            "critique": None,
            "prompt_optimization": _normalize_prompt_optimization(None),
            "parse_success": False,
            "parse_error": None,
            "schema_valid": False,
            "schema_error": None,
            "judge_status": "runtime_error",
            "judge_error": str(error),
            "image_input_confirmed": False,
        }

    @torch.no_grad()
    def evaluate(self, image, prompt_text: str) -> Dict[str, Any]:
        try:
            text = self._build_prompt(prompt_text)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
            image_input_confirmed = any(k in inputs for k in ("pixel_values", "image_grid_thw", "pixel_values_videos"))
            if not image_input_confirmed:
                logger.warning("Qwen judge processor did not produce image tensors; check multimodal input chain.")

            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            do_sample = self.temperature is not None and self.temperature > 0
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p

            outputs = self.model.generate(**inputs, **generate_kwargs)
            raw_text = self._decode_generated_text(outputs, inputs.get("input_ids"))

            if self.log_raw_output:
                logger.info("Qwen raw output: %s", raw_text if raw_text else "<EMPTY>")

            parsed, parse_success, parse_error, extracted_json = _safe_parse_judge_output(raw_text)

            if not parse_success:
                status = "empty_response" if parse_error == "empty_response" else "json_parse_failed"
                logger.warning("Qwen JSON parse failed: %s", parse_error)
                return {
                    "raw_text": raw_text,
                    "raw_response_text": raw_text,
                    "scores": {
                        "aesthetic": None,
                        "gray_smoothness": None,
                        "noise_artifact": None,
                        "prompt_alignment": None,
                    },
                    "confidence": None,
                    "labels": [],
                    "critique": None,
                    "prompt_optimization": _normalize_prompt_optimization(None),
                    "parse_success": False,
                    "parse_error": parse_error,
                    "schema_valid": False,
                    "schema_error": None,
                    "judge_status": status,
                    "judge_error": None,
                    "extracted_json_text": extracted_json,
                    "image_input_confirmed": image_input_confirmed,
                }

            schema_valid, schema_error, normalized = _validate_schema(parsed, strict_schema=self.strict_schema)
            if not schema_valid:
                logger.warning("Qwen schema validation failed: %s", schema_error)
                return {
                    "raw_text": raw_text,
                    "raw_response_text": raw_text,
                    "scores": normalized.get("scores", {}),
                    "confidence": normalized.get("confidence"),
                    "labels": normalized.get("labels", []),
                    "critique": normalized.get("critique"),
                    "prompt_optimization": normalized.get("prompt_optimization"),
                    "parse_success": True,
                    "parse_error": None,
                    "schema_valid": False,
                    "schema_error": schema_error,
                    "judge_status": "schema_invalid",
                    "judge_error": None,
                    "extracted_json_text": extracted_json,
                    "image_input_confirmed": image_input_confirmed,
                }

            return {
                "raw_text": raw_text,
                "raw_response_text": raw_text,
                "scores": normalized.get("scores", {}),
                "confidence": normalized.get("confidence"),
                "labels": normalized.get("labels", []),
                "critique": normalized.get("critique"),
                "prompt_optimization": normalized.get("prompt_optimization"),
                "parse_success": True,
                "parse_error": None,
                "schema_valid": True,
                "schema_error": None,
                "judge_status": "ok",
                "judge_error": None,
                "extracted_json_text": extracted_json,
                "image_input_confirmed": image_input_confirmed,
            }

        except Exception as exc:
            logger.warning("Qwen judge runtime error: %s", exc)
            return self._build_runtime_error_result(str(exc))

    @torch.no_grad()
    def compare_pair(self, image_a, image_b, prompt_text: str) -> Dict[str, Any]:
        try:
            text = self._build_compare_prompt(prompt_text)
            inputs = self.processor(text=[text], images=[image_a, image_b], return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            do_sample = self.temperature is not None and self.temperature > 0
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p

            outputs = self.model.generate(**inputs, **generate_kwargs)
            raw_text = self._decode_generated_text(outputs, inputs.get("input_ids"))
            parsed, parse_success, parse_error, _ = _safe_parse_judge_output(raw_text)
            parsed = parsed or {}

            result = {
                "raw_text": raw_text,
                "better": parsed.get("better") if isinstance(parsed, dict) else None,
                "confidence": _coerce_score(parsed.get("confidence")) if isinstance(parsed, dict) else None,
                "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "labels": _coerce_labels(parsed.get("labels")) if isinstance(parsed, dict) else [],
                "parse_success": parse_success,
                "parse_error": parse_error,
                "judge_status": "ok" if parse_success else "json_parse_failed",
            }
            return result
        except Exception as exc:
            return {
                "raw_text": None,
                "better": None,
                "confidence": None,
                "reason": None,
                "labels": [],
                "parse_success": False,
                "parse_error": str(exc),
                "judge_status": "runtime_error",
            }

    def evaluate_batch(self, images: List[Any], prompts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for img, prompt in zip(images, prompts):
            try:
                results.append(self.evaluate(img, prompt))
            except Exception as exc:
                logger.warning("Qwen judge fallback triggered: %s", exc)
                results.append(self._build_runtime_error_result(str(exc)))
        return results
