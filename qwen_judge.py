"""
Qwen-VL judge adapter for image quality evaluation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


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
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


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
    ):
        self.model_name = model_name
        self.device = device
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.local_files_only = bool(local_files_only) if local_files_only is not None else os.path.isdir(model_name)
        self.trust_remote_code = bool(trust_remote_code)

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
            "Return JSON only with fields: scores {aesthetic, gray_smoothness, noise_artifact, prompt_alignment} (1-10), "
            "confidence (0-1), labels (list), critique, prompt_optimization."
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

    @torch.no_grad()
    def evaluate(self, image, prompt_text: str) -> Dict[str, Any]:
        text = self._build_prompt(prompt_text)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        do_sample = self.temperature is not None and self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
        )

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        parsed = _extract_json(decoded) or {}
        scores = parsed.get("scores", {}) if isinstance(parsed, dict) else {}

        result = {
            "raw_text": decoded,
            "scores": {
                "aesthetic": _coerce_score(scores.get("aesthetic")),
                "gray_smoothness": _coerce_score(scores.get("gray_smoothness")),
                "noise_artifact": _coerce_score(scores.get("noise_artifact")),
                "prompt_alignment": _coerce_score(scores.get("prompt_alignment")),
            },
            "confidence": _coerce_score(parsed.get("confidence")) if isinstance(parsed, dict) else None,
            "labels": _coerce_labels(parsed.get("labels")) if isinstance(parsed, dict) else [],
            "critique": parsed.get("critique") if isinstance(parsed, dict) else None,
            "prompt_optimization": parsed.get("prompt_optimization") if isinstance(parsed, dict) else None,
        }

        return result

    @torch.no_grad()
    def compare_pair(self, image_a, image_b, prompt_text: str) -> Dict[str, Any]:
        text = self._build_compare_prompt(prompt_text)
        inputs = self.processor(text=[text], images=[image_a, image_b], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        do_sample = self.temperature is not None and self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
        )

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        parsed = _extract_json(decoded) or {}

        result = {
            "raw_text": decoded,
            "better": parsed.get("better") if isinstance(parsed, dict) else None,
            "confidence": _coerce_score(parsed.get("confidence")) if isinstance(parsed, dict) else None,
            "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
            "labels": _coerce_labels(parsed.get("labels")) if isinstance(parsed, dict) else [],
        }
        return result

    def evaluate_batch(self, images: List[Any], prompts: List[str]) -> List[Dict[str, Any]]:
        results = []
        for img, prompt in zip(images, prompts):
            try:
                results.append(self.evaluate(img, prompt))
            except Exception as exc:
                logger.warning("Qwen judge failed: %s", exc)
                results.append({
                    "raw_text": None,
                    "scores": {},
                    "confidence": None,
                    "labels": [],
                    "critique": None,
                    "prompt_optimization": None,
                })
        return results

