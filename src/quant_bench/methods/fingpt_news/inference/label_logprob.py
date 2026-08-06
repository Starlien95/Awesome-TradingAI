from __future__ import annotations

import gc
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quant_bench.methods.fingpt_news.news.cleaner import make_prompt
from quant_bench.runtime.core.atomic_io import atomic_write_csv, safe_read_csv

LABELS = ["negative", "neutral", "positive"]
LABEL_SCORE = {"negative": -1, "neutral": 0, "positive": 1}
logger = logging.getLogger(__name__)


def prompt_key(instruction: str, input_text: str) -> str:
    payload = f"{instruction}\n{input_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_token_ids(tokenizer, prompt: str, label: str, max_length: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    max_prompt_tokens = max(max_length - len(label_ids), 1)
    prompt_ids = prompt_ids[:max_prompt_tokens]
    input_ids = prompt_ids + label_ids
    return input_ids, list(range(len(prompt_ids), len(input_ids)))


def scores_to_probs(scores: dict[str, float]) -> dict[str, float]:
    values = np.array([scores[label] for label in LABELS], dtype=float)
    values = values - np.nanmax(values)
    exp = np.exp(values)
    probs = exp / exp.sum()
    return {label: float(prob) for label, prob in zip(LABELS, probs, strict=False)}


def torch_cuda_snapshot() -> dict[str, float | int | bool]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        device_index = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        mib = 1024 * 1024
        return {
            "cuda_available": True,
            "device_index": int(device_index),
            "allocated_mib": round(torch.cuda.memory_allocated(device_index) / mib, 1),
            "reserved_mib": round(torch.cuda.memory_reserved(device_index) / mib, 1),
            "free_mib": round(free_bytes / mib, 1),
            "total_mib": round(total_bytes / mib, 1),
        }
    except Exception:
        return {"cuda_available": False}


class LabelLogprobSentimentService:
    def __init__(
        self,
        base_model_id: str,
        adapter_id: str,
        cache_path: str | Path,
        max_length: int = 512,
        batch_size: int = 8,
        dry_run: bool = False,
    ):
        self.base_model_id = base_model_id
        self.adapter_id = adapter_id
        self.cache_path = Path(cache_path)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.dry_run = bool(dry_run)
        self.tokenizer = None
        self.model = None
        self.cache = self._load_cache()

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def _load_cache(self) -> dict[str, dict]:
        df = safe_read_csv(self.cache_path)
        if df.empty:
            return {}
        return {str(row["prompt_key"]): row for row in df.to_dict("records")}

    def _write_cache(self) -> None:
        if self.cache:
            atomic_write_csv(pd.DataFrame(list(self.cache.values())), self.cache_path)

    def candidates_with_prompt_keys(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()
        work = candidates.copy()
        work["prompt_key"] = [
            prompt_key(row.instruction, row.input)
            for row in work.itertuples(index=False)
        ]
        return work

    def cache_stats(self, candidates: pd.DataFrame) -> dict[str, int]:
        if candidates.empty:
            return {"candidates": 0, "unique_prompts": 0, "cached": 0, "missing": 0}
        work = self.candidates_with_prompt_keys(candidates)
        unique_keys = set(work["prompt_key"].dropna().astype(str))
        cached = unique_keys & set(self.cache.keys())
        return {
            "candidates": len(work),
            "unique_prompts": len(unique_keys),
            "cached": len(cached),
            "missing": int(len(unique_keys) - len(cached)),
        }

    def load_model(self) -> None:
        if self.dry_run or self.model is not None:
            return
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.time()
        logger.info(
            "[FinGPT][model] loading base_model=%s adapter=%s cuda=%s",
            self.base_model_id,
            self.adapter_id,
            torch_cuda_snapshot(),
        )
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.adapter_id)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        logger.info(
            "[FinGPT][model] loaded in %.1fs cuda=%s",
            time.time() - started,
            torch_cuda_snapshot(),
        )

    def unload_model(self) -> None:
        if self.model is None and self.tokenizer is None:
            return
        started = time.time()
        logger.info("[FinGPT][model] unloading cuda_before=%s", torch_cuda_snapshot())
        model = self.model
        tokenizer = self.tokenizer
        self.model = None
        self.tokenizer = None
        del model
        del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        logger.info(
            "[FinGPT][model] unloaded in %.1fs cuda_after=%s",
            time.time() - started,
            torch_cuda_snapshot(),
        )

    def infer_candidates(
        self,
        candidates: pd.DataFrame,
        output_path: str | Path,
        allow_model_load: bool = True,
        allow_partial: bool = False,
        deadline_epoch: float | None = None,
        progress_every_batches: int = 10,
    ) -> pd.DataFrame:
        if candidates.empty:
            result = pd.DataFrame()
            atomic_write_csv(result, output_path)
            return result
        started = time.time()
        work = self.candidates_with_prompt_keys(candidates)
        missing = work[~work["prompt_key"].isin(self.cache.keys())].drop_duplicates("prompt_key")
        total_missing = len(missing)
        logger.info(
            "[FinGPT][infer] candidates=%d unique_prompts=%d cache_hits=%d missing=%d allow_model_load=%s",
            len(work),
            work["prompt_key"].nunique(),
            work["prompt_key"].nunique() - total_missing,
            total_missing,
            allow_model_load,
        )
        if self.dry_run:
            for row in missing.itertuples(index=False):
                self.cache[row.prompt_key] = self._neutral_cache_row(row.prompt_key)
        elif not missing.empty and not allow_model_load:
            logger.info("[FinGPT][infer] model load is disabled; writing cached sentiment rows only.")
        elif not missing.empty:
            self.load_model()
            rows = missing.to_dict("records")
            total_batches = int(np.ceil(len(rows) / max(self.batch_size, 1)))
            for start in range(0, len(rows), self.batch_size):
                if deadline_epoch is not None and time.time() >= deadline_epoch:
                    logger.warning(
                        "[FinGPT][infer] stopping before batch %d/%d because model window deadline was reached.",
                        start // self.batch_size + 1,
                        total_batches,
                    )
                    break
                batch = rows[start : start + self.batch_size]
                prompts = [make_prompt(row["instruction"], row["input"]) for row in batch]
                batch_scores = self._score_label_logprobs(prompts)
                for row, scores in zip(batch, batch_scores, strict=False):
                    probs = scores_to_probs(scores)
                    label = max(scores, key=scores.get)
                    self.cache[row["prompt_key"]] = self._cache_row(row["prompt_key"], scores, probs, label)
                self._write_cache()
                batch_index = start // self.batch_size + 1
                if batch_index == 1 or batch_index == total_batches or batch_index % max(progress_every_batches, 1) == 0:
                    logger.info(
                        "[FinGPT][infer] progress batch=%d/%d newly_scored=%d/%d elapsed=%.1fs",
                        batch_index,
                        total_batches,
                        min(start + len(batch), len(rows)),
                        len(rows),
                        time.time() - started,
                    )

        out_rows = []
        skipped_missing = 0
        for src in work.to_dict("records"):
            cached = self.cache.get(src["prompt_key"])
            if cached is None:
                if allow_partial:
                    skipped_missing += 1
                    continue
                raise KeyError(f"missing sentiment cache for prompt_key={src['prompt_key']}")
            out_rows.append(
                {
                    **{k: src.get(k, "") for k in [
                        "bucket",
                        "symbol",
                        "asset",
                        "date",
                        "datetime",
                        "news_id",
                        "raw_news_id",
                        "source",
                        "subject",
                        "title",
                        "url",
                        "relevance_weight",
                        "match_level",
                    ]},
                    **cached,
                    "base_model_id": self.base_model_id,
                    "adapter_id": self.adapter_id,
                    "inferred_at": pd.Timestamp.utcnow().isoformat(),
                }
            )
        result = pd.DataFrame(out_rows)
        atomic_write_csv(result, output_path)
        self._write_cache()
        logger.info(
            "[FinGPT][infer] output_rows=%d skipped_missing=%d total_elapsed=%.1fs",
            len(result),
            skipped_missing,
            time.time() - started,
        )
        return result

    def _score_label_logprobs(self, prompts: list[str]) -> list[dict[str, float]]:
        import torch

        tokenizer = self.tokenizer
        model = self.model
        candidate_inputs = []
        candidate_positions = []
        prompt_indices = []
        candidate_labels = []
        for prompt_idx, prompt in enumerate(prompts):
            for label in LABELS:
                ids, positions = candidate_token_ids(tokenizer, prompt, label, self.max_length)
                candidate_inputs.append(ids)
                candidate_positions.append(positions)
                prompt_indices.append(prompt_idx)
                candidate_labels.append(label)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(len(ids) for ids in candidate_inputs)
        input_tensor = torch.full((len(candidate_inputs), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(candidate_inputs), max_len), dtype=torch.long)
        for row_idx, ids in enumerate(candidate_inputs):
            input_tensor[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_idx, : len(ids)] = 1

        input_tensor = input_tensor.to(model.device)
        attention_mask = attention_mask.to(model.device)
        with torch.inference_mode():
            logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        scores = [{label: float("-inf") for label in LABELS} for _ in prompts]
        for row_idx, positions in enumerate(candidate_positions):
            token_scores = []
            for pos in positions:
                if pos == 0:
                    continue
                token_id = int(input_tensor[row_idx, pos].item())
                token_scores.append(float(log_probs[row_idx, pos - 1, token_id].item()))
            scores[prompt_indices[row_idx]][candidate_labels[row_idx]] = float(np.mean(token_scores)) if token_scores else float("-inf")
        return scores

    @staticmethod
    def _cache_row(key: str, scores: dict[str, float], probs: dict[str, float], label: str) -> dict:
        return {
            "prompt_key": key,
            "model_label": label,
            "model_score": LABEL_SCORE[label],
            "article_score": probs["positive"] - probs["negative"],
            "logprob_negative": scores["negative"],
            "logprob_neutral": scores["neutral"],
            "logprob_positive": scores["positive"],
            "prob_negative": probs["negative"],
            "prob_neutral": probs["neutral"],
            "prob_positive": probs["positive"],
        }

    @staticmethod
    def _neutral_cache_row(key: str) -> dict:
        return {
            "prompt_key": key,
            "model_label": "neutral",
            "model_score": 0,
            "article_score": 0.0,
            "logprob_negative": -1.0,
            "logprob_neutral": 0.0,
            "logprob_positive": -1.0,
            "prob_negative": 0.25,
            "prob_neutral": 0.5,
            "prob_positive": 0.25,
        }
