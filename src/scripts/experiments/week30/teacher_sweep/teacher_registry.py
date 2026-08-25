from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HF_CACHE_ROOT = Path("/scratch/gilbreth/pham156/hf_cache")
MODEL_ORDER = ["none", "random", "qwen25", "biomistral", "meditron", "me_llama", "medgemma", "lingshu"]


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    source: str
    teacher_family: str
    display_name: str
    env_var: str | None
    local_patterns: tuple[str, ...]
    supports_text_encoding: bool = True
    supports_pseudotoken: bool = True


TEACHER_SPECS: dict[str, TeacherSpec] = {
    "none": TeacherSpec(
        key="none",
        source="none_current_init",
        teacher_family="none",
        display_name="No teacher",
        env_var=None,
        local_patterns=(),
        supports_text_encoding=False,
        supports_pseudotoken=False,
    ),
    "random": TeacherSpec(
        key="random",
        source="random_cohort_warmstart",
        teacher_family="random_control",
        display_name="Random cohort control",
        env_var=None,
        local_patterns=(),
        supports_text_encoding=False,
        supports_pseudotoken=False,
    ),
    "qwen25": TeacherSpec(
        key="qwen25",
        source="qwen25_teacher",
        teacher_family="general_llm",
        display_name="Qwen2.5",
        env_var="QWEN25_MODEL_PATH",
        local_patterns=("**/*Qwen2.5*", "**/*qwen2.5*", "**/*Qwen*Instruct*"),
    ),
    "biomistral": TeacherSpec(
        key="biomistral",
        source="biomistral_teacher",
        teacher_family="biomedical_text",
        display_name="BioMistral",
        env_var="BIOMISTRAL_MODEL_PATH",
        local_patterns=("**/*BioMistral*", "**/*biomistral*"),
    ),
    "meditron": TeacherSpec(
        key="meditron",
        source="meditron_teacher",
        teacher_family="clinical_text",
        display_name="Meditron",
        env_var="MEDITRON_MODEL_PATH",
        local_patterns=("**/*Meditron*", "**/*meditron*"),
    ),
    "me_llama": TeacherSpec(
        key="me_llama",
        source="me_llama_teacher",
        teacher_family="clinical_text",
        display_name="Medical LLaMA",
        env_var="ME_LLAMA_MODEL_PATH",
        local_patterns=("**/*medical*llama*", "**/*MedLLaMA*", "**/*medllama*"),
    ),
    "medgemma": TeacherSpec(
        key="medgemma",
        source="medgemma_teacher",
        teacher_family="medical_multimodal",
        display_name="MedGemma",
        env_var="MEDGEMMA_MODEL_PATH",
        local_patterns=("**/*MedGemma*", "**/*medgemma*"),
    ),
    "lingshu": TeacherSpec(
        key="lingshu",
        source="lingshu_teacher",
        teacher_family="medical_multimodal",
        display_name="Lingshu-7B",
        env_var="LINGSHU_MODEL_PATH",
        local_patterns=("**/Lingshu-7B", "**/*lingshu*"),
    ),
}


def iter_teacher_specs() -> Iterable[TeacherSpec]:
    return TEACHER_SPECS.values()


def get_teacher_spec(key: str) -> TeacherSpec:
    if key not in TEACHER_SPECS:
        raise KeyError(f"Unknown teacher key: {key}")
    return TEACHER_SPECS[key]


def _resolve_env_path(env_var: str | None) -> Path | None:
    if not env_var:
        return None
    value = os.getenv(env_var)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _find_first_match(patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        matches = sorted(HF_CACHE_ROOT.glob(pattern))
        for match in matches:
            if match.is_dir():
                candidates.append(match)
    if not candidates:
        return None

    def score(path: Path):
        has_config = (path / "config.json").exists()
        has_index = any(
            (path / name).exists()
            for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json", "pytorch_model.bin")
        )
        return (1 if has_config else 0, 1 if has_index else 0, len(path.parts))

    candidates = sorted(set(candidates), key=score, reverse=True)
    return candidates[0]


def resolve_teacher_path(spec: TeacherSpec) -> tuple[Path | None, str]:
    if spec.key in {"none", "random"}:
        return None, "control arm"
    env_path = _resolve_env_path(spec.env_var)
    if env_path is not None:
        return env_path, f"found via ${spec.env_var}"
    match = _find_first_match(spec.local_patterns)
    if match is not None:
        return match, "found in local HF cache"
    return None, "no local checkpoint found"


def teacher_availability_record(key: str) -> dict:
    spec = get_teacher_spec(key)
    path, reason = resolve_teacher_path(spec)
    available = key in {"none", "random"} or path is not None
    return {
        "teacher_model": spec.key,
        "source": spec.source,
        "teacher_family": spec.teacher_family,
        "display_name": spec.display_name,
        "available": available,
        "resolved_path": str(path) if path is not None else "",
        "reason": reason,
    }
