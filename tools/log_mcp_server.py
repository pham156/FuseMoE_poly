#!/usr/bin/env python3
"""Minimal MCP server for summarizing training/Slurm logs.

The server intentionally avoids external dependencies so VS Code can launch it
from a normal Python environment. It exposes request/response tools rather than
trying to push messages into the editor.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cuda_oom", re.compile(r"CUDA out of memory|CUDNN_STATUS_ALLOC_FAILED", re.I)),
    ("traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("runtime_error", re.compile(r"\b(RuntimeError|ValueError|AssertionError|KeyError|IndexError):")),
    ("nan_or_inf", re.compile(r"\b(?:nan|inf|NaN|Inf)\b")),
    ("killed", re.compile(r"\bKilled\b|oom-kill|Out Of Memory", re.I)),
    ("slurm", re.compile(r"slurmstepd|srun: error|SBATCH|CANCELLED|FAILED", re.I)),
    ("nccl", re.compile(r"\bNCCL\b|ProcessGroupNCCL|distributed", re.I)),
    ("cuda", re.compile(r"\bCUDA\b|CUBLAS|cuDNN|device-side assert", re.I)),
]

METRIC_PATTERN = re.compile(
    r"\b(loss|val_loss|accuracy|acc|epoch|step|grad|entropy|router|expert|lr)\b",
    re.I,
)


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def resolve_patterns(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        expanded = os.path.expanduser(os.path.expandvars(pattern))
        if not os.path.isabs(expanded):
            expanded = str(ROOT / expanded)
        for match in glob.glob(expanded, recursive=True):
            path = Path(match)
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def tail_lines(path: Path, max_lines: int) -> list[str]:
    lines: deque[str] = deque(maxlen=max_lines)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(ANSI_PATTERN.sub("", line.rstrip("\n")))
    return list(lines)


def classify(lines: list[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    categories: list[str] = []
    events: list[dict[str, Any]] = []
    metrics: deque[str] = deque(maxlen=20)

    for idx, line in enumerate(lines, start=1):
        if METRIC_PATTERN.search(line):
            metrics.append(line)
        for name, pattern in ERROR_PATTERNS:
            if pattern.search(line):
                if name not in categories:
                    categories.append(name)
                events.append({"line": idx, "category": name, "text": line[-500:]})

    return categories, events[-30:], list(metrics)


def scan_training_logs(args: dict[str, Any]) -> str:
    patterns = args.get("patterns") or ["out/Week_27/**/*.err", "out/Week_27/**/*.out"]
    max_files = int(args.get("max_files") or 8)
    tail = int(args.get("tail") or 250)
    context = int(args.get("context") or 80)

    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise ValueError("patterns must be a list of strings")

    paths = resolve_patterns(patterns)[:max_files]
    if not paths:
        return "No matching log files found."

    sections: list[str] = ["# Training Log Scan", ""]
    for path in paths:
        lines = tail_lines(path, tail)
        categories, events, metrics = classify(lines)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        mtime = path.stat().st_mtime

        sections.extend(
            [
                f"## {rel}",
                f"- bytes: {path.stat().st_size}",
                f"- mtime_epoch: {mtime:.0f}",
                f"- detected_categories: {', '.join(categories) if categories else 'none'}",
                "",
            ]
        )

        if events:
            sections.append("### Matched Events")
            for event in events:
                sections.append(f"- tail_line {event['line']} [{event['category']}]: {event['text']}")
            sections.append("")

        if metrics:
            sections.append("### Recent Metric-Like Lines")
            sections.extend(f"- {line[-500:]}" for line in metrics)
            sections.append("")

        interesting_indices = {event["line"] - 1 for event in events}
        if interesting_indices:
            start = max(0, max(interesting_indices) - context)
            excerpt = lines[start:]
        else:
            excerpt = lines[-min(context, len(lines)) :]

        sections.append("### Relevant Tail")
        sections.append("```text")
        sections.extend(excerpt[-context:])
        sections.append("```")
        sections.append("")

    return "\n".join(sections)


def read_log_excerpt(args: dict[str, Any]) -> str:
    path_arg = args.get("path")
    if not isinstance(path_arg, str) or not path_arg:
        raise ValueError("path is required")
    tail = int(args.get("tail") or 200)
    path = Path(os.path.expanduser(os.path.expandvars(path_arg)))
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    lines = tail_lines(path, tail)
    return f"# Log Excerpt: {rel}\n\n```text\n" + "\n".join(lines) + "\n```"


TOOLS = {
    "scan_training_logs": {
        "description": "Scan recent Slurm/training .err/.out logs and return compact failure context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns relative to the repository root.",
                },
                "max_files": {"type": "integer", "default": 8},
                "tail": {"type": "integer", "default": 250},
                "context": {"type": "integer", "default": 80},
            },
        },
        "handler": scan_training_logs,
    },
    "read_log_excerpt": {
        "description": "Read the last N lines of one log file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "tail": {"type": "integer", "default": 200},
            },
            "required": ["path"],
        },
        "handler": read_log_excerpt,
    },
}


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        return response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fusemoe-log-watcher", "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return response(
            request_id,
            {
                "tools": [
                    {
                        "name": name,
                        "description": spec["description"],
                        "inputSchema": spec["inputSchema"],
                    }
                    for name, spec in TOOLS.items()
                ]
            },
        )
    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in TOOLS:
            return error_response(request_id, -32602, f"unknown tool: {name}")
        try:
            text = TOOLS[name]["handler"](arguments)
            return response(request_id, {"content": [{"type": "text", "text": text}]})
        except Exception as exc:  # MCP clients should receive tool failures as errors.
            return error_response(request_id, -32000, str(exc))
    if request_id is not None:
        return error_response(request_id, -32601, f"unknown method: {method}")
    return None


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="Repository root; reserved for VS Code config clarity.")
    args = parser.parse_args()
    ROOT = Path(args.root).expanduser().resolve()

    while True:
        request = read_message()
        if request is None:
            return 0
        result = handle(request)
        if result is not None:
            write_message(result)


if __name__ == "__main__":
    raise SystemExit(main())
