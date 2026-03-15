#!/usr/bin/env python3
"""Pre-pull SWE-bench Docker images for a tasks file.

Why:
- mini-swe-agent starts a fresh Docker container per task.
- `docker run -d ...` may block while pulling the image; on macOS (especially Apple
  Silicon running x86_64 images) this can exceed the default startup timeout.

This script reads a tasks file (JSON array, JSON object, or JSONL) and pulls the
required images ahead of time.

Examples:
  python prepull_swebench_images.py --tasks tasks_50.json --max_workers 4
  python prepull_swebench_images.py --tasks tasks.jsonl --platform linux/amd64
  python prepull_swebench_images.py --tasks tasks_50.json --print_images

Environment:
  - MSWEA_DOCKER_EXECUTABLE: docker executable to use (default: docker)

Exit codes:
  0: all pulls succeeded (or were skipped)
  1: at least one pull failed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _load_tasks(tasks_path: Path) -> list[Any]:
    content = tasks_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    # JSON array or single JSON object
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        # JSON Lines
        tasks: list[Any] = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {tasks_path}: {e}") from e
        return tasks


def _as_instance_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"instance_id": item}
    if isinstance(item, dict):
        # Common fields across formats
        if "instance_id" in item:
            return item
        if "task_id" in item:
            copied = dict(item)
            copied["instance_id"] = str(item["task_id"])
            return copied
        if "id" in item:
            copied = dict(item)
            copied["instance_id"] = str(item["id"])
            return copied
        return item
    raise TypeError(f"Unsupported task item type: {type(item).__name__}")


def get_swebench_docker_image_name(instance: dict[str, Any]) -> str:
    """Mirrors mini-swe-agent logic for deriving the SWE-bench image name."""
    image_name = instance.get("image_name")
    if image_name:
        return str(image_name)

    iid = instance.get("instance_id")
    if not iid:
        raise ValueError("Task item is missing instance_id (or task_id/id)")

    # Docker doesn't allow double underscore; SWE-bench images use _1776_ as a magic token.
    id_docker_compatible = str(iid).replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


def _default_platform_arg() -> str | None:
    # SWE-bench images are x86_64; on Apple Silicon, pulling with --platform avoids
    # accidental arm64 pulls / unexpected behavior.
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "linux/amd64"
    return None


@dataclass(frozen=True)
class PullResult:
    image: str
    status: str  # pulled | skipped | failed
    output: str = ""


def _image_exists(docker: str, image: str) -> bool:
    proc = subprocess.run(
        [docker, "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def _pull_one(
    *,
    docker: str,
    image: str,
    platform_arg: str | None,
    timeout_s: int,
    force: bool,
) -> PullResult:
    if not force and _image_exists(docker, image):
        return PullResult(image=image, status="skipped")

    cmd: list[str] = [docker, "pull"]
    if platform_arg:
        cmd.extend(["--platform", platform_arg])
    cmd.append(image)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return PullResult(image=image, status="failed", output=f"timeout after {timeout_s}s")

    if proc.returncode == 0:
        return PullResult(image=image, status="pulled", output=proc.stdout.strip())

    combined = (proc.stdout or "") + ("\n" if proc.stderr else "") + (proc.stderr or "")
    return PullResult(image=image, status="failed", output=combined.strip()[:4000])


def _unique(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-pull SWE-bench Docker images for a tasks file")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON/JSONL")
    parser.add_argument("--max_workers", type=int, default=4, help="Parallel pulls")
    parser.add_argument("--timeout_s", type=int, default=3600, help="Timeout per docker pull")
    parser.add_argument(
        "--platform",
        default="auto",
        help="Docker platform to pull for (e.g. linux/amd64). Use 'auto' (default) or 'none' to omit.",
    )
    parser.add_argument("--force", action="store_true", help="Pull even if image already exists locally")
    parser.add_argument("--dry_run", action="store_true", help="Only print images, do not pull")
    parser.add_argument("--print_images", action="store_true", help="Print computed images and exit")

    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_path}", file=sys.stderr)
        return 1

    docker = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")

    platform_arg: str | None
    if args.platform == "auto":
        platform_arg = _default_platform_arg()
    elif args.platform == "none":
        platform_arg = None
    else:
        platform_arg = args.platform

    raw_items = _load_tasks(tasks_path)
    instances = [_as_instance_dict(item) for item in raw_items]
    images = _unique(get_swebench_docker_image_name(inst) for inst in instances)

    if args.print_images or args.dry_run:
        for img in images:
            print(img)
        return 0

    if not images:
        print("No tasks found; nothing to pull.")
        return 0

    print(f"Found {len(images)} unique SWE-bench images.")
    if platform_arg:
        print(f"Pull platform: {platform_arg}")

    failures: list[PullResult] = []
    pulled = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        fut_to_img = {
            ex.submit(
                _pull_one,
                docker=docker,
                image=img,
                platform_arg=platform_arg,
                timeout_s=max(1, args.timeout_s),
                force=bool(args.force),
            ): img
            for img in images
        }

        for fut in as_completed(fut_to_img):
            res = fut.result()
            if res.status == "pulled":
                pulled += 1
                print(f"pulled  {res.image}")
            elif res.status == "skipped":
                skipped += 1
                print(f"skipped {res.image}")
            else:
                failures.append(res)
                print(f"FAILED  {res.image}\n{res.output}\n", file=sys.stderr)

    print(f"Done. pulled={pulled} skipped={skipped} failed={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
