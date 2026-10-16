"""Small, auditable ABC-130K visual segmentation smoke harness.

Run with the selected provider's API key set. Source is read-only; output stays under result/runs.
The output is deliberately marked needs_review until a person accepts labels.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import html
import json
import math
import os
import re
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from PIL import Image, ImageDraw


SOURCE = Path("/home/data/tmp/open_data/ABC-130K/lerobot_v3_train")
PROJECT = Path(__file__).resolve().parents[1]
CAMERAS = ("top", "left_wrist", "right_wrist")
FPS = 30
TASK_INDEX = 54
VAL_IDS = {76, 78, 202, 329, 585, 711, 712, 839, 1092, 1094, 1221,
           1600, 2234, 2361, 2364, 2489, 2618, 2743, 2869, 3125}
SYSTEM_PROMPT = """You annotate a real robot demonstration from synchronized top, left wrist, and right wrist images. Use only visible evidence. Do not infer success, hidden motion, or which arm acted when unclear. The manipulated objects in this task are plastic bottles; do not rename a crumpled bottle as a bag. If object identity is unclear, say object. Return valid JSON only. Write short English imperative robot instructions with ONE atomic visible action per instruction. Separate reach, grasp, lift, transport, release, and retract when transitions are visible. The overall task is putting plastic bottles in a bin, but each instruction MUST describe the local visible action, not repeat the overall task."""
USER_PROMPT = """Source episode {episode}; video is 30 fps. The contact sheets sample every {sample_stride_frames} source frames and each row has a source frame ID. This window is [{start}, {end}) source frames. Each tile shows TOP, LEFT WRIST, RIGHT WRIST at the same frame. Split the whole window into consecutive visible action phases. Include idle/recovery if present. Use the displayed frame IDs to estimate boundaries. Do not use one phase to cover a visible reach, grasp, carry, and release sequence. Split when the visible action changes; do not force splits if it does not change. The first start_frame MUST be {start}, the last end_frame MUST be {end}, and phases MUST be contiguous with no gaps. Each phase should be at least about 1 second; if one action lasts longer than 12 seconds, keep it as one phase here; the harness will split its duration without inventing an action. Evidence must cite a camera and visible frame ID, not just 'visible motion'. Return only JSON of this shape: {{"phases":[{{"start_frame":0,"end_frame":100,"instruction_en":"Reach toward the plastic bottle.","evidence":"top camera F30 shows gripper approaching bottle","confidence":"high|medium|low"}}]}}. No markdown."""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def boxes(handle, start: int, end: int):
    handle.seek(start)
    while handle.tell() + 8 <= end:
        pos = handle.tell()
        header = handle.read(8)
        if len(header) < 8:
            break
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            size = struct.unpack(">Q", handle.read(8))[0]
            header_size = 16
        if size == 0:
            size = end - pos
        if size < header_size or pos + size > end:
            break
        yield pos, size, kind, header_size
        handle.seek(pos + size)


def mp4_frames(path: Path) -> int:
    """Read MP4 stsz sample count without decoding the entire video."""
    with path.open("rb") as f:
        top = list(boxes(f, 0, path.stat().st_size))
        moov = next((b for b in top if b[2] == b"moov"), None)
        if moov is None:
            raise ValueError(f"MP4 has no moov: {path}")
        current = moov
        for kind in (b"trak", b"mdia", b"minf", b"stbl", b"stsz"):
            current = next((b for b in boxes(f, current[0] + current[3],
                                             current[0] + current[1]) if b[2] == kind), None)
            if current is None:
                raise ValueError(f"MP4 has no {kind!r}: {path}")
        f.seek(current[0] + current[3] + 8)
        count = struct.unpack(">I", f.read(4))[0]
        if count <= 0:
            raise ValueError(f"MP4 has zero samples: {path}")
        return count


def source_video_map(global_end: int) -> dict[str, list[dict]]:
    result = {}
    for camera in CAMERAS:
        files = []
        cursor = 0
        while cursor < global_end:
            k = len(files)
            path = SOURCE / f"videos/observation.images.{camera}/chunk-{k // 1000:03d}/file-{k % 1000:03d}.mp4"
            count = mp4_frames(path)
            files.append({"path": str(path), "global_start": cursor, "global_end": cursor + count})
            cursor += count
        result[camera] = files
    return result


def locate(files: list[dict], global_start: int, count: int) -> dict:
    i = bisect.bisect_right([x["global_start"] for x in files], global_start) - 1
    if i < 0 or global_start + count > files[i]["global_end"]:
        raise ValueError(f"Episode crosses MP4 boundary at global frame {global_start}")
    return {"path": files[i]["path"], "file_start_frame": global_start - files[i]["global_start"]}


def load_episode(ep: int) -> tuple[pa.Table, int]:
    pf = pq.ParquetFile(SOURCE / f"data/chunk-{ep // 127 // 1000:03d}/file-{ep // 127 % 1000:03d}.parquet")
    table = pf.read_row_group(ep % 127)
    n = table.num_rows
    if n < 30 or table["episode_index"][0].as_py() != ep:
        raise ValueError(f"Unexpected row group for episode {ep}")
    start = table["index"][0].as_py()
    frames = table["frame_index"].to_pylist()
    tasks = table["task_index"].to_pylist()
    times = table["timestamp"].to_pylist()
    if frames != list(range(n)) or set(tasks) != {TASK_INDEX}:
        raise ValueError(f"Frame or task mismatch for episode {ep}")
    if any(abs(t - i / FPS) > 0.0002 for i, t in enumerate(times)):
        raise ValueError(f"Timestamp mismatch for episode {ep}")
    return table, start


def run_ffmpeg(args: list[str]) -> None:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-700:]}")


def make_sheets(ep: int, win_start: int, win_end: int, cameras: dict,
                output: Path, sample_stride_frames: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    sample_count = math.ceil((win_end - win_start) / sample_stride_frames)
    with tempfile.TemporaryDirectory() as tmp:
        images = {}
        for camera in CAMERAS:
            folder = Path(tmp) / camera
            folder.mkdir()
            src = cameras[camera]
            seek = (src["file_start_frame"] + win_start) / FPS
            run_ffmpeg(["-ss", f"{seek:.9f}", "-i", src["path"], "-vf",
                        f"select=not(mod(n\\,{sample_stride_frames}))", "-vsync", "vfr", "-frames:v",
                        str(sample_count), "-q:v", "3", str(folder / "%03d.jpg")])
            paths = sorted(folder.glob("*.jpg"))
            if len(paths) != sample_count:
                raise ValueError(f"{camera} sampling got {len(paths)}, expected {sample_count}")
            images[camera] = paths
        sheets = []
        for batch_start in range(0, sample_count, 12):
            batch_end = min(batch_start + 12, sample_count)
            canvas = Image.new("RGB", (1360, 792), "#f5f5f5")
            draw = ImageDraw.Draw(canvas)
            for j in range(batch_start, batch_end):
                tile = j - batch_start
                x = (tile % 2) * 680
                y = (tile // 2) * 132
                frame = win_start + j * sample_stride_frames
                draw.text((x + 3, y + 2), f"EP{ep} F{frame} T{frame/FPS:.2f}s", fill="black")
                for camera_index, camera in enumerate(CAMERAS):
                    with Image.open(images[camera][j]) as im:
                        canvas.paste(im.convert("RGB").resize((220, 105)),
                                     (x + camera_index * 225, y + 21))
            path = output / f"window_{win_start:05d}_{batch_start // 12:02d}.jpg"
            canvas.save(path, quality=85)
            sheets.append(path)
        return sheets


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def call_model(client: OpenAI, ep: int, start: int, end: int,
               sheets: list[Path], requests_dir: Path,
               sample_stride_frames: int, model: str,
               extra_body: dict) -> list[dict]:
    user_text = USER_PROMPT.format(episode=ep, start=start, end=end,
                                   sample_stride_frames=sample_stride_frames)
    content = [{"type": "text", "text": user_text}]
    for path in sheets:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    request_record = {"episode": ep, "start": start, "end": end,
                      "model": model, "sheets": [str(p) for p in sheets],
                      "sample_stride_frames": sample_stride_frames,
                      "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + user_text).encode()).hexdigest()}
    write_json(requests_dir / f"ep{ep}_{start:05d}_request.json", request_record)
    last_error = ""
    for attempt in range(2):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}]
            if last_error:
                messages.append({"role": "user", "content":
                                 f"Your previous response failed validation: {last_error}. Return exact valid JSON and contiguous phases."})
            response = client.chat.completions.create(
                model=model, messages=messages,
                stream=False, extra_body=extra_body, max_tokens=2400,
                timeout=90)
            answer = response.choices[0].message.content or ""
            write_json(requests_dir / f"ep{ep}_{start:05d}_raw_{attempt + 1}.json",
                       {"response_id": response.id, "finish_reason": response.choices[0].finish_reason,
                        "content": answer, "usage": response.usage.model_dump() if response.usage else None})
            parsed = parse_json(answer)
            phases = parsed["phases"]
            repair_small_boundary_gaps(phases, start, end)
            validate_phases(phases, start, end)
            write_json(requests_dir / f"ep{ep}_{start:05d}_response.json",
                       {"response_id": response.id, "content": answer,
                        "attempt": attempt + 1})
            return phases
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            write_json(requests_dir / f"ep{ep}_{start:05d}_error_{attempt + 1}.json",
                       {"error_type": type(exc).__name__, "message": str(exc)[:500]})
            if attempt == 1:
                raise
    raise AssertionError("unreachable")


def repair_small_boundary_gaps(phases: list, start: int, end: int) -> None:
    """Snap coarse, sampled-frame estimates by <=1 second; retain review flags."""
    if not isinstance(phases, list) or not phases:
        return
    cursor = start
    for phase in phases:
        a = phase.get("start_frame")
        if not isinstance(a, int) or abs(a - cursor) > FPS:
            return
        if a != cursor:
            phase["start_frame"] = cursor
            phase["boundary_adjusted"] = True
            phase["needs_review"] = True
        cursor = phase.get("end_frame")
        if not isinstance(cursor, int):
            return
    if abs(cursor - end) <= FPS and cursor != end:
        phases[-1]["end_frame"] = end
        phases[-1]["boundary_adjusted"] = True
        phases[-1]["needs_review"] = True


def validate_phases(phases: list, start: int, end: int) -> None:
    if not isinstance(phases, list) or not phases:
        raise ValueError("No model phases")
    cursor = start
    for phase in phases:
        a, b = phase.get("start_frame"), phase.get("end_frame")
        instruction = phase.get("instruction_en", "")
        if not isinstance(a, int) or not isinstance(b, int) or a != cursor or b <= a or b > end:
            raise ValueError(f"Non-contiguous or invalid model phase {a}, {b}, cursor {cursor}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Empty instruction")
        if instruction.lower().strip(" .") == "put the plastic bottles in the bin":
            raise ValueError("Global task copied as local instruction")
        cursor = b
    if cursor != end:
        raise ValueError(f"Model phases end at {cursor}, expected {end}")


def windows(n: int, max_frames: int = 720) -> list[tuple[int, int]]:
    # Balance the windows so a tiny tail is never sent alone.
    count = math.ceil(n / max_frames)
    return [(i * n // count, (i + 1) * n // count) for i in range(count)]


def normalize_phases(phases: list[dict], n: int) -> list[dict]:
    raw = [dict(p) for p in phases]
    # Merge tiny model phases into a neighbor, marking the result for review.
    i = 0
    while i < len(raw):
        if raw[i]["end_frame"] - raw[i]["start_frame"] >= 30:
            i += 1
            continue
        if len(raw) == 1:
            raise ValueError("Source episode shorter than one second")
        target = i - 1 if i > 0 else 1
        if target < i:
            raw[target]["end_frame"] = raw[i]["end_frame"]
            raw[target]["needs_review"] = True
            raw.pop(i)
            i = max(0, target)
        else:
            raw[target]["start_frame"] = raw[i]["start_frame"]
            raw[target]["needs_review"] = True
            raw.pop(i)
    # Coalesce identical neighboring instructions, then balance each long action.
    merged = []
    for p in raw:
        if merged and merged[-1]["instruction_en"].strip().lower() == p["instruction_en"].strip().lower():
            merged[-1]["end_frame"] = p["end_frame"]
            merged[-1]["needs_review"] = True
        else:
            merged.append(p)
    result = []
    for p in merged:
        a, b = p["start_frame"], p["end_frame"]
        pieces = math.ceil((b - a) / 360)
        for j in range(pieces):
            q = dict(p)
            q["start_frame"] = a + j * (b - a) // pieces
            q["end_frame"] = a + (j + 1) * (b - a) // pieces
            q["duration_split"] = pieces > 1
            q["needs_review"] = bool(q.get("needs_review") or pieces > 1 or q.get("confidence") != "high")
            result.append(q)
    cursor = 0
    for p in result:
        a, b = p["start_frame"], p["end_frame"]
        if a != cursor or not 30 <= b - a <= 360:
            raise ValueError(f"Invalid normalized segment {a}:{b} after {cursor}")
        cursor = b
    if cursor != n:
        raise ValueError(f"Incomplete normalized episode: {cursor}/{n}")
    return result


def export_video(src: dict, start: int, n: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    seek = (src["file_start_frame"] + start) / FPS
    run_ffmpeg(["-ss", f"{seek:.9f}", "-i", src["path"], "-frames:v", str(n),
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
                "-pix_fmt", "yuv420p", str(output)])
    count = mp4_frames(output)
    if count != n:
        raise ValueError(f"Video output frame count {count} != {n}: {output}")


def export_episode(ep: int, table: pa.Table, global_start: int, cameras: dict,
                   phases: list[dict], run: Path, next_index: int, next_global: int,
                   task_ids: dict) -> tuple[list[dict], int, int]:
    split = "val" if ep in VAL_IDS else "train"
    rows = []
    for j, phase in enumerate(phases):
        a, b = phase["start_frame"], phase["end_frame"]
        n = b - a
        segment_id = f"episode_{ep:06d}_s{j:03d}"
        directory = run / "segments" / split / segment_id
        directory.mkdir(parents=True, exist_ok=True)
        instruction = phase["instruction_en"].strip()
        if instruction not in task_ids:
            task_ids[instruction] = len(task_ids)
        task_id = task_ids[instruction]
        piece = table.slice(a, n)
        replacements = {"episode_index": pa.array([next_index] * n, type=pa.int64()),
                        "frame_index": pa.array(range(n), type=pa.int64()),
                        "timestamp": pa.array([k / FPS for k in range(n)], type=pa.float32()),
                        "index": pa.array(range(next_global, next_global + n), type=pa.int64()),
                        "task_index": pa.array([task_id] * n, type=pa.int64())}
        for key, array in replacements.items():
            idx = piece.schema.get_field_index(key)
            piece = piece.set_column(idx, key, array)
        # Existing source language annotations may describe a different interval.
        # Keep source in provenance and make the current local instruction explicit.
        if "language_persistent" in piece.column_names:
            idx = piece.schema.get_field_index("language_persistent")
            piece = piece.set_column(idx, "language_persistent",
                                     pa.array([[] for _ in range(n)], type=piece.schema.field(idx).type))
        pq.write_table(piece, directory / "data.parquet")
        (directory / "instruction.txt").write_text(instruction + "\n")
        for camera in CAMERAS:
            export_video(cameras[camera], a, n, directory / f"{camera}.mp4")
        if piece.num_rows != n or len(piece["action"][0].as_py()) != 14 or len(piece["observation.state"][0].as_py()) != 14:
            raise ValueError(f"Bad structured data for {segment_id}")
        manifest = {"segment_id": segment_id, "source_episode_index": ep,
                    "source_task_index": TASK_INDEX, "derived_episode_index": next_index,
                    "derived_task_index": task_id, "source_global_start": global_start,
                    "start_frame": a, "end_frame": b, "duration_frames": n,
                    "fps": FPS, "split": split, "instruction_en": instruction,
                    "confidence": phase.get("confidence", "unknown"),
                    "evidence": phase.get("evidence", ""),
                    "duration_split": phase.get("duration_split", False),
                    "boundary_adjusted": phase.get("boundary_adjusted", False),
                    "needs_review": True, "review_status": "pending",
                    "source_videos": {c: {"path": cameras[c]["path"],
                                         "file_start_frame": cameras[c]["file_start_frame"] + a}
                                      for c in CAMERAS}}
        write_json(directory / "manifest.json", manifest)
        row = dict(manifest)
        row["path"] = str(directory.relative_to(run))
        rows.append(row)
        next_index += 1
        next_global += n
    return rows, next_index, next_global


def preview(run: Path, episodes: list[int], all_rows: list[dict], model: str) -> None:
    folder = run / "preview"
    folder.mkdir(parents=True, exist_ok=True)
    links = []
    for ep in episodes:
        rows = [r for r in all_rows if r["source_episode_index"] == ep]
        if not rows:
            continue
        total = rows[-1]["end_frame"]
        blocks = []
        cards = []
        for r in rows:
            sid = r["segment_id"]
            name = html.escape(r["instruction_en"])
            length_pct = 100 * r["duration_frames"] / total
            blocks.append(f'<a href="#{sid}" style="width:{length_pct:.3f}%" title="{name}">{r["duration_frames"]/FPS:.1f}s</a>')
            root = "../" + r["path"]
            videos = "".join(f'<div><strong>{c}</strong><video controls preload="metadata" src="{root}/{c}.mp4"></video></div>' for c in CAMERAS)
            cards.append(f'<section id="{sid}"><h2>{sid}: {name}</h2><p>frames [{r["start_frame"]}, {r["end_frame"]}), {r["duration_frames"]/FPS:.2f}s · {r["split"]} · review pending</p><div class="views">{videos}</div></section>')
        sheet = f"sheets/source_ep_{ep:06d}/window_00000_00.jpg"
        page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Episode {ep} review</title>
<style>body{{font:16px system-ui;margin:24px;background:#f6f8fb;color:#162233}}.timeline{{display:flex;background:#ddd;border-radius:8px;overflow:hidden}}.timeline a{{box-sizing:border-box;text-align:center;background:#91c9df;border-right:2px solid white;color:#123;padding:18px 0;text-decoration:none;font-size:12px;overflow:hidden}}.timeline a:nth-child(even){{background:#f6c68b}}section{{background:white;margin:18px 0;padding:18px;border-radius:10px;box-shadow:0 1px 5px #ccc}}.views{{display:flex;gap:12px;flex-wrap:wrap}}video{{display:block;width:300px;max-width:100%}}h2{{font-size:18px}}a{{color:#146}}</style></head><body><p><a href="index.html">All episodes</a></p><h1>Source episode {ep}</h1><p>Full trajectory {total/FPS:.2f}s · {len(rows)} candidate subepisodes · labels require human review</p><div class="timeline">{''.join(blocks)}</div>{''.join(cards)}</body></html>'''
        page = page.replace('<div class="timeline">', f'<p><a href="{sheet}"><img src="{sheet}" alt="Three-view contact sheet" style="width:100%;max-width:1200px"></a></p><div class="timeline">', 1)
        (folder / f"source_ep_{ep:06d}.html").write_text(page)
        links.append(f'<li><a href="source_ep_{ep:06d}.html">Source episode {ep}</a>: {len(rows)} segments, {total/FPS:.1f}s<br><a href="source_ep_{ep:06d}.html"><img src="{sheet}" alt="episode {ep} contact sheet" style="max-width:680px;width:100%"></a></li>')
    (folder / "index.html").write_text('<!doctype html><html><head><meta charset="utf-8"><title>ABC-130K smoke review</title></head><body><h1>ABC-130K plastic bottles: ' + html.escape(model) + ' smoke review</h1><p>Candidate labels; human review pending. Not ready for training.</p><ul>' + ''.join(links) + '</ul></body></html>')


def verify_run(run: Path, source_tables: dict[int, pa.Table]) -> dict:
    rows = [json.loads(line) for line in (run / "index.jsonl").read_text().splitlines() if line]
    tasks_table = pq.read_table(run / "meta" / "tasks.parquet")
    task_map = {int(row["task_index"]): row["task"] for row in tasks_table.to_pylist()}
    seen_ids = set()
    expected_global = 0
    by_source = {}
    for derived_episode, row in enumerate(rows):
        sid = row["segment_id"]
        if sid in seen_ids:
            raise ValueError(f"Duplicate segment ID: {sid}")
        seen_ids.add(sid)
        ep = row["source_episode_index"]
        a, b = row["start_frame"], row["end_frame"]
        n = b - a
        if not 30 <= n <= 360 or n != row["duration_frames"]:
            raise ValueError(f"Bad duration: {sid}")
        if row["split"] != ("val" if ep in VAL_IDS else "train"):
            raise ValueError(f"Split mismatch: {sid}")
        if a != by_source.get(ep, 0):
            raise ValueError(f"Gap/overlap in source {ep}: {sid}")
        by_source[ep] = b
        directory = run / row["path"]
        piece = pq.read_table(directory / "data.parquet")
        src = source_tables[ep].slice(a, n)
        if piece.num_rows != n:
            raise ValueError(f"Parquet row mismatch: {sid}")
        for col in ("action", "observation.state"):
            if not piece[col].equals(src[col]):
                raise ValueError(f"Source {col} changed: {sid}")
            if any(len(x) != 14 for x in piece[col].to_pylist()):
                raise ValueError(f"Bad {col} dimensionality: {sid}")
        if piece["frame_index"].to_pylist() != list(range(n)):
            raise ValueError(f"Frame index mismatch: {sid}")
        if piece["episode_index"].to_pylist() != [derived_episode] * n:
            raise ValueError(f"Derived episode index mismatch: {sid}")
        if piece["index"].to_pylist() != list(range(expected_global, expected_global + n)):
            raise ValueError(f"Derived global index mismatch: {sid}")
        if any(abs(float(t) - i / FPS) > 0.0002 for i, t in enumerate(piece["timestamp"].to_pylist())):
            raise ValueError(f"Timestamp mismatch: {sid}")
        task_id = row["derived_task_index"]
        if piece["task_index"].to_pylist() != [task_id] * n:
            raise ValueError(f"Task ID mismatch: {sid}")
        if task_map.get(task_id) != row["instruction_en"] or (directory / "instruction.txt").read_text().strip() != row["instruction_en"]:
            raise ValueError(f"Training instruction mismatch: {sid}")
        if row["instruction_en"].strip().lower().strip(".") == "put the plastic bottles in the bin":
            raise ValueError(f"Global instruction leaked: {sid}")
        for camera in CAMERAS:
            if mp4_frames(directory / f"{camera}.mp4") != n:
                raise ValueError(f"Video frame mismatch: {sid}/{camera}")
        expected_global += n
    if set(by_source) != set(source_tables) or any(by_source[ep] != table.num_rows for ep, table in source_tables.items()):
        raise ValueError("Incomplete source coverage")
    return {"episodes": len(source_tables), "segments": len(rows),
            "frames": expected_global, "source_frames": sum(t.num_rows for t in source_tables.values()),
            "three_video_frame_counts_match": True, "action_state_equal_to_source": True,
            "task_text_mapping_match": True, "source_coverage_exact": True,
            "all_review_pending": all(x["review_status"] == "pending" for x in rows),
            "train_segments": sum(x["split"] == "train" for x in rows),
            "val_segments": sum(x["split"] == "val" for x in rows),
            "under_two_seconds": sum(x["duration_frames"] < 60 for x in rows),
            "boundary_adjusted": sum(bool(x.get("boundary_adjusted")) for x in rows),
            "duration_split": sum(bool(x.get("duration_split")) for x in rows),
            "ready_for_training": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "qwen"), default="deepseek")
    parser.add_argument("--episodes", nargs="+", type=int, default=[77, 78])
    parser.add_argument("--sample-stride-frames", type=int, default=15,
                        help="Frames between vision samples; 15 frames = 0.5 s at 30 fps")
    parser.add_argument("--max-window-frames", type=int, default=300,
                        help="Maximum frames in one model request; 300 frames = 10 s")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if not 1 <= args.sample_stride_frames <= FPS:
        parser.error("--sample-stride-frames must be in [1, 30]")
    if not FPS <= args.max_window_frames <= 1800:
        parser.error("--max-window-frames must be in [30, 1800]")
    provider = {
        "deepseek": {"key_env": "DEEPSEEK_API_KEY", "model": "deepseek-flash",
                     "base_url": "https://api.deepseek.com",
                     "extra_body": {"thinking": {"type": "disabled"}}},
        "qwen": {"key_env": "DASHSCOPE_API_KEY", "model": "qwen3.6-plus",
                 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 "extra_body": {"enable_thinking": False}},
    }[args.provider]
    if not os.environ.get(provider["key_env"]):
        parser.error(f"{provider['key_env']} must be set")
    if args.run_id is None:
        args.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + args.provider + "_smoke"
    run = PROJECT / "result" / "runs" / args.run_id
    if run.exists():
        parser.error(f"Run exists, choose a new --run-id: {run}")
    run.mkdir(parents=True)
    client = OpenAI(api_key=os.environ[provider["key_env"]], base_url=provider["base_url"])
    tables = {}
    global_starts = {}
    for ep in args.episodes:
        tables[ep], global_starts[ep] = load_episode(ep)
    mapping = source_video_map(max(global_starts[e] + tables[e].num_rows for e in args.episodes))
    source_rows = []
    for ep in args.episodes:
        cams = {c: locate(mapping[c], global_starts[ep], tables[ep].num_rows) for c in CAMERAS}
        source_rows.append({"episode": ep, "frames": tables[ep].num_rows,
                            "global_start": global_starts[ep], "split": "val" if ep in VAL_IDS else "train",
                            "cameras": cams})
        append_jsonl(run / "source_index.jsonl", source_rows[-1])
    write_json(run / "run.json", {"status": "running", "model": provider["model"],
                                  "provider": args.provider, "base_url": provider["base_url"],
                                  "episodes": args.episodes, "source": str(SOURCE),
                                  "fps": FPS, "sample_stride_frames": args.sample_stride_frames,
                                  "max_window_frames": args.max_window_frames,
                                  "ready_for_training": False,
                                  "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + USER_PROMPT).encode()).hexdigest()})
    all_rows = []
    next_episode = 0
    next_global = 0
    task_ids = {}
    for source in source_rows:
        ep = source["episode"]
        n = source["frames"]
        phases = []
        for a, b in windows(n, args.max_window_frames):
            sheets = make_sheets(ep, a, b, source["cameras"],
                                 run / "preview" / "sheets" / f"source_ep_{ep:06d}",
                                 args.sample_stride_frames)
            proposed = call_model(client, ep, a, b, sheets, run / "model_requests",
                                  args.sample_stride_frames, provider["model"],
                                  provider["extra_body"])
            phases.extend(proposed)
            print(f"episode {ep} window [{a},{b}): {len(proposed)} phases", flush=True)
        normalized = normalize_phases(phases, n)
        write_json(run / "proposals" / f"source_ep_{ep:06d}.json",
                   {"source_episode_index": ep, "raw_phases": phases, "segments": normalized})
        rows, next_episode, next_global = export_episode(
            ep, tables[ep], global_starts[ep], source["cameras"], normalized,
            run, next_episode, next_global, task_ids)
        for row in rows:
            append_jsonl(run / "index.jsonl", row)
        all_rows.extend(rows)
        preview(run, args.episodes, all_rows, provider["model"])
        print(f"episode {ep}: exported {len(rows)} segments", flush=True)
    tasks = pa.table({"task_index": pa.array(list(range(len(task_ids))), type=pa.int64()),
                      "task": pa.array(list(task_ids.keys()), type=pa.string())})
    (run / "meta").mkdir(exist_ok=True)
    pq.write_table(tasks, run / "meta" / "tasks.parquet")
    report = verify_run(run, tables)
    write_json(run / "qc" / "report.json", report)
    write_json(run / "run.json", {"status": "smoke_completed_needs_review",
                                  "model": provider["model"], "provider": args.provider,
                                  "base_url": provider["base_url"], "episodes": args.episodes,
                                  "source": str(SOURCE), "fps": FPS,
                                  "sample_stride_frames": args.sample_stride_frames,
                                  "max_window_frames": args.max_window_frames,
                                  "ready_for_training": False,
                                  "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + USER_PROMPT).encode()).hexdigest(),
                                  "qc": report})
    print(f"preview: {run / 'preview/index.html'}", flush=True)


if __name__ == "__main__":
    main()
