"""Colab session helpers: environment report, Drive, zip handling.

Everything here degrades gracefully outside Colab (the local fallback used
when running notebooks on a workstation), and every long copy or download
shows progress - Drive transfers are slow enough that silence reads as a
hang.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm


def setup_report() -> dict:
    """Print and return the facts that determine what this session can do."""
    import platform
    import sys

    import torch

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
        "vram_gb": None,
        "disk_free_gb": round(shutil.disk_usage("/").free / 2**30, 1),
        "cpu_count": os.cpu_count(),
        "in_colab": "google.colab" in sys.modules or os.path.exists("/content"),
    }
    if report["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = properties.name
        report["vram_gb"] = round(properties.total_memory / 2**30, 1)
    for key, value in report.items():
        print(f"{key:>14}: {value}")
    return report


def mount_drive():
    """Mount Google Drive on Colab; returns the MyDrive path (None locally)."""
    if not os.path.exists("/content"):
        return None
    mount_point = Path("/content/drive")
    if not (mount_point / "MyDrive").exists():
        from google.colab import drive

        drive.mount(str(mount_point))
    return mount_point / "MyDrive"


def copy_with_progress(src, dst, chunk_mb: int = 16) -> Path:
    """Chunked file copy with a progress bar (Drive-friendly), atomic move."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    total = src.stat().st_size
    with open(src, "rb") as fin, open(tmp, "wb") as fout, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"copy {src.name}"
    ) as bar:
        while True:
            chunk = fin.read(chunk_mb * 2**20)
            if not chunk:
                break
            fout.write(chunk)
            bar.update(len(chunk))
    os.replace(tmp, dst)
    return dst


def download_with_progress(url: str, dst) -> Path:
    """Stream a URL to disk with a progress bar, atomic move into place."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(tmp, "wb") as fout, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {dst.name}"
        ) as bar:
            while True:
                chunk = response.read(16 * 2**20)
                if not chunk:
                    break
                fout.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dst)
    return dst


def ensure_zip_local(cfg, local_dir="/content") -> Path:
    """The dataset zip on fast local disk, downloading/copying as needed.

    Order: local copy if present; else copy from Drive; else download from
    S3 to local disk and back it up to Drive.
    """
    from starx.config import raw_zip_path

    local_zip = Path(local_dir) / "r1.0.1.zip"
    drive_zip = raw_zip_path(cfg)
    if local_zip.exists():
        return local_zip
    if drive_zip.exists():
        return copy_with_progress(drive_zip, local_zip)
    download_with_progress(cfg.data_url, local_zip)
    if Path(cfg.drive_root).exists():
        copy_with_progress(local_zip, drive_zip)
    return local_zip


def zip_inventory(zip_path) -> list:
    """All member names in the zip (fast: central directory only)."""
    with zipfile.ZipFile(zip_path) as zf:
        return zf.namelist()


def extract_members(zip_path, members, dest) -> list:
    """Extract specific members flat into dest; returns extracted paths."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in members:
            target = dest / Path(member).name
            with zf.open(member) as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            out.append(target)
    return out


def extract_design(zip_path, design_id: str, dest) -> dict:
    """Pull one design's files out of the zip; {suffix: local path}."""
    members = [n for n in zip_inventory(zip_path) if design_id in n]
    extracted = extract_members(zip_path, members, dest)
    return {p.suffix.lstrip("."): p for p in extracted}


def design_members(names, design_id: str) -> list:
    """All zip member names belonging to one design."""
    import os.path

    return [n for n in names if os.path.basename(n).startswith(design_id)]


def json_member(names, design_id: str):
    """The design's construction-sequence json member (exact stem match)."""
    import os.path

    for name in names:
        if os.path.basename(name) == f"{design_id}.json":
            return name
    return None


def build_zip_index(names, design_ids) -> dict:
    """One-pass member map {design_id: {json, obj, png}} for the whole zip.

    obj is the design's final mesh: the plain <id>.obj when present,
    otherwise the highest construction-step obj (which sorts last).
    """
    import os.path

    id_set = set(design_ids)
    index = {design_id: {} for design_id in design_ids}
    for name in names:
        base = os.path.basename(name)
        stem, dot, ext = base.partition(".")
        if not dot:
            continue
        design_id = stem if stem in id_set else stem.rsplit("_", 1)[0]
        if design_id not in id_set:
            continue
        entry = index[design_id]
        if ext == "json" and stem == design_id:
            entry["json"] = name
        elif ext == "png" and stem == design_id:
            entry["png"] = name
        elif ext == "obj":
            if stem == design_id:
                entry["obj"] = name
                entry["obj_is_final"] = True
            elif not entry.get("obj_is_final") and name > entry.get("obj", ""):
                entry["obj"] = name
    return index


def final_obj_member(names, design_id: str):
    """The design's final mesh: the plain <id>.obj if present, otherwise the
    last per-construction-step obj (<id>_NNNN.obj sorts last)."""
    import os.path

    objs = sorted(
        n
        for n in names
        if os.path.basename(n).startswith(design_id)
        and n.endswith(".obj")
    )
    if not objs:
        return None
    for name in objs:
        if os.path.basename(name) == f"{design_id}.obj":
            return name
    return objs[-1]
