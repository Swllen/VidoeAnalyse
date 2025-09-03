#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量版：两种/三种处理范围可选
- profiles_recursive：递归查找所有名为 'profile' 的文件夹，将其中的图片合成为视频，输出到 profile/OUTPUT_REL
- subdirs_level1     ：仅处理 ROOT_DIR 下一层的每个子文件夹，将其内图片合成为视频，输出到 子文件夹/video/output.mp4
- dirs_recursive     ：递归处理所有包含图片的目录（排除输出目录），输出到 该目录/video/output.mp4
"""

import re, sys, subprocess, shutil
from pathlib import Path
from typing import List, Optional

# =========================
# CONFIG —— 在这里修改即可
# =========================
FFMPEG_BIN = r"D:\SoftWare\ffmpeg\ffmpeg-2025-08-20-git-4d7c609be3-essentials_build\ffmpeg-2025-08-20-git-4d7c609be3-essentials_build\bin\ffmpeg.exe"
ROOT_DIR    = Path(r"E:\BU_Research\LabData\0901\StageTest\BPC303\Z_Channel3")

# 处理范围：'profiles_recursive' | 'subdirs_level1' | 'dirs_recursive'
PROCESS_SCOPE = "profiles_recursive"

FPS         = 10.0
SELECT_EXPR = ""                 # 如 "1-100,205,300-450:2"；留空表示用全部
FILES_LIST  = None               # 例：Path("keep.txt")，每行一个相对/绝对路径
CODEC       = "ffv1"             # 'ffv1'（推荐 .mkv）或 'h264-lossless'（推荐 .mp4/.mkv）

# 输出相对路径
# - 对于 profiles_recursive：相对于 profile 目录
OUTPUT_REL_PROFILE = Path("video/output.mp4")
# - 对于 子目录/任意目录 模式：相对于该目录
OUTPUT_REL_DIR     = Path("video/output.mp4")

SKIP_EXISTING = False             # 已存在则跳过
DRY_RUN       = False            # 只打印命令，不执行
# =========================


def parse_selection(expr: str):
    sel = set()
    if not expr:
        return sel
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        m = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", p)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            step = int(m.group(3)) if m.group(3) else 1
            if step <= 0:
                raise ValueError(f"步长必须 > 0: {p}")
            rng = range(start, end + 1, step) if start <= end else range(start, end - 1, -step)
            sel.update(rng)
        else:
            if not p.isdigit():
                raise ValueError(f"无法解析选择项：{p}")
            sel.add(int(p))
    return sel


def extract_index(p: Path):
    m = re.search(r"(\d+)(?=\.[A-Za-z0-9]+$)", p.name)
    return int(m.group(1)) if m else None


def build_listfile(image_paths: List[Path], listfile: Path, fps: float):
    with listfile.open("w", encoding="utf-8") as f:
        dur = 1.0 / fps
        for img in image_paths:
            f.write(f"file '{img.as_posix()}'\n")
            f.write(f"duration {dur:.12f}\n")
        if image_paths:
            f.write(f"file '{image_paths[-1].as_posix()}'\n")


def resolve_ffmpeg():
    if FFMPEG_BIN and Path(FFMPEG_BIN).exists():
        return str(Path(FFMPEG_BIN))
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    print("错误：未检测到 ffmpeg。请设置 FFMPEG_BIN 或将 ffmpeg 加入 PATH。", file=sys.stderr)
    sys.exit(2)


def collect_images(input_dir: Path) -> List[Path]:
    imgs = sorted(
        [p for p in input_dir.iterdir()
         if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]],
        key=lambda p: (extract_index(p) if extract_index(p) is not None else float("inf"),
                       p.name.lower())
    )
    return imgs


def choose_images(all_imgs: List[Path], input_dir: Path, files_list: Optional[Path], select_expr: str) -> List[Path]:
    if files_list:
        list_file = Path(files_list)
        if not list_file.exists():
            print(f"错误：FILES_LIST 文件不存在：{list_file}", file=sys.stderr)
            return []
        lines = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        chosen = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = (input_dir / ln).resolve()
            if not p.exists():
                print(f"警告：清单中不存在 {ln}，忽略。", file=sys.stderr)
                continue
            chosen.append(p)
        return chosen
    else:
        selset = parse_selection(select_expr) if select_expr else None
        if selset:
            chosen = []
            for p in all_imgs:
                idx = extract_index(p)
                if idx is not None and idx in selset:
                    chosen.append(p.resolve())
            return chosen
        else:
            return [p.resolve() for p in all_imgs]


def build_ffmpeg_cmd(ffmpeg_bin: str, listfile: Path, fps: float, codec: str, out_path: Path) -> List[str]:
    codec_norm = codec.lower().strip()
    if out_path.suffix.lower() == ".mp4" and codec_norm == "ffv1":
        print(f"提示：输出为 .mp4，不支持 FFV1。自动改为 h264-lossless。 -> {out_path}", file=sys.stderr)
        codec_norm = "h264-lossless"

    if codec_norm == "ffv1":
        vcodec = ["-c:v", "ffv1", "-level", "3"]
    elif codec_norm == "h264-lossless":
        vcodec = ["-c:v", "libx264", "-crf", "0", "-preset", "veryslow", "-pix_fmt", "yuv444p"]
    else:
        raise ValueError("CODEC 只能是 'ffv1' 或 'h264-lossless'")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-safe", "0",
        "-f", "concat",
        "-i", str(listfile),
        "-r", f"{fps}",
        *vcodec,
        str(out_path)
    ]
    return cmd


# ========== 三种处理单元 ==========
def process_profile(profile_dir: Path, ffmpeg_bin: str) -> bool:
    try:
        if not profile_dir.is_dir():
            print(f"[跳过] 非目录：{profile_dir}")
            return False

        all_imgs = collect_images(profile_dir)
        if not all_imgs:
            print(f"[跳过] 无图片：{profile_dir}")
            return False

        chosen = choose_images(all_imgs, profile_dir, FILES_LIST, SELECT_EXPR)
        if not chosen:
            print(f"[跳过] 未选中任何图片：{profile_dir}")
            return False

        out_path = (profile_dir / OUTPUT_REL_PROFILE).resolve()
        if SKIP_EXISTING and out_path.exists():
            print(f"[跳过] 已存在输出：{out_path}")
            return True

        workdir = out_path.parent
        workdir.mkdir(parents=True, exist_ok=True)
        listfile = (workdir / (out_path.stem + "_list.txt")).resolve()
        build_listfile(chosen, listfile, FPS)

        cmd = build_ffmpeg_cmd(ffmpeg_bin, listfile, FPS, CODEC, out_path)

        sample_show = ", ".join([p.name for p in chosen[:3]]) + (" ..." if len(chosen) > 3 else "")
        print(f"\n=== 处理 profile：{profile_dir}")
        print(f"将合成 {len(chosen)} 帧；示例：{sample_show}")
        print("即将执行命令：", " ".join(map(str, cmd)))

        if DRY_RUN:
            print("DRY_RUN = True，不实际执行。")
            return True

        subprocess.check_call(cmd)
        print(f"完成：{out_path}")
        print(f"中间列表文件：{listfile}")
        return True

    except subprocess.CalledProcessError:
        print(f"[失败] ffmpeg 运行失败：{profile_dir}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[失败] {profile_dir} -> {e}", file=sys.stderr)
        return False


def process_dir_images(dir_path: Path, ffmpeg_bin: str) -> bool:
    """把某个目录自身的图片合成为视频，输出到 目录/OUTPUT_REL_DIR"""
    try:
        if not dir_path.is_dir():
            return False
        all_imgs = collect_images(dir_path)
        if not all_imgs:
            return False

        chosen = choose_images(all_imgs, dir_path, FILES_LIST, SELECT_EXPR)
        if not chosen:
            return False

        out_path = (dir_path / OUTPUT_REL_DIR).resolve()
        if SKIP_EXISTING and out_path.exists():
            print(f"[跳过] 已存在输出：{out_path}")
            return True

        workdir = out_path.parent
        workdir.mkdir(parents=True, exist_ok=True)
        listfile = (workdir / (out_path.stem + "_list.txt")).resolve()
        build_listfile(chosen, listfile, FPS)

        cmd = build_ffmpeg_cmd(ffmpeg_bin, listfile, FPS, CODEC, out_path)
        sample_show = ", ".join([p.name for p in chosen[:3]]) + (" ..." if len(chosen) > 3 else "")
        print(f"\n=== 处理目录：{dir_path}")
        print(f"将合成 {len(chosen)} 帧；示例：{sample_show}")
        print("即将执行命令：", " ".join(map(str, cmd)))

        if DRY_RUN:
            print("DRY_RUN = True，不实际执行。")
            return True

        subprocess.check_call(cmd)
        print(f"完成：{out_path}")
        print(f"中间列表文件：{listfile}")
        return True

    except subprocess.CalledProcessError:
        print(f"[失败] ffmpeg 运行失败：{dir_path}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[失败] {dir_path} -> {e}", file=sys.stderr)
        return False
# ==========================================


def main():
    if FPS <= 0:
        print("错误：FPS 必须 > 0", file=sys.stderr)
        sys.exit(2)

    if not ROOT_DIR.is_dir():
        print(f"错误：找不到根目录 {ROOT_DIR}", file=sys.stderr)
        sys.exit(2)

    ffmpeg_bin = resolve_ffmpeg()

    success_dirs = []
    failed_dirs = []

    scope = PROCESS_SCOPE.strip().lower()

    if scope == "profiles_recursive":
        # 递归查找所有名为 'profile' 的文件夹
        profile_dirs = [p for p in ROOT_DIR.rglob("profile") if p.is_dir()]
        if not profile_dirs:
            print("错误：未在 ROOT_DIR 下找到任何 'profile' 目录")
            sys.exit(2)

        print(f"发现 {len(profile_dirs)} 个 profile 目录，将逐一处理。")
        for p in sorted(profile_dirs):
            if process_profile(p, ffmpeg_bin):
                success_dirs.append(p)
            else:
                failed_dirs.append(p)

    elif scope == "subdirs_level1":
        # 只处理根目录下一层的子文件夹
        subdirs = [d for d in ROOT_DIR.iterdir() if d.is_dir()]
        if not subdirs:
            print("错误：ROOT_DIR 下没有子文件夹可处理")
            sys.exit(2)

        print(f"发现 {len(subdirs)} 个一级子文件夹，将逐一处理（目录内图片→video/output.mp4）。")
        for d in sorted(subdirs):
            ok = process_dir_images(d, ffmpeg_bin)
            (success_dirs if ok else failed_dirs).append(d)

    elif scope == "dirs_recursive":
        # 递归处理所有包含图片的目录（排除输出目录）
        candidate_dirs = set()
        for p in ROOT_DIR.rglob("*"):
            if p.is_dir():
                # 跳过常见输出目录，避免把生成的视频再次加入
                if p.name.lower() in {"video", "videos"}:
                    continue
                # 目录下是否有可用图片
                if any((p / f).is_file() and (p / f).suffix.lower() in
                       [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]
                       for f in [x.name for x in p.iterdir() if x.is_file()]):
                    candidate_dirs.add(p)

        if not candidate_dirs:
            print("错误：未发现包含图片的目录")
            sys.exit(2)

        print(f"发现 {len(candidate_dirs)} 个包含图片的目录，将逐一处理（目录内图片→video/output.mp4）。")
        for d in sorted(candidate_dirs):
            ok = process_dir_images(d, ffmpeg_bin)
            (success_dirs if ok else failed_dirs).append(d)

    else:
        print(f"错误：未知的 PROCESS_SCOPE：{PROCESS_SCOPE}（应为 'profiles_recursive' | 'subdirs_level1' | 'dirs_recursive'）")
        sys.exit(2)

    # ===== 汇总 =====
    total = len(success_dirs) + len(failed_dirs)
    print("\n========== 汇总结果 ==========")
    print(f"成功 {len(success_dirs)} / 共 {total}")

    if success_dirs:
        print("\n成功的目录：")
        for d in success_dirs:
            print("  ✔", d)

    if failed_dirs:
        print("\n失败的目录：")
        for d in failed_dirs:
            print("  ✘", d)

    print("================================")

    if len(success_dirs) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
