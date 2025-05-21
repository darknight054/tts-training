#!/usr/bin/env python3
"""
Resample every .wav file in a directory tree to a target sample rate.

Dependencies
------------
  pip install pydub tqdm

You also need FFmpeg (pydub’s back-end):
  • Linux/macOS: install via your package manager (e.g., `brew install ffmpeg`)
  • Windows: download the FFmpeg build and add its /bin folder to your PATH.
"""

import argparse
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm


def resample_wav(src_path: Path, out_dir: Path, target_sr: int) -> Path:
    """Load -> resample -> export a single WAV file.

    Args:
        src_path: Path to the source WAV file
        out_dir: Directory where the resampled file will be saved
        target_sr: Target sample rate in Hz

    Returns:
        Path to the resampled file
    """
    audio = AudioSegment.from_wav(src_path)
    audio = audio.set_frame_rate(target_sr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / src_path.name
    audio.export(out_path, format="wav")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Resample .wav files in a directory tree to a target sample rate"
    )
    parser.add_argument(
        "--source-dir", "-s",
        type=Path,
        default=Path("data/raw-audio"),
        help="Directory containing original .wav files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/processed-audio"),
        help="Directory where resampled .wav files will be saved"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively include subdirectories (default)"
    )
    parser.add_argument(
        "--no-recursive", dest="recursive",
        action="store_false",
        help="Do not recurse; only process top-level directory"
    )
    parser.add_argument(
        "--target-sr", "-t",
        type=int,
        default=44100,
        help="Target sample rate in Hz (default: 44100)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite in-place without confirmation if source == output"
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir
    recursive = args.recursive if args.recursive is not None else True
    target_sr = args.target_sr
    force = args.force

    pattern = "**/*.wav" if recursive else "*.wav"
    wav_paths = list(source_dir.glob(pattern))

    if not wav_paths:
        print(f"No .wav files found in {source_dir}.")
        return

    if source_dir.resolve() == output_dir.resolve():
        if not force:
            print("Warning: Output directory is the same as source directory. Original files will be overwritten.")
            resp = input("Continue? (y/n): ").lower().strip()
            if resp != 'y':
                print("Operation cancelled.")
                return

    print(f"Source directory : {source_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Recursive         : {recursive}")
    print(f"Target sample rate: {target_sr} Hz")

    processed, skipped = 0, 0
    for wav_file in tqdm(wav_paths, desc="Resampling", unit="file"):
        try:
            rel_path = wav_file.relative_to(source_dir) if recursive else wav_file.name
            out_subdir = (output_dir / rel_path.parent) if recursive else output_dir
            resample_wav(wav_file, out_subdir, target_sr)
            processed += 1
        except Exception as e:
            print(f"⚠️ Skipped {wav_file}: {e}")
            skipped += 1

    print(f"Done! Resampled {processed} files at {target_sr} Hz.")
    if skipped:
        print(f"Skipped {skipped} files due to errors.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
