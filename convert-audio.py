#!/usr/bin/env python3
"""
Resample every .wav file in a directory tree to 44 100 Hz.

Dependencies
------------
pip install pydub tqdm

You also need FFmpeg (pydub’s back-end):  
  • Linux/macOS: install via your package manager (e.g., `brew install ffmpeg`)  
  • Windows: download the FFmpeg build and add its /bin folder to your PATH.
"""

from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

# ---------- CONFIG -----------------------------------------------------------
SOURCE_DIR = Path("data/raw-audio")   # <- change this
OUTPUT_DIR = Path("data/processed-audio") # <- output directory
RECURSIVE  = True                          # set False to skip sub-folders
TARGET_SR  = 44_100                        # samples per second
# -----------------------------------------------------------------------------

def resample_wav(src_path: Path, out_dir: Path, target_sr: int):
    """Load -> resample -> export a single WAV.
    
    Args:
        src_path: Path to the source WAV file
        out_dir: Directory where the resampled file will be saved
        target_sr: Target sample rate in Hz
        
    Returns:
        Path to the resampled file
    """
    audio = AudioSegment.from_wav(src_path)          # pydub auto-detects SR
    audio = audio.set_frame_rate(target_sr)
    
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep the original filename but place it in the output directory
    out_path = out_dir / src_path.name
    audio.export(out_path, format="wav")
    return out_path

def main():
    pattern = "**/*.wav" if RECURSIVE else "*.wav"
    wav_paths = list(SOURCE_DIR.glob(pattern))

    if not wav_paths:
        print("No .wav files found.")
        return
        
    # Create output directory structure
    if OUTPUT_DIR == SOURCE_DIR:
        print("Warning: Output directory is the same as source directory.")
        print("Original files will be overwritten.")
        proceed = input("Continue? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Operation cancelled.")
            return
    
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target sample rate: {TARGET_SR} Hz")
    
    # Keep track of processed files
    processed_count = 0
    skipped_count = 0

    for wav_file in tqdm(wav_paths, desc="Resampling", unit="file"):
        try:
            # Calculate the relative path from SOURCE_DIR to preserve directory structure
            rel_path = wav_file.relative_to(SOURCE_DIR) if RECURSIVE else wav_file.name
            
            # Create the corresponding output directory
            if RECURSIVE:
                out_subdir = OUTPUT_DIR / rel_path.parent
            else:
                out_subdir = OUTPUT_DIR
                
            # Resample and save the file
            output_file = resample_wav(wav_file, out_subdir, TARGET_SR)
            processed_count += 1
        except Exception as exc:
            print(f"⚠️  Skipped {wav_file}: {exc}")
            skipped_count += 1

    print(f"Done! Resampled {processed_count} files at {TARGET_SR} Hz.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files due to errors.")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
