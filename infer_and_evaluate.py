#!/usr/bin/env python3
"""
infer_and_evaluate.py

One-stop script to:
1) Select random examples from metadata.
2) Run inference with MeloTTS on those examples.
3) Compute similarity metrics (KL, L1, L2, DTW).
4) Save waveforms, spectrograms, loss curves.

Usage:
    python infer_and_evaluate.py \
      --ckpt_path PATH/to/checkpoint.pth \
      --config_path PATH/to/config.json \
      --metadata PATH/to/Metadata.csv \
      --output_dir ./results \
      --num_examples 3
"""
import os
import argparse
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from melo.api import TTS

def parse_args():
    p = argparse.ArgumentParser(description="Infer & evaluate MeloTTS on random examples")
    p.add_argument("--ckpt_path",  required=True, help="Path to MeloTTS checkpoint (.pth)")
    p.add_argument("--config_path", default="/workspace/data/models/config.json", help="Path to MeloTTS config.json")
    p.add_argument("--metadata",    required=True, help="Path to pipe-separated list")
    p.add_argument("--output_dir",  default="/workspace/results", help="Directory for synth and plots")
    p.add_argument("--num_examples",type=int, default=3, help="How many random examples to process")
    return p.parse_args()

def read_metadata_line(line):
    parts = line.strip().split("|", 6)
    if len(parts) != 7:
        raise ValueError(f"Malformed metadata line: {line}")
    audiopath, spk_name, lang, text, phones, tone, word2ph = parts
    return audiopath, spk_name, lang, text.strip('"'), phones, tone, word2ph

def compute_mel_spectrogram(y, sr, n_mels=80):
    n_fft = 2048 if sr >= 40000 else 1024
    hop_length = n_fft // 4
    return librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                          hop_length=hop_length, n_mels=n_mels, power=2.0), hop_length

def kl_divergence(mel_pred, mel_gt):
    T = min(mel_pred.shape[1], mel_gt.shape[1])
    P = mel_pred[:, :T].astype(np.float64) + 1e-10
    Q = mel_gt[:, :T].astype(np.float64) + 1e-10
    P /= P.sum(axis=0, keepdims=True)
    Q /= Q.sum(axis=0, keepdims=True)
    kl = np.sum(P * np.log(P / Q), axis=0)
    return float(np.mean(kl))

def l1_waveform_loss(y_pred, y_gt):
    L = min(len(y_pred), len(y_gt))
    return float(np.mean(np.abs(y_pred[:L] - y_gt[:L])))

def l2_waveform_loss(y_pred, y_gt):
    L = min(len(y_pred), len(y_gt))
    return float(np.mean((y_pred[:L] - y_gt[:L])**2))

def dtw_distance(mel_pred, mel_gt):
    from librosa.sequence import dtw
    M_pred_db = librosa.power_to_db(mel_pred, ref=np.max)
    M_gt_db   = librosa.power_to_db(mel_gt,   ref=np.max)
    D, wp = dtw(X=M_gt_db, Y=M_pred_db, metric='euclidean')
    return float(D[-1, -1] / len(wp))

def save_waveform_plot(y_gt, y_pred, out_path):
    plt.figure(figsize=(8,3))
    plt.plot(y_gt, label="GT", alpha=0.8)
    plt.plot(y_pred, label="Synth", alpha=0.6)
    plt.legend(); plt.title("Waveform")
    plt.xlabel("Samples"); plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def save_mel_plots(mel_gt, mel_pred, sr, hop_length, out_path):
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    librosa.display.specshow(librosa.power_to_db(mel_gt, ref=np.max),
                             sr=sr, hop_length=hop_length,
                             y_axis='mel', x_axis='time')
    plt.title("GT Mel-Spectrogram"); plt.colorbar(format='%+2.0f dB')
    plt.subplot(2,1,2)
    librosa.display.specshow(librosa.power_to_db(mel_pred, ref=np.max),
                             sr=sr, hop_length=hop_length,
                             y_axis='mel', x_axis='time')
    plt.title("Synth Mel-Spectrogram"); plt.colorbar(format='%+2.0f dB')
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_l1_loss_curve(mel_gt, mel_pred, out_path):
    T = min(mel_gt.shape[1], mel_pred.shape[1])
    err = np.mean(np.abs(mel_pred[:,:T] - mel_gt[:,:T]), axis=0)
    plt.figure(figsize=(6,2))
    plt.plot(err); plt.title("Mel L1 Error per Frame")
    plt.xlabel("Frame"); plt.ylabel("Error")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_mel_l2_curve(mel_gt, mel_pred, out_path):
    T = min(mel_gt.shape[1], mel_pred.shape[1])
    mel_sq_err = np.mean((mel_pred[:, :T] - mel_gt[:, :T])**2, axis=0)
    plt.figure(figsize=(6,2))
    plt.plot(mel_sq_err)
    plt.title("Frame-wise Mel L2 Error")
    plt.xlabel("Frame"); plt.ylabel("MSE")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Read metadata
    with open(args.metadata, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    metadata = {}
    for line in lines:
        audiopath, spk_name, lang, text, phones, tone, w2ph = read_metadata_line(line)
        metadata[audiopath] = {
            'spk_name': spk_name, 'lang': lang, 'text': text,
            'phones': phones, 'tone': tone, 'w2ph': w2ph
        }

    if len(metadata) < args.num_examples:
        raise RuntimeError("Not enough examples in metadata.")

    # Random selection
    random.seed(42)
    samples = random.sample(list(metadata.keys()), args.num_examples)

    # Initialize TTS model
    tts = TTS(language="EN", config_path=args.config_path, ckpt_path=args.ckpt_path)

    # Evaluate each sample
    for i, audiopath in enumerate(samples, 1):
        meta = metadata[audiopath]
        print(f"\n=== Example {i} ===")
        print("Audio Path:", audiopath)
        print("Text      :", meta['text'])
        print("Phones    :", meta['phones'])
        print("Tone      :", meta['tone'])
        # Load GT
        y_gt, sr = librosa.load(audiopath, sr=None)
        # Inference
        fname = os.path.basename(audiopath)
        out_wav = os.path.join(args.output_dir, f"example_{i}", f"{fname}")
        for spk_name, spk_id in tts.hps.data.spk2id.items():
            os.makedirs(os.path.dirname(out_wav), exist_ok=True)
            tts.tts_to_file(meta['text'], spk_id, out_wav)
        y_pred, _ = librosa.load(out_wav, sr=sr)
        # Compute mels
        mel_gt, hop = compute_mel_spectrogram(y_gt, sr)
        mel_pred, _   = compute_mel_spectrogram(y_pred, sr)
        # Metrics
        kld = kl_divergence(mel_pred, mel_gt)
        l1  = l1_waveform_loss(y_pred, y_gt)
        l2  = l2_waveform_loss(y_pred, y_gt)
        try:
            dtw = dtw_distance(mel_pred, mel_gt)
        except:
            dtw = None
        print(f"KL Divergence: {kld:.4f}")
        print(f"L1 Loss:       {l1:.4f}")
        print(f"L2 Loss:       {l2:.4f}")
        print(f"DTW Distance:  {dtw:.4f}" if dtw is not None else "DTW Distance:  N/A")
        # Save plots
        base = os.path.join(args.output_dir, f"example_{i}", "plot")
        save_waveform_plot(y_gt, y_pred, base + "_waveform.png")
        save_mel_plots(mel_gt, mel_pred, sr, hop, base + "_mels.png")
        save_l1_loss_curve(mel_gt, mel_pred, base + "_l1_loss.png")
        save_mel_l2_curve(mel_gt, mel_pred, base + "_l2_loss.png")
        print(f"Plots and synth saved under {os.path.join(args.output_dir, f'example_{i}')}")

if __name__ == "__main__":
    main()
