import os
import argparse
import json
import pandas as pd
import regex as re
import pywordsegment
from rapidfuzz import process
import language_tool_python

# 1. Define your regex alignment rules
grammar_regex_rules = [
    (r'(?:,\s*){2,}', ', '),
    (r'([?!.;:])(?:\s*[?!.;:])+', r'\1'),
    (r'^(?:,\s*)+', ''),
    (r'(?:,\s*)+$', ''),
    (r'\.{2,}', '.'),
    (r"\s+([,\.\!\?:;])", r"\1"),
    (r"([,\.\!\?:;])([^\s])", r"\1 \2"),
    (r"\s{2,}", ' '),
    (r"^'{2,}|'{2,}$", ''),
    (r"(?<=\w)''(?=\w)", "'"),
]

def clean_alignment(text: str) -> str:
    for pat, repl in grammar_regex_rules:
        text = re.sub(pat, repl, text)
    return text

def expand_slang(text: str, slang_dict: dict) -> str:
    pattern = re.compile(
        r'\b(' + '|'.join(re.escape(k) for k in slang_dict.keys()) + r')\b',
        re.IGNORECASE
    )
    return pattern.sub(lambda m: slang_dict[m.group(0).lower()], text)

def segment_text(text: str) -> str:
    tokens = text.split()
    segmented = []
    for tok in tokens:
        if tok.isalpha() and tok.lower() == tok and len(tok) > 7:
            parts = pywordsegment.WordSegmenter.segment(tok)
            segmented.extend(parts if len(parts) > 1 else [tok])
        else:
            segmented.append(tok)
    return ' '.join(segmented)

def process_text(text: str, slang_dict: dict) -> str:
    text = clean_alignment(text)
    text = expand_slang(text, slang_dict)
    text = segment_text(text)
    return text

def main():
    p = argparse.ArgumentParser(
        description="Clean, expand, dedupe and save transcripts"
    )
    p.add_argument(
        "--input-csv", "-i",
        default="../data/metadata.csv",
        help="Path to the metadata CSV (with a 'transcript' and 'wav_file' column)"
    )
    p.add_argument(
        "--output-list", "-o",
        default="../data/cleaned_transcripts.list",
        help="Where to write the cleaned transcript list"
    )
    p.add_argument(
        "--data-path", "-d",
        default="/workspace/data/processed-audio/",
        help="Prefix path to prepend to each wav_file"
    )
    p.add_argument(
        "--slang-json", "-s",
        default="slang_abbreviations.json",
        help="Path to the slang dictionary JSON"
    )
    args = p.parse_args()

    # Load data
    df = pd.read_csv(args.input_csv)

    # Load slang dictionary
    with open(args.slang_json, 'r', encoding='utf-8') as f:
        slang_dict = json.load(f)

    # Build clusters to dedupe
    orig_texts = df['transcript'].tolist()
    clusters = []
    threshold = 90
    for t in orig_texts:
        for cl in clusters:
            if process.extractOne(t, [cl[0]])[1] >= threshold:
                cl.append(t)
                break
        else:
            clusters.append([t])

    # Grammar checker
    tool = language_tool_python.LanguageTool('en-US')

    # Select best from each cluster & map originals → cleaned
    final_map = {}
    removed_set = set()
    for cl in clusters:
        removed_set.update(cl[1:])
        best_err = float('inf')
        best_proc = None
        for orig in cl:
            cand = process_text(orig, slang_dict)
            err_count = len(tool.check(cand))
            if err_count < best_err:
                best_err = err_count
                best_proc = cand
        for orig in cl:
            final_map[orig] = best_proc

    # Write cleaned list
    os.makedirs(os.path.dirname(args.output_list), exist_ok=True)
    with open(args.output_list, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            orig = row['transcript']
            if orig in removed_set:
                continue
            wav = os.path.join(args.data_path, row['wav_file'])
            cleaned = clean_alignment(final_map.get(orig, ''))
            fout.write(f"{wav}|EN-default|EN|{cleaned}\n")

if __name__ == "__main__":
    main()
