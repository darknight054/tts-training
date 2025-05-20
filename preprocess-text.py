import pandas as pd
import regex as re
import json
import pywordsegment
from rapidfuzz import process
import language_tool_python

# 1. Load data
input_csv = '../data/metadata.csv'
output_txt = '../data/cleaned_transcripts.list'
df = pd.read_csv(input_csv)
data_path = "/workspace/data/processed-audio/"

# 2. Load slang dictionary
with open('slang_abbreviations.json', 'r', encoding='utf-8') as f:
    slang_dict = json.load(f)

# 3. Define regex alignment rules
# - Collapse excessive letter repetitions
# - Collapse repeated commas and other punctuation
# - Strip stray leading commas and internal duplicate punctuation
# - Preserve single end-of-sentence punctuation
# - Existing punctuation and spacing fixes
grammar_regex_rules = [              # collapse 3+ repeated letters to one (Heyyyy -> Hey)
    (r'(?:,\s*){2,}', ', '),                      # collapse repeated commas
    (r'([?!.;:])(?:\s*[?!.;:])+', r'\1'),        # collapse repeated sentence-ending punctuation (?, !, ., ;) to single
    (r'^(?:,\s*)+', ''),                          # strip leading commas/spaces
    (r'(?:,\s*)+$', ''),                          # strip trailing commas/spaces
    (r'\.{2,}', '.'),                             # collapse other repeated periods
    (r"\s+([,\.\!\?:;])", r"\1"),          # remove space before punctuation
    (r"([,\.\!\?:;])([^\s])", r"\1 \2"),  # ensure space after punctuation
    (r"\s{2,}", ' '),                           # collapse multiple spaces
    (r"^'{2,}|'{2,}$", ''),                      # strip double apostrophes at start/end
    (r"(?<=\w)''(?=\w)", "'"),               # fix stray double apostrophes between words
]

def clean_alignment(text: str) -> str:
    for pat, repl in grammar_regex_rules:
        text = re.sub(pat, repl, text)
    return text

# 4. Expand slang
slang_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in slang_dict.keys()) + r')\b',
    re.IGNORECASE
)
def expand_slang(text: str) -> str:
    return slang_pattern.sub(lambda m: slang_dict[m.group(0).lower()], text)

# 5. Segment concatenated words using pywordsegment, token-wise
def segment_text(text: str) -> str:
    tokens = text.split()
    segmented = []
    for tok in tokens:
        if tok.isalpha() and tok.lower() == tok and len(tok) > 7:
            parts = pywordsegment.WordSegmenter.segment(tok)
            if len(parts) > 1:
                segmented.extend(parts)
            else:
                segmented.append(tok)
        else:
            segmented.append(tok)
    return ' '.join(segmented)

# 6. Load punctuation model and grammar checker
# punct_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
tool = language_tool_python.LanguageTool('en-US')

# 7. Full processing pipeline
def process_text(text: str) -> str:
    text = clean_alignment(text)
    text = expand_slang(text)
    text = segment_text(text)
    # text = punct_model.add_punctuation_capitalization([text])[0]
    # text = clean_alignment(text)  # re-clean to remove any stray punctuation after model
    return text

# 8. Cluster & dedupe using RapidFuzz
orig_texts = df['transcript'].tolist()
clusters = []
threshold = 90
for t in orig_texts:
    placed = False
    for cl in clusters:
        if process.extractOne(t, [cl[0]])[1] >= threshold:
            cl.append(t)
            placed = True
            break
    if not placed:
        clusters.append([t])

# 9. Select best from each cluster based on grammar errors
final_map = {}
removed_set = set()
for cl in clusters:
    for dup in cl[1:]:
        removed_set.add(dup)
    best_err = float('inf')
    best_proc = None
    for orig in cl:
        cand = process_text(orig)
        err_count = len(tool.check(cand))
        if err_count < best_err:
            best_err = err_count
            best_proc = cand
    for orig in cl:
        final_map[orig] = best_proc

# 10. Write out final cleaned transcripts (excluding removed duplicates)
with open(output_txt, 'w', encoding='utf-8') as fout:
    for idx, row in df.iterrows():
        orig = row['transcript']
        if orig in removed_set:
            continue
        wav = data_path + row['wav_file']
        cleaned = final_map.get(orig, '')
        vifinal = clean_alignment(cleaned)  # final cleanup
        fout.write(f"{wav}|EN-default|EN|{vifinal}\n")

# 11. Save CSV of removed transcripts and flag original data
removed_df = df[df['transcript'].isin(removed_set)]
removed_df.to_csv('removed_texts.csv', index=False)
flagged = df.copy()
flagged['removed'] = flagged['transcript'].isin(removed_set)
flagged.to_csv('metadata_flagged.csv', index=False)
