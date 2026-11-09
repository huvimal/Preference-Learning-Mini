# Preference Learning Mini

A CPU-only pairwise ranker for Vietnamese answers. It learns from labeled
preferences and compares its choices with a baseline that picks the longer
answer. The included 32 pairs are synthetic teaching examples. They contain
no private workplace data.

## Run on Windows

Use Python 3.11 or a compatible newer version.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ranker.py
Get-Content reports/evaluation.json -Encoding utf8
.\.venv\Scripts\python.exe ranker.py --predict data/example.json
```

The ranker is fit from the bundled training set on each invocation. The
prediction command prints A, B, or a tie and the signed model margin.
The margin is a relative score and has not been calibrated as confidence.

## How learning works

1. Fit character-level TF-IDF on the 20 training pairs only.
2. Add a feature measuring word overlap between the prompt and each answer.
3. Subtract the feature vectors for the two answers and fit a logistic model
   on both orientations of each training pair.
4. Evaluate on 12 held-out pairs from four categories. Labels and categories
   are used for evaluation, and are excluded from model inputs.

The scorer has no intercept. Swapping A and B changes the sign of its margin.
The report checks this on each held-out pair. Text features may still correlate
with length. This is supervised preference ranking with fixed text features.

## Evaluation

- **Longer baseline** counts whitespace-separated words and picks the longer
  response.
- **Tie-adjusted accuracy** gives 1 point for a correct choice and 0.5 for a tie.
  The report shows exact correct, incorrect, and tie counts separately.
- **Length audit** reports how often the ranker selected the longer response
  among pairs of unequal length. Read it alongside performance when the
  preferred response is shorter or longer.
- **Category breakdown** covers format, math, grounding and coding. Each
  category has three test pairs, so individual mistakes move scores sharply.

The fixed split is labeled in `data/pairs.jsonl`. The loader rejects group,
normalized prompt, or unordered answer-pair overlap between train and test.
These checks do not detect all paraphrases or shared templates. The held-out
examples were written for the same small teaching dataset, so this evaluation
does not establish performance on real users, new domains, or unfamiliar tasks.

## Pilot results

| Method | Correct choices | Accuracy |
| --- | ---: | ---: |
| Always pick the longer answer | 4/12 | 33.3% |
| Trained pairwise ranker | 9/12 | 75.0% |

The ranker got 8/8 right when the preferred answer was shorter and 1/4
when it was longer. Neither method produced ties. Swapping A and B reversed
the score margin consistently across all 12 held-out synthetic pairs.

In `pair-08`, the prompt requested exactly two bullet points. The ranker
preferred the answer containing only one tool name, missing an explicit
format requirement. The pilot uses 20 training pairs and 12 test pairs.

## Extend with your own data

Keep the same JSONL schema with id, group, category, split, prompt, chosen and
rejected fields. Group related prompts together and place a whole group in
one split. Use self-created or authorized examples, and hold back a new final
test set if you change the model based on this report.

```powershell
.\.venv\Scripts\python.exe ranker.py --data private_data/my-pairs.jsonl --output reports/my-run.json
```

For a portfolio write-up, show a specific model error, the length breakdown,
and the results of both methods. A result below the baseline is useful when
paired with a concrete explanation of the failure.
