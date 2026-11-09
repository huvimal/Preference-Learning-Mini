"""Train a small pairwise ranker, evaluate it, or score two new responses."""
import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
SEED = 42


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def words(text):
    return re.findall(r"\w+", normalize(text))


def overlap(prompt, response):
    prompt_words = set(words(prompt))
    return len(prompt_words & set(words(response))) / max(len(prompt_words), 1)


def load_pairs(path):
    records = []
    assigned = {}
    ids = set()
    required = ("id", "group", "category", "split", "prompt", "chosen", "rejected")
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError(f"Line {line_no}: every required field must be a non-empty string.")
        split = row["split"]
        if split not in ("train", "test") or row["id"] in ids:
            raise ValueError(f"Line {line_no}: invalid split or repeated id.")
        answers = tuple(sorted(normalize(row[key]) for key in ("chosen", "rejected")))
        if answers[0] == answers[1]:
            raise ValueError(f"Line {line_no}: chosen and rejected must differ.")
        for key in (("group", normalize(row["group"])), ("prompt", normalize(row["prompt"])), ("pair", answers)):
            previous_split = assigned.setdefault(key, split)
            if previous_split != split:
                raise ValueError(f"Line {line_no}: {key[0]} appears in both train and test.")
        ids.add(row["id"])
        records.append(row)
    train = [row for row in records if row["split"] == "train"]
    test = [row for row in records if row["split"] == "test"]
    if not train or not test:
        raise ValueError("Dataset needs both train and test examples.")
    return train, test


class Ranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.model = LogisticRegression(fit_intercept=False, solver="liblinear", random_state=SEED)

    def _features(self, prompts, answers):
        lexical = self.vectorizer.transform(answers)
        relevance = csr_matrix([[overlap(prompt, answer)] for prompt, answer in zip(prompts, answers)])
        return hstack((lexical, relevance), format="csr")

    def _difference(self, prompts, left, right):
        return self._features(prompts, left) - self._features(prompts, right)

    def fit(self, rows):
        prompts = [row["prompt"] for row in rows]
        chosen = [row["chosen"] for row in rows]
        rejected = [row["rejected"] for row in rows]
        self.vectorizer.fit(chosen + rejected)
        diff = self._difference(prompts, chosen, rejected)
        # Balance A/B labels by training on both orientations of each train pair.
        labels = np.concatenate((np.ones(len(rows)), np.zeros(len(rows))))
        self.model.fit(vstack((diff, -diff), format="csr"), labels)
        return self

    def margins(self, prompts, left, right):
        return self.model.decision_function(self._difference(prompts, left, right))


def pick(margin, tolerance=1e-10):
    if abs(margin) <= tolerance:
        return "tie"
    return "A" if margin > 0 else "B"


def summarize(rows):
    summary = {"pairs": len(rows)}
    for field in ("ranker", "longer_baseline"):
        correct = sum(row[field] == row["preferred"] for row in rows)
        ties = sum(row[field] == "tie" for row in rows)
        summary[field] = {
            "correct": correct,
            "incorrect": len(rows) - correct - ties,
            "ties": ties,
            "tie_adjusted_accuracy": (correct + 0.5 * ties) / len(rows) if rows else None,
        }
    return summary


def evaluate(ranker, rows):
    rng = random.Random(SEED)
    preferred = [rng.choice(("A", "B")) for _ in rows]
    left = [row["chosen"] if side == "A" else row["rejected"] for row, side in zip(rows, preferred)]
    right = [row["rejected"] if side == "A" else row["chosen"] for row, side in zip(rows, preferred)]
    prompts = [row["prompt"] for row in rows]
    margins = ranker.margins(prompts, left, right)
    reversed_margins = ranker.margins(prompts, right, left)
    categories, lengths = defaultdict(list), defaultdict(list)
    results = []
    for row, a, b, margin, reversed_margin, label in zip(rows, left, right, margins, reversed_margins, preferred):
        a_len, b_len = len(a.split()), len(b.split())
        chosen_len, rejected_len = len(row["chosen"].split()), len(row["rejected"].split())
        group = "shorter" if chosen_len < rejected_len else "longer" if chosen_len > rejected_len else "equal"
        result = {
            "id": row["id"], "category": row["category"], "preferred": label,
            "ranker": pick(margin), "longer_baseline": pick(a_len - b_len),
            "preferred_length": group, "words_a": a_len, "words_b": b_len,
            "margin_a_minus_b": float(margin),
            "swap_consistent": bool(np.isclose(margin, -reversed_margin, atol=1e-10, rtol=0)),
        }
        results.append(result)
        categories[row["category"]].append(result)
        lengths[group].append(result)
    unequal = [row for row in results if row["words_a"] != row["words_b"]]
    longer_picks = sum(row["ranker"] == row["longer_baseline"] for row in unequal)
    return {
        "overall": summarize(results),
        "by_category": {name: summarize(group) for name, group in sorted(categories.items())},
        "by_preferred_length": {name: summarize(group) for name, group in sorted(lengths.items())},
        "length_audit": {"unequal_length_pairs": len(unequal), "ranker_selected_longer": longer_picks},
        "swap_consistent_pairs": sum(row["swap_consistent"] for row in results),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/pairs.jsonl")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--predict", type=Path, help="JSON with prompt, answer_a and answer_b.")
    args = parser.parse_args()
    try:
        train, test = load_pairs(args.data)
        ranker = Ranker().fit(train)
        if args.predict:
            example = json.loads(args.predict.read_text(encoding="utf-8-sig"))
            names = ("prompt", "answer_a", "answer_b")
            if not isinstance(example, dict) or any(
                not isinstance(example.get(name), str) or not example[name].strip() for name in names
            ):
                raise ValueError("Prediction input needs non-empty prompt, answer_a and answer_b.")
            margin = float(ranker.margins([example["prompt"]], [example["answer_a"]], [example["answer_b"]])[0])
            print(json.dumps({"preferred": pick(margin), "margin_a_minus_b": margin}, indent=2))
            return
        report = {"train_pairs": len(train), "test_pairs": len(test), "seed": SEED,
                  "scikit_learn": sklearn.__version__, **evaluate(ranker, test)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Run incomplete: {error}\n")
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(json.dumps(report["overall"], indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
