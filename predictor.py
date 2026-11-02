"""Next-word prediction engine (no Streamlit code here, so it is easy to test)."""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent / "model"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass
class Encoded:
    ids: list[int]
    unknown: list[str]


class NextWordPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        # Imported here so importing this module stays cheap.
        from keras.models import load_model

        with open(model_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        self.word_index: dict[str, int] = vocab["word_index"]
        self.num_words: int = vocab["num_words"]
        self.max_len: int = vocab["max_len"]
        self.index_word = {i: w for w, i in self.word_index.items()}
        self.model = load_model(model_dir / "lstm_model.keras", compile=False)
        self.vocab_size = self.model.output_shape[-1]

    # ---------- text -> ids ----------
    @staticmethod
    def clean(text: str) -> list[str]:
        """Same cleaning as training: lowercase, strip ALL punctuation."""
        return text.lower().translate(_PUNCT_TABLE).split()

    def encode(self, text: str) -> Encoded:
        ids, unknown = [], []
        for w in self.clean(text):
            i = self.word_index.get(w)
            if i is not None and i < self.num_words:
                ids.append(i)
            else:
                unknown.append(w)
        return Encoded(ids, unknown)

    def _pad(self, ids: list[int]) -> np.ndarray:
        ids = ids[-self.max_len:]  # keep the most recent words
        arr = np.zeros((1, self.max_len), dtype="int32")  # 'pre' padding with 0
        if ids:
            arr[0, -len(ids):] = ids
        return arr

    # ---------- prediction ----------
    def probabilities(self, ids: list[int]) -> np.ndarray:
        p = self.model.predict(self._pad(ids), verbose=0)[0].astype("float64")
        p[0] = 0.0  # index 0 is padding, never a real word
        return p / p.sum()

    def top_k(self, text: str, k: int = 5) -> tuple[list[tuple[str, float]], Encoded]:
        enc = self.encode(text)
        if not enc.ids:
            return [], enc
        p = self.probabilities(enc.ids)
        best = np.argsort(p)[::-1][:k]
        return [(self.index_word[int(i)], float(p[i])) for i in best if int(i) in self.index_word], enc

    def _sample(self, p: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> int:
        if temperature <= 0:
            return int(np.argmax(p))
        keep = np.argsort(p)[::-1][:top_k]
        logits = np.log(p[keep] + 1e-12) / temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(rng.choice(keep, p=probs))

    def generate(self, text: str, num_words: int = 10, temperature: float = 0.8,
                 top_k: int = 10, seed: int | None = None) -> tuple[str, Encoded]:
        enc = self.encode(text)
        if not enc.ids:
            return text, enc
        rng = np.random.default_rng(seed)
        ids, out = list(enc.ids), []
        for _ in range(num_words):
            i = self._sample(self.probabilities(ids), temperature, top_k, rng)
            word = self.index_word.get(i)
            if not word:
                break
            out.append(word)
            ids.append(i)
        return (text.strip() + " " + " ".join(out)).strip(), enc
