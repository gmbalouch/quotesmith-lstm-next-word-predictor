"""Run: python smoke_test.py  — checks that the model loads and predicts."""
from predictor import NextWordPredictor

p = NextWordPredictor()
for t in ["What are you doing", "I am going to", "Life is"]:
    preds, enc = p.top_k(t, 3)
    print(f"{t!r} ->", [(w, round(s, 3)) for w, s in preds])
print(p.generate("I am going to", 10, temperature=0.8, seed=1)[0])
