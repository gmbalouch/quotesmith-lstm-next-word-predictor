"""Export the trained Keras model to a NumPy .npz file (run this only after retraining).

Needs TensorFlow installed locally:  pip install tensorflow-cpu
Usage (from the project root):       python tools/export_weights.py
"""
import sys
from pathlib import Path

import numpy as np
from keras.models import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

model = load_model("model/lstm_model.keras", compile=False)
emb, lstm, dense = model.layers
E = emb.get_weights()[0]
K, R, b = lstm.get_weights()
W, c = dense.get_weights()
np.savez_compressed("model/weights.npz", embedding=E.astype("float32"), kernel=K.astype("float32"),
                    recurrent=R.astype("float32"), bias=b.astype("float32"),
                    dense_w=W.astype("float32"), dense_b=c.astype("float32"))

# verify the NumPy version matches Keras
from predictor import NextWordPredictor
p = NextWordPredictor()
x = np.random.randint(1, 9000, size=(1, model.input_shape[1])).astype("int32")
ref = model.predict(x, verbose=0)[0]
mine = p._forward(x[0].tolist())
print("max abs difference vs Keras:", float(np.abs(ref - mine).max()))
