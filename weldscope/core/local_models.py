"""Bundled ONNX inference. Loading never downloads a model or contacts a hub."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import BertTokenizer, PyOrtFunction

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"


class LocalModel:
    def __init__(self, model_name, **_kwargs):
        folder = MODEL_ROOT / model_name
        config = json.loads((folder / "config.json").read_text())
        self.max_length = config["max_length"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = PyOrtFunction.from_customop(
            BertTokenizer,
            vocab_file=str(folder / "vocab.txt"),
            do_lower_case=1,
            strip_accents=1,
        )
        self.tokenizer.extra_session_options.update(intra_op_num_threads=1, inter_op_num_threads=1)

    def tokens(self, text, second=None):
        ids, types, _, _ = self.tokenizer(
            np.array([text] if second is None else [text, second], dtype=object)
        )
        # Match BERT's longest-first pair truncation, preserving both SEP tokens.
        if len(ids) > self.max_length:
            if second is None:
                ids = np.concatenate([ids[: self.max_length - 1], [102]])
                types = np.zeros(len(ids), dtype=np.int64)
            else:
                split = int(np.flatnonzero(types == 1)[0])
                first, last = list(ids[1 : split - 1]), list(ids[split:-1])
                while len(first) + len(last) > self.max_length - 3:
                    (first if len(first) >= len(last) else last).pop()
                ids = np.array([101, *first, 102, *last, 102], dtype=np.int64)
                types = np.array([0] * (len(first) + 2) + [1] * (len(last) + 1))
        return {
            "input_ids": ids.reshape(1, -1).astype(np.int64),
            "token_type_ids": types.reshape(1, -1).astype(np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }

    def encode(self, texts, **_kwargs):
        single = isinstance(texts, str)
        values = []
        for text in [texts] if single else texts:
            inputs = self.tokens(text)
            hidden = self.session.run(None, inputs)[0]
            values.append(hidden.mean(axis=1)[0])
        result = np.asarray(values, dtype=np.float32)
        return result[0] if single else result

    def predict(self, pairs):
        return np.array(
            [
                float(self.session.run(None, self.tokens(first, second))[0].reshape(-1)[0])
                for first, second in pairs
            ]
        )
