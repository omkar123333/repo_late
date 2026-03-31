import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import cv2
import numpy as np
import operator
import time
from collections import deque
from string import ascii_uppercase

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_model_from_legacy_json(json_text):
    data = json.loads(json_text)
    layers_cfg = data.get("config", {}).get("layers", [])

    model = Sequential()
    for idx, layer in enumerate(layers_cfg):
        class_name = layer.get("class_name")
        cfg = layer.get("config", {})

        if class_name == "InputLayer":
            continue

        if class_name == "Conv2D":
            kwargs = dict(
                filters=cfg["filters"],
                kernel_size=tuple(cfg["kernel_size"]),
                strides=tuple(cfg.get("strides", [1, 1])),
                padding=cfg.get("padding", "valid"),
                activation=cfg.get("activation", None),
                data_format=cfg.get("data_format", "channels_last"),
                use_bias=cfg.get("use_bias", True),
                dilation_rate=tuple(cfg.get("dilation_rate", [1, 1])),
                name=cfg.get("name", f"conv2d_{idx}")
            )
            if "batch_input_shape" in cfg and cfg["batch_input_shape"]:
                shape = cfg["batch_input_shape"][1:]
                if shape and all(v is not None for v in shape):
                    kwargs["input_shape"] = tuple(shape)
            model.add(Conv2D(**kwargs))
        elif class_name == "MaxPooling2D":
            model.add(MaxPooling2D(
                pool_size=tuple(cfg.get("pool_size", [2, 2])),
                strides=tuple(cfg.get("strides", [2, 2])),
                padding=cfg.get("padding", "valid"),
                data_format=cfg.get("data_format", "channels_last"),
                name=cfg.get("name", f"maxpool_{idx}")
            ))
        elif class_name == "Flatten":
            model.add(Flatten(name=cfg.get("name", f"flatten_{idx}")))
        elif class_name == "Dense":
            model.add(Dense(
                units=cfg["units"],
                activation=cfg.get("activation", None),
                use_bias=cfg.get("use_bias", True),
                name=cfg.get("name", f"dense_{idx}")
            ))
        elif class_name == "Dropout":
            model.add(Dropout(
                rate=cfg.get("rate", 0.5),
                name=cfg.get("name", f"dropout_{idx}")
            ))
    return model

import sys
class SignLanguageBackend:
    def __init__(self):
       # self.base_dir = os.path.dirname(os.path.abspath(__file__))
       

        if getattr(sys, 'frozen', False):
            self.base_dir = sys._MEIPASS
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            self.spell = SpellChecker() if SpellChecker is not None else None
        except Exception:
            self.spell = None

        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.vs.set(cv2.CAP_PROP_FPS, 30)
        if not self.vs.isOpened():
            raise RuntimeError("Unable to access camera. Please check camera permissions/device.")

        with open(os.path.join(self.base_dir, "Models", "model_new.json"), "r") as f:
            self.loaded_model = build_model_from_legacy_json(f.read())
        self.loaded_model.load_weights(os.path.join(self.base_dir, "Models", "model_new.h5"))

        with open(os.path.join(self.base_dir, "Models", "model-bw_dru.json"), "r") as f:
            self.loaded_model_dru = build_model_from_legacy_json(f.read())
        self.loaded_model_dru.load_weights(os.path.join(self.base_dir, "Models", "model-bw_dru.h5"))

        with open(os.path.join(self.base_dir, "Models", "model-bw_tkdi.json"), "r") as f:
            self.loaded_model_tkdi = build_model_from_legacy_json(f.read())
        self.loaded_model_tkdi.load_weights(os.path.join(self.base_dir, "Models", "model-bw_tkdi.h5"))

        with open(os.path.join(self.base_dir, "Models", "model-bw_smn.json"), "r") as f:
            self.loaded_model_smn = build_model_from_legacy_json(f.read())
        self.loaded_model_smn.load_weights(os.path.join(self.base_dir, "Models", "model-bw_smn.h5"))

        self.ct = {'blank': 0}
        for ch in ascii_uppercase:
            self.ct[ch] = 0

        self.blank_flag = 0
        self.current_symbol = "Empty"
        self.word = ""
        self.sentence = ""

        self.last_inference_at = 0.0
        self.inference_interval = 0.12
        self.last_confidence = 0.0
        self.stable_window = deque(maxlen=7)
        self.accept_count = 4

    def _safe_suggestions(self, word):
        if self.spell is None:
            return []
        cleaned = (word or "").strip().lower()
        if not cleaned:
            return []
        try:
            candidates = self.spell.candidates(cleaned)
            if not candidates:
                return []
            ranked = sorted(
                candidates,
                key=lambda w: (self.spell.word_probability(w), -abs(len(w) - len(cleaned)), w),
                reverse=True
            )
            return ranked[:5]
        except Exception:
            return []

    def _predict_symbol(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        reshaped = (test_image.reshape(1, 128, 128, 1).astype("float32")) / 255.0

        result = self.loaded_model.predict(reshaped, verbose=0)
        result_dru = self.loaded_model_dru.predict(reshaped, verbose=0)
        result_tkdi = self.loaded_model_tkdi.predict(reshaped, verbose=0)
        result_smn = self.loaded_model_smn.predict(reshaped, verbose=0)

        prediction = {'blank': float(result[0][0])}
        idx = 1
        for ch in ascii_uppercase:
            prediction[ch] = float(result[0][idx])
            idx += 1

        ranked = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = ranked[0][0]
        confidence = ranked[0][1]

        if current_symbol in ('D', 'R', 'U'):
            p = {'D': float(result_dru[0][0]), 'R': float(result_dru[0][1]), 'U': float(result_dru[0][2])}
            p = sorted(p.items(), key=operator.itemgetter(1), reverse=True)
            current_symbol, confidence = p[0]

        if current_symbol in ('D', 'I', 'K', 'T'):
            p = {
                'D': float(result_tkdi[0][0]),
                'I': float(result_tkdi[0][1]),
                'K': float(result_tkdi[0][2]),
                'T': float(result_tkdi[0][3]),
            }
            p = sorted(p.items(), key=operator.itemgetter(1), reverse=True)
            current_symbol, confidence = p[0]

        if current_symbol in ('M', 'N', 'S'):
            p = {'M': float(result_smn[0][0]), 'N': float(result_smn[0][1]), 'S': float(result_smn[0][2])}
            p = sorted(p.items(), key=operator.itemgetter(1), reverse=True)
            if p[0][0] == 'S':
                current_symbol, confidence = p[0]

        self.last_confidence = float(confidence)
        self.stable_window.append(current_symbol)

        if len(self.stable_window) >= self.accept_count:
            freq = {k: 0 for k in set(self.stable_window)}
            for s in self.stable_window:
                freq[s] += 1
            stable_symbol = max(freq.items(), key=lambda x: x[1])[0]
            stable_count = freq[stable_symbol]

            if stable_count >= self.accept_count and self.last_confidence >= 0.40:
                self.current_symbol = stable_symbol

        if self.current_symbol not in self.ct:
            self.current_symbol = 'blank'
        self.ct[self.current_symbol] += 1

        if self.current_symbol == 'blank':
            for ch in ascii_uppercase:
                self.ct[ch] = 0

        if self.ct[self.current_symbol] > 20:
            for ch in ascii_uppercase:
                if ch == self.current_symbol:
                    continue
                if abs(self.ct[self.current_symbol] - self.ct[ch]) <= 8:
                    self.ct['blank'] = 0
                    for j in ascii_uppercase:
                        self.ct[j] = 0
                    return

            self.ct['blank'] = 0
            for ch in ascii_uppercase:
                self.ct[ch] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.sentence) > 0:
                        self.sentence += " "
                    self.sentence += self.word
                    self.word = ""
            else:
                if len(self.sentence) > 128:
                    self.sentence = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def read_processed_frame(self):
        ok, frame = self.vs.read()
        if not ok or frame is None:
            return None

        frame = cv2.flip(frame, 1)
        x1 = int(0.5 * frame.shape[1])
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = int(0.5 * frame.shape[1])

        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        now = time.time()
        if now - self.last_inference_at >= self.inference_interval:
            self._predict_symbol(res)
            self.last_inference_at = now
        return frame

    def get_status(self):
        suggestions = self._safe_suggestions(self.word)
        return {
            "character": self.current_symbol,
            "word": self.word,
            "sentence": self.sentence,
            "suggestions": suggestions,
            "confidence": round(float(self.last_confidence), 3)
        }

    def apply_suggestion(self, idx):
        suggestions = self._safe_suggestions(self.word)
        if 0 <= idx < len(suggestions):
            self.word = ""
            if self.sentence:
                self.sentence += " "
            self.sentence += suggestions[idx]

    def clear(self):
        self.word = ""
        self.sentence = ""

    def release(self):
        if self.vs is not None:
            self.vs.release()
