# Importing Libraries

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import cv2
import sys
import time
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# Application :


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


class Application:

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            self.spell = SpellChecker() if SpellChecker is not None else None
        except Exception:
            self.spell = None

        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            raise RuntimeError("Unable to access camera. Please check camera permissions/device.")

        self.current_image = None
        self.current_image2 = None

        model_new_json_path = os.path.join(base_dir, "Models", "model_new.json")
        with open(model_new_json_path, "r") as json_file:
            self.model_json = json_file.read()

        self.loaded_model = build_model_from_legacy_json(self.model_json)
        self.loaded_model.load_weights(os.path.join(base_dir, "Models", "model_new.h5"))

        model_dru_json_path = os.path.join(base_dir, "Models", "model-bw_dru.json")
        with open(model_dru_json_path, "r") as json_file_dru:
            self.model_json_dru = json_file_dru.read()

        self.loaded_model_dru = build_model_from_legacy_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(os.path.join(base_dir, "Models", "model-bw_dru.h5"))

        model_tkdi_json_path = os.path.join(base_dir, "Models", "model-bw_tkdi.json")
        with open(model_tkdi_json_path, "r") as json_file_tkdi:
            self.model_json_tkdi = json_file_tkdi.read()

        self.loaded_model_tkdi = build_model_from_legacy_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(os.path.join(base_dir, "Models", "model-bw_tkdi.h5"))

        model_smn_json_path = os.path.join(base_dir, "Models", "model-bw_smn.json")
        with open(model_smn_json_path, "r") as json_file_smn:
            self.model_json_smn = json_file_smn.read()

        self.loaded_model_smn = build_model_from_legacy_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(os.path.join(base_dir, "Models", "model-bw_smn.h5"))

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(text="Suggestions :", fg="red", font=("Courier", 30, "bold"))

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

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
                key=lambda w: (
                    self.spell.word_probability(w),
                    -abs(len(w) - len(cleaned)),
                    w
                ),
                reverse=True
            )
            return ranked
        except Exception:
            return []

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok and frame is not None:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1:y2, x1:x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            predicts = self._safe_suggestions(self.word)

            if len(predicts) > 0:
                self.bt1.config(text=predicts[0], font=("Courier", 20))
            else:
                self.bt1.config(text="")

            if len(predicts) > 1:
                self.bt2.config(text=predicts[1], font=("Courier", 20))
            else:
                self.bt2.config(text="")

            if len(predicts) > 2:
                self.bt3.config(text=predicts[2], font=("Courier", 20))
            else:
                self.bt3.config(text="")

        self.root.after(5, self.video_loop)

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1), verbose=0)

        prediction = {}
        prediction['blank'] = result[0][0]

        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # LAYER 2
        if self.current_symbol in ('D', 'R', 'U'):
            prediction = {
                'D': result_dru[0][0],
                'R': result_dru[0][1],
                'U': result_dru[0][2]
            }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ('D', 'I', 'K', 'T'):
            prediction = {
                'D': result_tkdi[0][0],
                'I': result_tkdi[0][1],
                'K': result_tkdi[0][2],
                'T': result_tkdi[0][3]
            }
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ('M', 'N', 'S'):
            prediction1 = {
                'M': result_smn[0][0],
                'N': result_smn[0][1],
                'S': result_smn[0][2]
            }
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)

            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0
                    for j in ascii_uppercase:
                        self.ct[j] = 0
                    return

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):
        predicts = self._safe_suggestions(self.word)
        if len(predicts) > 0:
            self.word = ""
            self.str += " "
            self.str += predicts[0]

    def action2(self):
        predicts = self._safe_suggestions(self.word)
        if len(predicts) > 1:
            self.word = ""
            self.str += " "
            self.str += predicts[1]

    def action3(self):
        predicts = self._safe_suggestions(self.word)
        if len(predicts) > 2:
            self.word = ""
            self.str += " "
            self.str += predicts[2]

    def action4(self):
        predicts = self._safe_suggestions(self.word)
        if len(predicts) > 3:
            self.word = ""
            self.str += " "
            self.str += predicts[3]

    def action5(self):
        predicts = self._safe_suggestions(self.word)
        if len(predicts) > 4:
            self.word = ""
            self.str += " "
            self.str += predicts[4]

    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Application...")
    (Application()).root.mainloop()
