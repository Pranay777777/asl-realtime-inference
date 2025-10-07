import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# import helpers from your train.py
import train as T

DATA_DIR = Path("data/asl_alphabet/asl_alphabet_train")

def test_build_model_output_shape():
    model = T.build_model((64, 64, 3), num_classes=26)
    assert model.output_shape[-1] == 26

def test_datagen_one_batch():
    train_gen, valid_gen = T.make_datagens(val_split=0.1)
    classes = [chr(i) for i in range(ord("A"), ord("Z")+1)]

    flow = train_gen.flow_from_directory(
        directory=str(DATA_DIR),
        target_size=(64, 64),
        batch_size=8,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        classes=classes,
    )
    x, y = next(flow)
    assert x.shape == (8, 64, 64, 3)
    assert y.shape == (8, 26)
    assert (x >= 0).all() and (x <= 1).all()
