import csv
import sys
from os import path

def get_model_dir():

    try:
        r = path.join(sys._MEIPASS, "model")
    except Exception:
        r = path.abspath(".")

    return r

