import os


"""Configuration settings for TFT model."""
# WORKING_DIR = "/kaggle/input/demandForecasting"
WORKING_DIR = ""
TFT_CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "TFT", "checkpoints")
RAW = f"{WORKING_DIR}/data_raw/"
TFT_OUT_DIR = f"{WORKING_DIR}/TFT/data/"

# encoder features
ENC_VARS = [
    "sales",
    "transactions",
    "dcoilwtico",
    "onpromotion",
    "dow",
    "month",
    "weekofyear",
    "is_holiday",
    "is_workday"
]
# known future features
DEC_VARS = [
    "onpromotion",
    "dow",
    "month",
    "weekofyear",
    "is_holiday",
    "is_workday"
]
# static features
STATIC_COLS = [
    "store_nbr",
    "family",
    "state",
    "cluster"
    ]

REALS_TO_SCALE = [
    "transactions",
    "dcoilwtico"
    ]
