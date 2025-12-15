# Variables for TFT model
# encoder features
enc_vars = [
    "sales", "transactions", "dcoilwtico", "onpromotion",
    "dow", "month", "weekofyear", "is_holiday", "is_workday",
]
# known future features
dec_vars = [
    "onpromotion", "dow", "month", "weekofyear",
    "is_holiday", "is_workday",
]
# static features
static_cols = ["store_nbr", "family", "state", "cluster"]

reals_to_scale = ["transactions", "dcoilwtico"]