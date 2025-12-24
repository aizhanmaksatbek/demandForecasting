import os
import pandas as pd
from config.settings import RAW, TFT_OUT_DIR

os.makedirs(TFT_OUT_DIR, exist_ok=True)


def load_csv(name, parse_dates=None):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


def make_calendar(date_min, date_max, date_col="date"):
    cal = pd.DataFrame({date_col: pd.date_range(date_min, date_max, freq="D")})
    cal["dow"] = cal[date_col].dt.dayofweek.astype("int8")
    cal["month"] = cal[date_col].dt.month.astype("int8")
    cal["weekofyear"] = cal[date_col].dt.isocalendar().week.astype("int16")
    return cal


def make_holidays(hol):
    hol = hol.copy()
    hol["transferred"] = hol["transferred"].fillna(False).astype(bool)
    hol["is_holiday"] = 0
    hol.loc[(hol["type"].isin(["Holiday", "Additional"])) &
            (~hol["transferred"]), "is_holiday"] = 1
    hol["is_workday"] = (hol["type"] == "Work Day").astype("int8")
    hol = (hol[["date", "is_holiday", "is_workday"]]
           .groupby("date", as_index=False).max())
    return hol


def preprocess():
    train = load_csv("train.csv", parse_dates=["date"])
    assert train is not None, "Missing train.csv in data/raw"
    stores = load_csv("stores.csv")
    oil = load_csv("oil.csv", parse_dates=["date"])
    trans = load_csv("transactions.csv", parse_dates=["date"])
    hol = load_csv("holidays_events.csv", parse_dates=["date"])

    df = train.copy()

    # full grid (store, family, date)
    date_min, date_max = df["date"].min(), df["date"].max()
    stores_u = df[["store_nbr"]].drop_duplicates()
    family_u = df[["family"]].drop_duplicates()
    full = (stores_u.assign(key=1)
            .merge(family_u.assign(key=1), on="key", how="outer")
            .drop("key", axis=1)
            )
    cal = make_calendar(date_min, date_max, "date")
    full = (full.assign(key=1)
            .merge(cal.assign(key=1), on="key", how="outer")
            .drop("key", axis=1)
            )
    df = full.merge(df, on=["date", "store_nbr", "family"], how="left")

    # store metadata
    if stores is not None:
        stores = stores.rename(columns={"type": "store_type"})
        df = df.merge(stores[["store_nbr", "state", "store_type", "cluster"]],
                      on="store_nbr", how="left"
                      )

    # transactions
    if trans is not None:
        df = df.merge(trans[["date", "store_nbr", "transactions"]],
                      on=["date", "store_nbr"], how="left"
                      )

    # oil price (forward fill)
    if oil is not None:
        df = df.merge(oil, on="date", how="left")
        df["dcoilwtico"] = df["dcoilwtico"].ffill()

    # holidays
    if hol is not None:
        hol_flags = make_holidays(hol)
        df = df.merge(hol_flags, on="date", how="left")

    # fillna
    df["sales"] = df["sales"].fillna(0.0)
    df["onpromotion"] = df["onpromotion"].fillna(0.0)
    df["transactions"] = df["transactions"].fillna(0.0)
    df["dcoilwtico"] = df["dcoilwtico"].fillna(method="ffill").fillna(0.0)
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")
    df["is_workday"] = df["is_workday"].fillna(0).astype("int8")

    # enforce types
    num_cols = ["sales", "onpromotion", "transactions", "dcoilwtico"]
    df[num_cols] = df[num_cols].astype("float32")

    df = df.sort_values(["store_nbr", "family", "date"])
    out = os.path.join(TFT_OUT_DIR, "panel.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} shape={df.shape}")


if __name__ == "__main__":
    preprocess()
