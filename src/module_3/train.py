import os
import logging
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from datetime import datetime

ORDER_DATE = "order_date"
DATA_PATH = "files/feature_frame.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join("logs", "train_model.log"),
    filemode="a",
)
logger = logging.getLogger()


def perim_modelo(df: pd.DataFrame, min_items: int = 5) -> pd.DataFrame:
    """
    Given a df that has columns id | order_id it generates a new dataframe which
    contains the orders that have more items than the argument min_items
    """
    assert {"variant_id", "order_id"}.issubset(set(df.columns))
    order_counts = df.groupby("order_id").size().reset_index(name="item_count")
    order_counts_filtered = order_counts[order_counts["item_count"] >= min_items]
    df = df.merge(order_counts_filtered, on="order_id", how="inner")
    return df


def col_to_datetime(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    assert col_name in df.columns
    try:
        df[col_name] = pd.to_datetime(df[col_name])
    except:
        print(f"Could not change types at {df.head()}, with column {col_name}")
    return df


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    else:
        df_base = pd.read_csv(path)
    return df_base


def get_scores(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, str]:
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return {"auc": roc_auc}


def retrain_model(
    model: Pipeline, X: pd.DataFrame, target_col: str, train_cols=list[str]
) -> Pipeline:
    assert target_col not in train_cols
    y = X[target_col]
    X = X[train_cols]
    model.fit(X, y)
    return model


def serialize_model(model: Pipeline, name: str, prefix_path=None) -> None:
    if prefix_path is not None:
        path = prefix_path + name
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path does not exist: {path}")
    else:
        prefix_path = "models/"
    with open(f"{prefix_path}{name}.pkl", "wb") as file:
        pickle.dump(model, file)


def load_model(path: str) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    try:
        with open(path, "rb") as f:
            loaded_pipeline = pickle.load(f)
    except EOFError:
        print("Error: The file seems to be empty or corrupted.")
    return loaded_pipeline


if __name__ == "__main__":
    logger.info("Training of model started")
    df_base = pd.read_csv(DATA_PATH)
    logger.info(f"Feature frame loaded. Path: {DATA_PATH}")
    target = "outcome"
    columnas_train = [
        "ordered_before",
        "abandoned_before",
        "avg_days_to_buy_variant_id",
        "global_popularity",
        "normalised_price",
    ]

    df_base = (
        df_base.pipe(perim_modelo, 5)
        .pipe(col_to_datetime, ORDER_DATE)
        .pipe(col_to_datetime, "created_at")
    )

    df_base.sort_values(by=ORDER_DATE, inplace=True)
    quantiles = df_base[ORDER_DATE].quantile([0.7, 0.9, 1.0])

    quantile_0 = df_base[ORDER_DATE].min()
    quantile_70 = quantiles.iloc[0]
    quantile_90 = quantiles.iloc[1]
    quantile_100 = quantiles.iloc[2]

    df_train = df_base[df_base[ORDER_DATE] <= quantile_70]
    df_val = df_base[
        (df_base[ORDER_DATE] > quantile_70) & (df_base[ORDER_DATE] <= quantile_90)
    ]
    df_test = df_base[df_base[ORDER_DATE] > quantile_90]

    logging.info(
        f"Data splits created:: {quantile_0}\n q_70: {quantile_70}\n q_90: {quantile_90}\n End: {quantile_100}"
    )

    X_train = df_train[columnas_train]
    y_train = df_train[target]

    X_val = df_val[columnas_train]
    y_val = df_val[target]

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("log_reg", LogisticRegression())]
    )
    pipeline.fit(X_train, y_train)
    scores_train = get_scores(pipeline, X_train, y_train)
    scores_val = get_scores(pipeline, X_val, y_val)

    logging.info(f"Train Scores: {scores_train}")
    logging.info(f"Validation Scores: {scores_val}")

    final_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("log_reg", LogisticRegression())]
    )

    final_pipeline = retrain_model(
        final_pipeline, X=df_base, target_col=target, train_cols=columnas_train
    )
    logging.info("Final model trained")
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d%H%M")
    serialize_model(
        model=final_pipeline, name=f"modelo_LR_supermercados_{formatted_datetime}"
    )
    logging.info(
        "Final model serialized at models/modelo_LR_supermercados_{formatted_datetime}.pkl"
    )
