import json
import pandas as pd
import os
import pickle
from sklearn.pipeline import Pipeline
from typing import List, NamedTuple


class ExpectedEventKeysFit(NamedTuple):
    model_parametrisation: dict
    train_cols: list[str]
    target: str
    data: any
    pipe: str
    pipeline: callable
    date: str


class ExpectedEventKeysPredict(NamedTuple):
    date: str
    users: str


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


def data_pipe(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.pipe(perim_modelo, 5)
        .pipe(col_to_datetime, "order_date")
        .pipe(col_to_datetime, "created_at")
    )
    return df


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


def verify_event_logic(
    train_cols: List[str], target: str, raw_df: pd.DataFrame
) -> bool:
    result = False
    try:
        cond_3 = isinstance(raw_df, pd.DataFrame)
        cond_1 = set(train_cols).issubset(set(raw_df.columns))
        cond_2 = target in raw_df.columns
        result = all[cond_1, cond_2, cond_3]
    except:
        pass
    return result


class Model:
    def __init__(self, name: str):
        self.model_name = name
        self.pipeline = None
        self.model_loaded = False
        self.expected_event_keys_fit = ExpectedEventKeysFit(
            "model_parametrisation",
            "train_cols",
            "target",
            "data",
            "pipe",
            "pipeline",
            "date",
        )
        self.expected_event_keys_pred = ExpectedEventKeysPredict("date", "users")

    def handler_fit(self, event, _):
        try:
            if set(self.expected_event_keys_fit).issubset(set(event.keys())):
                model_parametrisation = event[
                    self.expected_event_keys_fit.model_parametrisation
                ]
                train_cols = event[self.expected_event_keys_fit.train_cols]
                target = event[self.expected_event_keys_fit.target]
                raw_df = event[self.expected_event_keys_fit.data]
                pipe = event[self.expected_event_keys_fit.pipe]
                model_date = event[self.expected_event_keys_fit.date]

                verify_event_logic(train_cols, target, raw_df)

                prepared_df = raw_df.pipe(pipe)[[[train_cols]]]
                model_pipeline = event[self.expected_event_keys_fit.pipeline]
                model_pipeline.named_steps("classifier").set_params(
                    model_parametrisation
                )
                model_pipeline.fit(prepared_df)
                return {
                    "statusCode": "200",
                    "body": json.dumps(
                        {
                            "model_path": [f"models/push_{model_date}.pkl"],
                        }
                    ),
                }
            else:
                return {
                    "statusCode": "400",
                    "error": "Incomplete JSON format, missing keys on event fit",
                }
        except Exception as e:
            return {
                "statusCode": "400",
                "error": f"Unexpected error {e}",
            }

    example_infer = {"user_id": "prediction", "user_id2": "prediction"}

    def unload_model(self):
        try:
            self.pipeline = None
            self.model_loaded = False
        except:
            raise RuntimeError("There was an error unloading the model")

    def handler_predict(self, event: dict):
        try:
            if set(self.expected_event_keys_pred).issubset(set(event.keys())):
                model_date = event[self.expected_event_keys_pred.date]
                data_to_predict = pd.DataFrame.from_dict(
                    json.loads(event[self.expected_event_keys_pred.users])
                )
                if not self.model_loaded:
                    # load_model ya tiene cubierto el flujo de excepciones
                    self.pipeline = load_model(f"models/push_{model_date}.pkl")
                    self.model_loaded = True

                pipeline = self.pipeline
                predictions = pipeline.predict_proba(data_to_predict)[:, 1]
                infer_json = dict(zip(data_to_predict.index, predictions))
                return {
                    "statusCode": "200",
                    "body": json.dumps({"prediction": infer_json}),
                }
            else:
                return {
                    "statusCode": "400",
                    "error": "Incomplete JSON format, missing keys on event fit",
                }
        except Exception as e:
            return {
                "statusCode": "400",
                "error": e,
            }


if __name__ == "__main__":
    # Example
    DATA_PATH = "files/feature_frame.csv"
    proposed_date = "20240606"
    train_cols = [
        "user_order_seq",
        "normalised_price",
        "discount_pct",
        "global_popularity",
        "count_adults",
        "count_children",
        "count_babies",
        "count_pets",
        "people_ex_baby",
        "days_since_purchase_variant_id",
        "avg_days_to_buy_variant_id",
        "std_days_to_buy_variant_id",
        "days_since_purchase_product_type",
        "avg_days_to_buy_product_type",
        "std_days_to_buy_product_type",
        "ordered_before",
        "abandoned_before",
        "active_snoozed",
        "set_as_regular",
    ]
    df_base = pd.read_csv(DATA_PATH)
    df_base = df_base[train_cols].iloc[:5]
    model = Model(name="testing_model")
    model.pipeline = load_model(f"models/push_{proposed_date}.pkl")
    mock_predict_event = {"date": proposed_date, "users": df_base.to_json()}
    print(model.handler_predict(event=mock_predict_event))
