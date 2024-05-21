import pandas as pd
from .train import load_model

class Model:
    def __init__(self, model_path: str):
        self.pipeline = load_model(model_path)

    def infer(self, x_row: dict[str, any]) -> float:
        """
        Score is infered from a dict with the correct schema, socore::float is returned
        """
        X_infer = pd.DataFrame.from_dict(x_row, orient="index").T
        y_pred_prob = self.pipeline.predict_proba(X_infer)
        return y_pred_prob[:, 1][0]
