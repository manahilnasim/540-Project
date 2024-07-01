import json
import pathlib
import tarfile
import xgboost
import pandas as pd
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = xgboost.Booster()
    model.load_model("xgboost-model")

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    X_test = df.iloc[:, 1:].to_numpy()

    predictions = model.predict(xgboost.DMatrix(X_test))
    mse = mean_squared_error(y_test, predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse}
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
