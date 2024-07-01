import json
import tarfile
import pickle
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = pickle.load(open("xgboost-model", "rb"))

    test_path = "/opt/ml/processing/test2/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    report_dict = {
        "classification_metrics": {
            "accuracy": acc,
        },
    }
    print(acc)

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
