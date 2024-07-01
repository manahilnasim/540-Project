import os
import pandas as pd

if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    train = pd.read_csv(f"{base_dir}/train1/train_data.csv", header=None)
    test = pd.read_csv(f"{base_dir}/test1/test_data_with_outcome.csv", header=None)
    validation = pd.read_csv(f"{base_dir}/val1/validation_data.csv", header=None)

    prod = test.iloc[round(len(test)*0.8):]
    test = test.iloc[:round(len(test)*0.8)]

    pd.DataFrame(train).to_csv(f"{base_dir}/train2/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{base_dir}/validation2/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test2/test.csv", header=False, index=False)
    pd.DataFrame(prod).to_csv(f"{base_dir}/prod2/prod.csv", header=False, index=False)
