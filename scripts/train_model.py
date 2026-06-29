from pathlib import Path
import argparse
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "financial_complaints_training.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--run-name", type=str, default="tfidf-logreg-baseline")

    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_experiment("financial-complaints-classification")

    df = pd.read_csv(DATASET_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        df["complaint_text"],
        df["product_category"],
        test_size=0.2,
        random_state=42,
        stratify=df["product_category"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, args.ngram_max),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model_type", "TF-IDF + Logistic Regression")
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_range", f"1-{args.ngram_max}")
        mlflow.log_param("C", args.c)
        mlflow.log_param("dataset", str(DATASET_PATH))
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("test_size", 0.2)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        report = classification_report(y_test, y_pred)

        report_path = PROJECT_ROOT / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")

        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="financial_complaints_classifier",
        )

        print("Training completed")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")


if __name__ == "__main__":
    main()