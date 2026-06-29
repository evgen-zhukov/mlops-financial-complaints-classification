from pathlib import Path
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.complaints_dataset import (
    DATASET_URL,
    create_balanced_sample,
    download_file,
    extract_zip,
    create_training_dataset,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ZIP_PATH = RAW_DATA_DIR / "complaints.csv.zip"
RAW_SAMPLE_CSV_PATH = RAW_DATA_DIR / "financial_complaints_sample.csv"
PROCESSED_SAMPLE_PARQUET_PATH = PROCESSED_DATA_DIR / "financial_complaints_sample.parquet"


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_file(DATASET_URL, ZIP_PATH)
    extracted_csv_path = extract_zip(ZIP_PATH, RAW_DATA_DIR)

    create_balanced_sample(
        input_csv_path=extracted_csv_path,
        output_csv_path=RAW_SAMPLE_CSV_PATH,
        output_parquet_path=PROCESSED_SAMPLE_PARQUET_PATH,
        samples_per_class=40,
    )

    training_dataset_path = PROJECT_ROOT / "data" / "processed" / "financial_complaints_training.csv"

    create_training_dataset(
        extracted_csv_path,
        training_dataset_path,
    )


if __name__ == "__main__":
    main()