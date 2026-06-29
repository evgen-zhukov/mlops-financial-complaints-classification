from pathlib import Path
import logging
import zipfile

import pandas as pd
import requests


logger = logging.getLogger(__name__)

DATASET_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

SELECTED_PRODUCTS = [
    "Credit card",
    "Checking or savings account",
    "Mortgage",
    "Debt collection",
    "Credit reporting or other personal consumer reports",
]

TEXT_COLUMN = "Consumer complaint narrative"
LABEL_COLUMN = "Product"


def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        logger.info("File already exists: %s", output_path)
        return

    logger.info("Downloading dataset from %s", url)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    logger.info("Downloaded file to %s", output_path)


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    expected_csv_path = output_dir / "complaints.csv"

    if expected_csv_path.exists():
        logger.info("CSV file already exists: %s", expected_csv_path)
        return expected_csv_path

    logger.info("Extracting %s", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
        extracted_files = zip_ref.namelist()

    csv_files = [file for file in extracted_files if file.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("No CSV file found in the ZIP archive.")

    extracted_csv_path = output_dir / csv_files[0]
    logger.info("Extracted CSV file: %s", extracted_csv_path)

    return extracted_csv_path


def create_balanced_sample(
    input_csv_path: Path,
    output_csv_path: Path,
    output_parquet_path: Path,
    samples_per_class: int = 40,
    random_state: int = 42,
    chunk_size: int = 50_000,
) -> None:
    logger.info("Reading dataset in chunks from %s", input_csv_path)

    collected_parts: list[pd.DataFrame] = []
    counts_by_product = {product: 0 for product in SELECTED_PRODUCTS}

    use_columns = [TEXT_COLUMN, LABEL_COLUMN]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_csv_path,
            usecols=use_columns,
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = chunk.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        chunk = chunk[chunk[LABEL_COLUMN].isin(SELECTED_PRODUCTS)]

        if chunk.empty:
            continue

        chunk = chunk.sample(frac=1, random_state=random_state + chunk_number)

        selected_rows = []

        for product in SELECTED_PRODUCTS:
            already_collected = counts_by_product[product]
            remaining = samples_per_class - already_collected

            if remaining <= 0:
                continue

            product_rows = chunk[chunk[LABEL_COLUMN] == product].head(remaining)

            if not product_rows.empty:
                selected_rows.append(product_rows)
                counts_by_product[product] += len(product_rows)

        if selected_rows:
            collected_parts.append(pd.concat(selected_rows, ignore_index=True))

        logger.info(
            "Processed chunk %s | collected: %s",
            chunk_number,
            counts_by_product,
        )

        if all(count >= samples_per_class for count in counts_by_product.values()):
            logger.info("Collected enough samples for all selected products.")
            break

    if not collected_parts:
        raise ValueError("No matching records were found in the dataset.")

    sampled_df = pd.concat(collected_parts, ignore_index=True)

    sampled_df = sampled_df.rename(
        columns={
            TEXT_COLUMN: "complaint_text",
            LABEL_COLUMN: "product_category",
        }
    )

    sampled_df = sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sampled_df.insert(0, "record_id", range(1, len(sampled_df) + 1))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_df.to_csv(output_csv_path, index=False)
    sampled_df.to_parquet(output_parquet_path, index=False)

    logger.info("Saved CSV sample to %s", output_csv_path)
    logger.info("Saved Parquet sample to %s", output_parquet_path)
    logger.info("Final sample size: %s rows", len(sampled_df))
    logger.info("Final class distribution: %s", sampled_df["product_category"].value_counts().to_dict())

def create_training_dataset(
    input_csv_path: Path,
    output_csv_path: Path,
    samples_per_class: int = 500,
    random_state: int = 42,
) -> None:
    logger.info("Creating training dataset from %s", input_csv_path)

    use_columns = [TEXT_COLUMN, LABEL_COLUMN]

    chunks = pd.read_csv(
        input_csv_path,
        usecols=use_columns,
        chunksize=50000,
        low_memory=False,
    )

    collected = []

    for chunk in chunks:
        chunk = chunk.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        chunk = chunk[chunk[LABEL_COLUMN].isin(SELECTED_PRODUCTS)]

        collected.append(chunk)

    df = pd.concat(collected, ignore_index=True)

    sampled_df = (
        df.groupby(LABEL_COLUMN, group_keys=False)
        .sample(
            n=samples_per_class,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )

    sampled_df = sampled_df.rename(
        columns={
            TEXT_COLUMN: "complaint_text",
            LABEL_COLUMN: "product_category",
        }
    )

    sampled_df.insert(0, "record_id", range(1, len(sampled_df) + 1))

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_df.to_csv(output_csv_path, index=False)

    logger.info("Training dataset saved to %s", output_csv_path)
    logger.info("Training dataset size: %s", len(sampled_df))