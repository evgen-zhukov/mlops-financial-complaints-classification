# mlops-financial-complaints-classification

## Project Overview

This project implements the first stage of an MLOps pipeline for financial complaint classification.


The goal of this homework is to demonstrate data management practices:
* infrasrtucture rollout in an AWS EC2 Instance + Docker
* dataset preparation
* data annotation with Label Studio
* dataset versioning with DVC
* object storage using MinIO
* reproducible project structure

The selected use case is classification of financial consumer complaints into product categories.

## Dataset

The source dataset is the public CFPB Consumer Complaint Database.
https://www.consumerfinance.gov/data-research/consumer-complaints/
The preparation script downloads the original CFPB dataset, extracts it, filters selected product categories, and creates a balanced sample.

Selected categories:

* Credit card
* Checking or savings account
* Mortgage
* Debt collection
* Credit reporting or other personal consumer reports

Prepared files:


data/raw/financial_complaints_sample.csv
data/processed/financial_complaints_sample.parquet


The sample contains 200 records, 40 records per category.


## Data Preparation

To prepare the dataset:


python scripts/prepare_dataset.py


The script performs the following steps:

1. Downloads the original CFPB complaints dataset.
2. Extracts the CSV file.
3. Filters complaints with non-empty complaint text.
4. Selects five financial product categories.
5. Creates a balanced sample (200 complains).
6. Saves the result as CSV and Parquet.

## Annotation Tool

The annotation tool used in this project is Label Studio.

Label Studio is running on an AWS EC2 instance using Docker.

The annotation task is a single-label text classification task.

Labeling interface:


<View>
  <Text name="complaint" value="$complaint_text"/>

  <Choices name="product_category" toName="complaint" choice="single">
    <Choice value="Credit card"/>
    <Choice value="Checking or savings account"/>
    <Choice value="Mortgage"/>
    <Choice value="Debt collection"/>
    <Choice value="Credit reporting or other personal consumer reports"/>
  </Choices>
</View>


The input file for Label Studio is:


data/raw/financial_complaints_sample.csv


## Annotation Versions

Two annotation exports were created from Label Studio:

data/labeled/project-1-at-2026-06-28-10-14-ddee4add.json
data/labeled/project-1-at-2026-06-28-10-28-804cb3f7.json

Version summary:

* v1: 22 labeled complaints
* v2: 27 labeled complaints

## Dataset Versioning

Dataset versioning is implemented with DVC.

DVC tracks the prepared dataset and annotation exports using `.dvc` metadata files.

Large data files are stored in MinIO, which is used as an S3-compatible object storage.

DVC remote:

s3://mlops-dataset/dvc


## Data Lineage

The data flow is:

CFPB source dataset
        ↓
prepare_dataset.py
        ↓
balanced CSV sample
        ↓
Label Studio annotation
        ↓
annotation export JSON
        ↓
DVC versioning
        ↓
MinIO object storage
