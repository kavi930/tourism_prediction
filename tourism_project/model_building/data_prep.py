
import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from huggingface_hub import upload_file # Import upload_file

# Define the Hugging Face repository ID for datasets
HF_DATASET_REPO_ID = "kavita89/tourism-dataset"

def run_data_preparation(data_path, output_dir, preprocessor_path):

    # ----------------------------
    # Load Dataset
    # ----------------------------
    df = pd.read_csv(data_path)

    # Remove unwanted columns
    drop_cols = ["Unnamed: 0", "CustomerID"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # ----------------------------
    # Separate Features & Target
    # ----------------------------
    X = df.drop("ProdTaken", axis=1)
    y = df["ProdTaken"]

    # ----------------------------
    # Identify Feature Types
    # ----------------------------
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # ----------------------------
    # Preprocessing Pipeline
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # ----------------------------
    # Train-Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ----------------------------
    # Transform Data
    # ----------------------------
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # ----------------------------
    # Apply SMOTE
    # ----------------------------
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed,
        y_train,
    )

    # ----------------------------
    # Save Outputs Locally and Upload to Hugging Face
    # ----------------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    # Save and upload preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Uploading {os.path.basename(preprocessor_path)}...")
    upload_file(
        path_or_fileobj=preprocessor_path,
        path_in_repo=os.path.basename(preprocessor_path),
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully uploaded {os.path.basename(preprocessor_path)}.")

    # Save and upload X_train_resampled.csv
    X_train_resampled_path = os.path.join(output_dir, "X_train_resampled.csv")
    pd.DataFrame(X_train_resampled).to_csv(X_train_resampled_path, index=False)
    print(f"Uploading X_train_resampled.csv...")
    upload_file(
        path_or_fileobj=X_train_resampled_path,
        path_in_repo="X_train_resampled.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully uploaded X_train_resampled.csv.")

    # Save and upload y_train_resampled.csv
    y_train_resampled_path = os.path.join(output_dir, "y_train_resampled.csv")
    pd.Series(y_train_resampled).to_csv(y_train_resampled_path, index=False)
    print(f"Uploading y_train_resampled.csv...")
    upload_file(
        path_or_fileobj=y_train_resampled_path,
        path_in_repo="y_train_resampled.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully uploaded y_train_resampled.csv.")

    # Save and upload X_test_processed.csv
    X_test_processed_path = os.path.join(output_dir, "X_test_processed.csv")
    pd.DataFrame(X_test_processed).to_csv(X_test_processed_path, index=False)
    print(f"Uploading X_test_processed.csv...")
    upload_file(
        path_or_fileobj=X_test_processed_path,
        path_in_repo="X_test_processed.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully uploaded X_test_processed.csv.")

    # Save and upload y_test.csv
    y_test_path = os.path.join(output_dir, "y_test.csv")
    pd.Series(y_test).to_csv(y_test_path, index=False)
    print(f"Uploading y_test.csv...")
    upload_file(
        path_or_fileobj=y_test_path,
        path_in_repo="y_test.csv",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Successfully uploaded y_test.csv.")

    print("Data preparation and upload completed successfully.")


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "python data_prep.py "
            "<input_csv> "
            "<output_directory> "
            "<preprocessor.pkl>"
        )
        sys.exit(1)

    data_path = sys.argv[1]
    output_dir = sys.argv[2]
    preprocessor_path = sys.argv[3]

    run_data_preparation(
        data_path,
        output_dir,
        preprocessor_path,
    )
