
from huggingface_hub import create_repo, upload_file

repo_id = "kavita89/tourism-dataset"

# Create the dataset repository
create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True
)

# Upload the CSV
upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type="dataset",
)

print(f"Dataset uploaded: https://huggingface.co/datasets/{repo_id}")
