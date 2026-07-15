
import os
from huggingface_hub import HfApi
from google.colab import userdata

# Ensure the hosting directory exists
os.makedirs("tourism_project/hosting", exist_ok=True)


from huggingface_hub import HfApi
from google.colab import userdata

# Get Hugging Face token from Colab Secret
# Create a secret named: HF_TOKEN
HF_TOKEN = userdata.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

print("Uploading project to Hugging Face Space...")

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="kavita89/MLOps_Prediction",
    repo_type="space",
    path_in_repo="",
    ignore_patterns=[
        "__pycache__/*",
        "*.pyc",
        ".ipynb_checkpoints/*"
    ]
)

print("Upload completed successfully!")
print("Space URL:")
print("https://huggingface.co/spaces/kavita89/MLOps_Prediction")
