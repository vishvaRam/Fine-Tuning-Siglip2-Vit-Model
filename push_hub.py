import os
from huggingface_hub import HfApi

tok = "token"

repo_id = f"Vishva007/siglip2-finetune"
# token = os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi(token=tok)

try:
    api.create_repo(repo_id, private=False, exist_ok=True) # Added exist_ok for robustness
    print(f"Repo {repo_id} created or already exists")
except Exception as e:
    print(f"Error creating/accessing repo: {e}")
    exit() 

try:
    api.upload_folder(
        folder_path="siglip2-person-looking-finetuned/",
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="model",
        revision="main"
    )
    print(f"Model files uploaded to {repo_id}")
except Exception as e:
    print(f"Error uploading files: {e}")
