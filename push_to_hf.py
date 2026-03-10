import os
from huggingface_hub import HfApi

def push():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set.")
        return

    api = HfApi()
    repo_id = "LOOFYYLO/dynamic-entropy-genuineness-v2-1"

    files_to_push = [
        "genuine_model.py",
        "train_v2_advanced.py",
        "README.md",
        "V2_2_TECHNICAL_REPORT.md",
        "analysis_results.json"
    ]

    print(f"Pushing to {repo_id}...")
    for file in files_to_push:
        if os.path.exists(file):
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                token=token
            )
            print(f"Pushed {file}")

if __name__ == "__main__":
    push()
