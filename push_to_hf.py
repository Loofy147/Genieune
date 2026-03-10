import os
from huggingface_hub import HfApi

def push():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set.")
        return

    api = HfApi()
    repo_id = "LOOFYYLO/dynamic-entropy-genuineness-v2-2-space"

    # Force SDK change via README metadata if necessary (some HF versions require it)
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token
    )
    print("Forced SDK metadata update in README.md")

if __name__ == "__main__":
    push()
