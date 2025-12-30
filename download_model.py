from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Nop659/dla_hw",
    filename="model_best.pth",
    local_dir="src/configs/best_model",
)

print("Saved to:", path)
