from transformers import AutoTokenizer, AutoModel
import torch

def load_model_and_tokenizer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=get_env("HF_TOKEN"))
    model = AutoModel.from_pretrained(model_name, token=get_env("HF_TOKEN"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device
