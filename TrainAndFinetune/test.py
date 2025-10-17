from transformers import AutoModel
model = AutoModel.from_pretrained("moka-ai/m3e-base", trust_remote_code=True)
model.save_pretrained("./models/m3e-base")
