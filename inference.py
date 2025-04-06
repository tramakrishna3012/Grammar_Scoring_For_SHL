import torch
from happytransformer import TTSettings

beam_settings = TTSettings(num_beams=5, min_length=1, max_length=20)

def correct_text(model, tokenizer, text):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_text = "grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)