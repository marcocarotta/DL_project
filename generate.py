import torch
from torch.nn import functional as F

class TextGenerator:
    def __init__(self, model, dataset, block_size=100, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.stoi = dataset.get_stoi()
        self.itos = dataset.get_itos()
        self.vocab_size = len(self.stoi)
        self.block_size = block_size
    
    def generate(self, start_text, length=100, temperature=1.0):
        # Convert start_text to tensor
        input_indices = [self.stoi.get(char, 0) for char in start_text]
        input_tensor = torch.tensor(input_indices, dtype=torch.long, device=self.device).unsqueeze(0)
        
        generated_text = start_text
        
        for _ in range(length):
            if input_tensor.size(1) > self.block_size:
                input_tensor = input_tensor[:, -self.block_size:]
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_char_idx = torch.multinomial(probs, num_samples=1)
            
            next_char = self.itos[next_char_idx.item()]
            generated_text += next_char
            input_tensor = torch.cat([input_tensor, next_char_idx], dim=1)
        
        return generated_text

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")