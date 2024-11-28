import torch
from torch.nn import functional as F

class TextGenerator:
    def __init__(self, model, dataset, block_size=100, device='cpu'):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
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

    def get_next_char(self, probs, p=0.9):
        """
        implement top p nucleus sampling.
        we only consider the most influent chars such that the sum of their probabilities is <= p
        """
        # Sort probabilities and corresponding original indices 
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask for cumulative probability <= p
        mask = cumulative_probs <= p
        if not torch.any(mask):
            # Ensure at least one token is considered
            mask[0] = True
        
        # Select the probabilities and indices within the top-p mass
        filtered_probs = sorted_probs[mask]
        filtered_indices = sorted_indices[mask]
        
        # Normalize the filtered probabilities 
        # Maybe i could not normalize but i didn't really understand the torch.multiomial what will output then
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        # Sample from the filtered distribution
        next_char_idx = torch.multinomial(filtered_probs, num_samples=1)
        return filtered_indices[next_char_idx]

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")