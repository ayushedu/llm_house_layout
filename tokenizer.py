import torch

class BlueprintTokenizer:
    def __init__(self, dataset_path="house_dataset.txt"):
        # Load data to build exact unique character list
        with open(dataset_path, "r") as f:
            self.text = f.read()
            
        self.chars = sorted(list(set(self.text)))
        # Append an explicit Padding Token for safety sequence operations
        if '[PAD]' not in self.chars:
            self.chars.append('[PAD]')
            
        self.vocab_size = len(self.chars)
        
        # Bi-directional integer lookup tables
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_id = self.char_to_id['[PAD]']

    def encode(self, text_string):
        """Converts strings into lists of token ID numbers."""
        return [self.char_to_id[ch] for ch in text_string if ch in self.char_to_id]

    def decode(self, id_list):
        """Converts integer lists back into viewable code text strings."""
        return "".join([self.id_to_char[i] for i in id_list if i in self.id_to_char and self.id_to_char[i] != '[PAD]'])

# Quick verification test
if __name__ == "__main__":
    tokenizer = BlueprintTokenizer()
    print(f"Unique vocabulary characters found: {tokenizer.vocab_size}")
    sample = '<svg width="400">'
    encoded = tokenizer.encode(sample)
    print(f"Encoded representation: {encoded}")
    print(f"Decoded verification:  '{tokenizer.decode(encoded)}'")
