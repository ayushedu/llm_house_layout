import random
import os

def generate_single_house_string():
    """Generates a highly dense, token-friendly layout string that a small LLM can track perfectly."""
    split_y = random.randint(180, 220)
    top_split_x = random.randint(160, 240)
    bot_split_x = random.randint(160, 240)
    
    colors = {"LIVI": "#a2d2ff", "BED": "#ffc8dd", "KITC": "#b7e4c7", "BATH": "#fcd5ce"}
    
    # Static predictable layout order eliminates spatial confusion for the weights
    layout = [
        {"type": "LIVI", "x": 20, "y": 20, "w": top_split_x - 20, "h": split_y - 20},
        {"type": "BED",  "x": top_split_x, "y": 20, "w": 400 - top_split_x - 20, "h": split_y - 20},
        {"type": "KITC", "x": 20, "y": split_y, "w": bot_split_x - 20, "h": 400 - split_y - 20},
        {"type": "BATH", "x": bot_split_x, "y": split_y, "w": 400 - bot_split_x - 20, "h": 400 - split_y - 20}
    ]
    
    # We use a strict format: [50x30] ROOM x y w h ROOM x y w h ... [END]
    svg_tokens = ["[50x30]"] # Anchor prompt
    for r in layout:
        svg_tokens.append(f"{r['type']} {r['x']} {r['y']} {r['w']} {r['h']}")
    svg_tokens.append("[END]")
    
    return " ".join(svg_tokens)

class BlueprintTokenizer:
    def __init__(self, dataset_path="house_dataset.txt"):
        if not os.path.exists(dataset_path):
            self.words = ['[PAD]', '[50x30]', 'LIVI', 'BED', 'KITC', 'BATH', '[END]']
        else:
            with open(dataset_path, "r") as f:
                text = f.read()
            self.words = sorted(list(set(text.split())))
            if '[PAD]' not in self.words: self.words.append('[PAD]')
            
        self.vocab_size = len(self.words)
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.id_to_word = {i: w for i, w in enumerate(self.words)}
        self.pad_id = self.word_to_id['[PAD]']

    def encode(self, text_string):
        print('encoding...', text_string)
        return [self.word_to_id[w] for w in text_string.split() if w in self.word_to_id]

    def decode(self, id_list):
        return " ".join([self.id_to_word.get(i, '') for i in id_list if self.id_to_word.get(i, '') != '[PAD]'])

if __name__ == "__main__":
    print("Generating 2,000 anchored house layout samples...")
    with open("house_dataset.txt", "w") as f:
        for _ in range(2000):
            f.write(generate_single_house_string() + "\n")
