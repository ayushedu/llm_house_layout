import random

def generate_single_house_string():
    """Generates a randomized, syntactically clean SVG house floor plan string."""
    # Base layout size
    canvas_w, canvas_h = 400, 400
    
    # Randomize split between top and bottom floors
    split_y = random.randint(180, 220)
    
    # Randomize room widths
    top_split_x = random.randint(160, 240)
    bot_split_x = random.randint(160, 240)
    
    # Style hex codes
    colors = {
        "living": ["#a2d2ff", "#bde0fe"],
        "bed":    ["#ffafcc", "#ffc8dd"],
        "kitchen":["#b7e4c7", "#d8f3dc"],
        "bath":   ["#fcd5ce", "#ffe5d9"]
    }
    
    # Shuffle destinations so houses don't all look identical
    top_rooms = ["living", "bed"]
    random.shuffle(top_rooms)
    bot_rooms = ["kitchen", "bath"]
    random.shuffle(bot_rooms)
    
    # Build structural data dictionaries
    layout = [
        # Top Left Room
        {"type": top_rooms[0], "x": 20, "y": 20, "w": top_split_x - 20, "h": split_y - 20, "color": random.choice(colors[top_rooms[0]])},
        # Top Right Room
        {"type": top_rooms[1], "x": top_split_x, "y": 20, "w": canvas_w - top_split_x - 20, "h": split_y - 20, "color": random.choice(colors[top_rooms[1]])},
        # Bottom Left Room
        {"type": bot_rooms[0], "x": 20, "y": split_y, "w": bot_split_x - 20, "h": canvas_h - split_y - 20, "color": random.choice(colors[bot_rooms[0]])},
        # Bottom Right Room
        {"type": bot_rooms[1], "x": bot_split_x, "y": split_y, "w": canvas_w - bot_split_x - 20, "h": canvas_h - split_y - 20, "color": random.choice(colors[bot_rooms[1]])}
    ]
    
    # Construct raw SVG line by line without linebreaks inside the sample
    # (We compress it to a single line per house so it's easy for the LLM data loader)
    
    # Introduce random realistic dimensions to the training sample
    width_ft = random.choice([30, 40, 50, 60])
    length_ft = random.choice([25, 30, 35, 40])

    # Start the text sample with an explicit dimension token anchor
    svg = f"[{width_ft}x{length_ft}]"
    svg += f'<svg width="400" height="400" xmlns="http://w3.org"><house>'
    svg = f'<svg width="{canvas_w}" height="{canvas_h}" xmlns="http://w3.org">'
    svg += '<house>'
    
    for r in layout:
        name = r["type"].upper()[:4] # Keep text labels predictably short (LIVI, BED, KITC, BATH)
        svg += f'<rect x="{r["x"]}" y="{r["y"]}" w="{r["w"]}" h="{r["h"]}" fill="{r["color"]}" stroke="#333" stroke-w="3"/>'
        # Put text labels roughly in the center of the calculated box
        text_x = r["x"] + 15
        text_y = r["y"] + 30
        svg += f'<text x="{text_x}" y="{text_y}" font-s="14">{name}</text>'
        
    svg += '</house></svg>'
    return svg

# Generate 2,000 samples for our training set
num_samples = 2000
print(f"Generating {num_samples} synthetic house blueprints...")

with open("house_dataset.txt", "w") as f:
    for _ in range(num_samples):
        house_string = generate_single_house_string()
        f.write(house_string + "\n")

print("Dataset successfully created as 'house_dataset.txt'!")
