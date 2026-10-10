import torch
import torch.nn.functional as F
import sys
from dataset_and_tokenizer import BlueprintTokenizer
from train_blueprint_llm import BlueprintTransformer

def compile_tokens_to_svg(token_string):
    """Compiles clean structural coordinate tokens into a browser-compliant HTML layout file."""
    colors = {"LIVI": "#a2d2ff", "BED": "#ffc8dd", "KITC": "#b7e4c7", "BATH": "#fcd5ce"}
    
    # Fixed the standard W3C namespace URL so browsers recognize the vector canvas layout
    html_output = '<svg width="400" height="400" xmlns="http://w3.org">\n'
    
    words = token_string.split()
    for i in range(1, len(words) - 4, 5):
        try:
            room_type = words[i]
            x, y, w, h = words[i+1], words[i+2], words[i+3], words[i+4]
            color = colors.get(room_type, "#ffffff")
            
            # Draw standard shapes directly without the custom '<house>' container tag blocking it
            html_output += f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}" stroke="#333" stroke-width="3"/>\n'
            html_output += f'  <text x="{int(x)+15}" y="{int(y)+30}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#333">{room_type}</text>\n'
        except (ValueError, IndexError):
            continue 
            
    html_output += "</svg>"
    return html_output


def generate_house_layout(prompt_text, max_new_tokens=25, temperature=0.6):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BlueprintTokenizer("house_dataset.txt")
    
    model = BlueprintTransformer(vocab_size=tokenizer.vocab_size, max_seq_len=25).to(DEVICE)
    model.load_state_dict(torch.load("blueprint_llm_weights.pth", map_location=DEVICE))
    model.eval()
    
    input_ids = tokenizer.encode(prompt_text)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    generated_ids = input_ids.copy()
    
    print("\n--- Autoregressive Data Sequence Decoding Stream ---")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Simple Top-k filtering to lock down syntax choices
            v, _ = torch.topk(next_token_logits, 2)
            next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token_id)
            
            new_word = tokenizer.id_to_word.get(next_token_id, '')
            sys.stdout.write(f"{new_word} ")
            sys.stdout.flush()
            
            if new_word == "[END]":
                break
                
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
            input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
            
    raw_tokens_output = tokenizer.decode(generated_ids)
    clean_compiled_html = compile_tokens_to_svg(raw_tokens_output)
    
    output_filename = "ai_generated_layout.html"
    with open(output_filename, "w") as f:
        f.write(clean_compiled_html)
    print(f"\n\n[Success] Flawless layout blueprint rendered and stored to: '{output_filename}'")

if __name__ == "__main__":
    generate_house_layout(prompt_text="[50x30]")
