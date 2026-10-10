# Introduction
Custom Generative Transformer architecture from scratch using PyTorch to act as an automated interior designer. 
Given a text prompt size limitation like [50x30], the model automatically calculates and places bedrooms, kitchens, bathrooms, and living rooms without clipping windows or overlapping walls. 
Since you are building it from scratch locally, it won’t be as fancy as other models in the market, but it will provide overall understanding.

# Exection Steps
1. Generate random data
```
python dataset_and_tokenizer.py
```
3. Train Model
```
python train_blueprint_llm.py
```
3. Generate Output
```
python generate_house.py
``` 
4. View output - open `ai_generated_layout.html` in browser
