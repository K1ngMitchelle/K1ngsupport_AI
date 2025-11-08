import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# Load model from local directory
model_path = "./model"

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    with open(os.path.join(model_path, "label_map.json"), "r") as f:
        label_map = json.load(f)
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer, reverse_label_map = None, None, {}

def predict(query):
    if not query.strip():
        return "Please enter a query", "0.000", "Waiting..."
    
    if model is None:
        return "Model not loaded", "0.000", "Error"
    
    # Tokenize and predict
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    
    category = reverse_label_map.get(pred_id, "Unknown")
    
    if confidence > 0.7:
        action = "AI Automated Response"
    else:
        action = "Human Agent"
    
    return category, f"{confidence:.3f}", action

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Customer Query", lines=3, placeholder="Enter your customer support question..."),
    outputs=[
        gr.Textbox(label="Category"),
        gr.Textbox(label="Confidence"), 
        gr.Textbox(label="Action")
    ],
    title="K1ngSupport AI - Customer Support Classifier",
    description="Enter a customer support question to see how the AI categorizes it and recommends action.",
    examples=[
        ["Where is my order?"],
        ["How do I return a product?"],
        ["Reset my password"],
        ["What are your shipping options?"],
        ["I need help with my account"]
    ]
)

if __name__ == "__main__":
    demo.launch()
