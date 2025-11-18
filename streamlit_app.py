import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

st.set_page_config(page_title="K1ngSupport AI", page_icon="🚀")
st.title("K1ngSupport AI - Customer Support Classifier")
st.write("Enter a customer support question to see how the AI categorizes it.")

# Load model
@st.cache_resource
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(".")
        tokenizer = AutoTokenizer.from_pretrained(".")
        with open("label_map.json", "r") as f:
            label_map = json.load(f)
        reverse_label_map = {v: k for k, v in label_map.items()}
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, reverse_label_map
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, {}

model, tokenizer, reverse_label_map = load_model()

query = st.text_area("Customer Query", placeholder="e.g., Where is my order? How do I reset my password?...", height=100)

if st.button("🚀 Classify Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a query")
    elif model is None:
        st.error("Model not loaded")
    else:
        with st.spinner("Analyzing query..."):
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_id].item()
            
            category = reverse_label_map.get(pred_id, "Unknown")
            action = "AI Automated Response" if confidence > 0.7 else "Human Agent"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", category)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("Action", action)

st.markdown("---")
st.markdown("Built with ❤️ by K1ngMitchelle")
