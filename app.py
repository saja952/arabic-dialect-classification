import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Arabic Dialect Classifier",
    page_icon="🗣️",
    layout="centered"
)

MODEL_DIR = "saja-hamasha/arabic-dialect-classifier"
device = "cuda" if torch.cuda.is_available() else "cpu"

DIALECT_NAME = {
    "E": "🇪🇬 مصري (Egyptian)",
    "G": "🇬🇨 خليجي (Gulf)",
    "J": "🇯🇴 أردني / شامي (Jordanian)",
    "Y": "🇾🇪 يمني (Yemeni)"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        subfolder="best_model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        subfolder="best_model"
    ).to(device)

    model.eval()

    id2label = model.config.id2label
    id2label = {int(k): v for k, v in id2label.items()}

    return tokenizer, model, id2label


tokenizer, model, id2label = load_model()

def predict(text, max_len=192):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    label_char = id2label[pred_id]
    dialect_full = DIALECT_NAME.get(label_char, label_char)

    return label_char, dialect_full, probs


st.title("🗣️ Arabic Dialect Classifier / تصنيف اللهجات العربية")
st.write("Model: MARBERT fine-tuned for dialect classification")

st.markdown("### اللهجات المتوقعة:")
for k in ["E", "G", "J", "Y"]:
    st.write(f"- **{DIALECT_NAME[k]}**")

text = st.text_area(
    "اكتب نص باللهجة العربية:",
    height=120,
    placeholder="مثال: شو الأخبار اليوم؟"
)

if st.button("اتوقع اللهجة"):
    if not text.strip():
        st.warning("يرجى إدخال نص أولاً.")
    else:
        label_char, dialect_full, probs = predict(text)


        st.success(f"اللهجة المتوقعة: **{dialect_full}**")

        st.subheader("احتمالات كل لهجة:")
        for i, p in enumerate(probs):
            char = id2label[i]
            name = DIALECT_NAME.get(char, char)
            st.write(f"{name}: {p:.3f}")
            st.progress(float(p))
