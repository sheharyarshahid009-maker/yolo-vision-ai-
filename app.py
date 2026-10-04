"""
YOLO Vision AI - Stylish Object Detection 🎯
"""

import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="YOLO Vision AI", page_icon="🎯", layout="wide")

# ---------- Stylish CSS ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.main-title {
    font-size: 3.2rem; font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { text-align: center; color: #b0b0d0; font-size: 1.15rem; margin-bottom: 1.5rem; }
.stat-card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 1rem; text-align: center;
}
.stat-num { font-size: 2rem; font-weight: 800; color: #00f5a0; }
.stat-label { color: #b0b0d0; font-size: 0.85rem; }
.obj-chip {
    display: inline-block; border-radius: 20px; padding: 0.35rem 1rem;
    margin: 0.25rem; font-weight: 700; font-size: 0.9rem; color: #0f0c29;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
}
.footer { text-align: center; color: #707090; margin-top: 2rem; font-size: 0.85rem; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- Hero ----------
st.markdown('<div class="main-title">🎯 YOLO Vision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Tasveer upload karo — AI ki nazar se duniya dekho! 80 tarah ki cheezein, ek click me.</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.markdown('<div class="stat-card"><div class="stat-num">80</div><div class="stat-label">Object Classes</div></div>', unsafe_allow_html=True)
c2.markdown('<div class="stat-card"><div class="stat-num">⚡</div><div class="stat-label">Real-Time Speed</div></div>', unsafe_allow_html=True)
c3.markdown('<div class="stat-card"><div class="stat-num">🎯</div><div class="stat-label">YOLOv8 Powered</div></div>', unsafe_allow_html=True)

st.markdown("---")

uploaded = st.file_uploader("📤 Apni tasveer yahan upload karo", type=["jpg", "jpeg", "png", "webp", "jfif"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)

    with st.spinner("🔍 AI dekh raha hai..."):
        results = model(img, verbose=False)

    annotated = img.copy()
    detected = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            name = model.names[int(box.cls[0])]
            detected.append((name, conf))
            label = f"{name} {conf:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 245, 160), 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), (0, 245, 160), -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 12, 41), 2)

    col1, col2 = st.columns(2)
    col1.image(img, caption="📷 Original", use_container_width=True)
    col2.image(annotated, caption="🎯 AI Detection", use_container_width=True)

    if detected:
        st.markdown("### ✨ Pehchani hui cheezein")
        chips = "".join(
            f'<span class="obj-chip">{n} {c:.0%}</span>' for n, c in sorted(set(detected))
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Kuch nahi mila — koi saaf tasveer try karo!")

st.markdown('<div class="footer">Built with YOLOv8 + Streamlit &nbsp;|&nbsp; by Engineer Muhammad Shehryar Khan</div>', unsafe_allow_html=True)
