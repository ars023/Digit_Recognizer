import numpy as np
import plotly.express as px
import streamlit as st
import ast
import matplotlib.pyplot as plt 
from pathlib import Path

st.set_page_config(page_title="Number Predictor", layout="wide")
st.title("KNN Model to predict hand-drawn digits")

ROOT = Path(__file__).resolve().parent           # repo root (where Predictor.py lives)
DATA_DIR = ROOT / "data"                         # repo_root/data

if not DATA_DIR.exists():
    st.error(f"Missing folder: {DATA_DIR}")
    st.stop()

# ---------- KNN helpers ----------
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def knn(train_data, train_labels, test_data, k):
    predictions = []
    for test_point in test_data:
        distances = [distance(test_point, train_point) for train_point in train_data]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [train_labels[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return np.array(predictions), k_indices

def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

def parse_array(a):
    return list(ast.literal_eval(a))

@st.cache_resource
def load_data():
    train_data = np.load(DATA_DIR / "train_data.npy")   # shape (N, 784) or (N, 28, 28)
    train_labels = np.load(DATA_DIR / "train_labels.npy")
    test_data = np.load(DATA_DIR / "test_data.npy")     # optional, used if you want to compare
    test_labels = np.load(DATA_DIR / "test_labels.npy")
    # Flatten if needed:
    if train_data.ndim == 3:
        train_data = train_data.reshape(len(train_data), -1)
    if test_data.ndim == 3:
        test_data = test_data.reshape(len(test_data), -1)
    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = load_data()


input = st.text_input("Sample", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None)
# ---------- Sidebar inputs ----------
st.sidebar.title("Parameters")
k = st.sidebar.number_input("Number of Neighbors (K)", min_value=1, value=3, max_value=10, step=1)


if (!input){
    st.markdown("Please provide a vector representing your number!")
}
else{
    #st.markdown(f"**Weight (slope):** {parse_array(input)}")
    sample = np.array(parse_array(input), dtype=float).reshape(1, 784)
    pred, indices = knn(train_data, train_labels, sample, k)
    
    
    st.write("### Your drawing (downsampled to 28×28):")
    
    # sample: shape (1, 784) or (784,)
    arr = sample.reshape(-1)            # flatten if needed
    img = arr.reshape(28, 28)
    
    # If your convention is 1 = black, 0 = white, invert for display:
    img_disp = img                # comment this out if 0=black,1=white
    
    st.image(img_disp, width=500, clamp=True)  # clamp=True for float [0,1]
    
    
    st.markdown(f"**Prediction:** {pred[0]}")
    
    st.write("### Nearest neighbors:")
    cols = st.columns(k)
    for i, idx in enumerate(indices):
        with cols[i]:
            st.image(train_data[idx].reshape(28, 28), width=100, clamp=True)
            st.caption(f"Label: {train_labels[idx]}")
}
st.markdown("---")


