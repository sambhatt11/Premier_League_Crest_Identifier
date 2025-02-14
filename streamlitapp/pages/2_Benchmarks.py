import streamlit as st 
import matplotlib.pyplot as plt 
import pickle
import os
import numpy as np 
import pandas as pd 
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Set the page configuration as the very first command
st.set_page_config(page_title="Deep Learning App", layout="wide")
# Encode the image to base64
background_image_base64 = encode_image("streamlitapp/background.jpg")  # Replace with your image path

# Create the background image CSS style with the base64-encoded string
background_image = f"url(data:image/jpeg;base64,{background_image_base64})"
st.markdown(f"""
    <style>
        .stApp {{
            position: relative;
            background-image: {background_image};
            background-size: cover;  /* Ensure the image covers the full screen */
            background-position: center center;  /* Center the image */
            background-attachment: fixed;  /* Keep the background fixed during scroll */
            height: 100vh;  /* Ensure the background covers the entire page */
        }}
    </style>
""", unsafe_allow_html=True)

# Custom CSS to apply the background image
st.markdown(f"""
    <style>
        .stApp {{
            background-image: {background_image};
            background-size: cover;  /* Ensure the image covers the full screen */
            background-position: center center;  /* Center the image */
            background-attachment: fixed;  /* Keep the background fixed during scroll */
        }}
    </style>
""", unsafe_allow_html=True)

 

st.title("📊 Model Comparison: Benchmarking Plots")
st.write("This page presents various plots to benchmark and compare models based on their training history data.")

# List of models and corresponding history file paths
model_names = [
    "ResNet", "MobileNet", "VGG16",
    "Xception", "EfficientNet","NAS Model", "Own Model 1", "Own Model 2"
]
history_files = {
    "ResNet": "History/resnet_history.pkl",
    "MobileNet": "History/mobilenet_history.pkl",
    "VGG16": "History/vgg_history.pkl",
    "Xception": "History/xception_history.pkl",
    "EfficientNet": "History/EfficientNet_history.pkl",
    "NAS Model": "History/nas_history.pkl",
    "Own Model 1": "History/own1_history.pkl",
    "Own Model 2": "History/own2_history.pkl"
}

# Select models to compare
selected_models = st.multiselect("Select models to compare:", model_names, default=model_names)

# Load model histories
histories = {}
for model in selected_models:
    if os.path.exists(history_files[model]):
        with open(history_files[model], "rb") as f:
            histories[model] = pickle.load(f)
    else:
        st.warning(f"History file for {model} not found.")

# ------------------------------------------------------------------------------
# 1. Learning Curves: Accuracy and Loss Over Epochs
# ------------------------------------------------------------------------------
st.subheader("Learning Curves: Accuracy and Loss")
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy curves for all selected models
for model, history in histories.items():
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    axs[0].plot(epochs, history["accuracy"], label=f"{model} Train")
    axs[0].plot(epochs, history["val_accuracy"], linestyle="--", label=f"{model} Val")
axs[0].set_title("Accuracy Over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Accuracy")
axs[0].legend(fontsize="small")

# Plot loss curves if available
loss_plotted = False
for model, history in histories.items():
    if "loss" in history and "val_loss" in history:
        epochs = np.arange(1, len(history["loss"]) + 1)
        axs[1].plot(epochs, history["loss"], label=f"{model} Train")
        axs[1].plot(epochs, history["val_loss"], linestyle="--", label=f"{model} Val")
        loss_plotted = True

if loss_plotted:
    axs[1].set_title("Loss Over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend(fontsize="small")
else:
    axs[1].set_visible(False)
    st.info("Loss curves not available (history files do not contain 'loss' and 'val_loss').")

st.pyplot(fig)

# ------------------------------------------------------------------------------
# 2. Final Validation Accuracy Bar Chart
# ------------------------------------------------------------------------------
st.subheader("Final Validation Accuracy Comparison")
fig2, ax2 = plt.subplots(figsize=(10, 5))
# Extract final validation accuracy for each model
final_val_acc = [
    histories[model]["val_accuracy"][-1] if model in histories else 0
    for model in selected_models
]
ax2.bar(selected_models, final_val_acc, color="skyblue")
ax2.set_ylabel("Final Validation Accuracy")
ax2.set_title("Final Validation Accuracy for Each Model")
ax2.set_ylim(0, 1)  # Assuming accuracy is in the range [0, 1]
st.pyplot(fig2)

# ------------------------------------------------------------------------------
# 3. Accuracy Gap Plot (Training Accuracy - Validation Accuracy)
# ------------------------------------------------------------------------------
st.subheader("Accuracy Gap (Train - Validation)")
fig3, ax3 = plt.subplots(figsize=(10, 5))
accuracy_gap = []
for model in selected_models:
    if model in histories:
        train_acc = histories[model]["accuracy"][-1]
        val_acc = histories[model]["val_accuracy"][-1]
        accuracy_gap.append(train_acc - val_acc)
    else:
        accuracy_gap.append(0)
ax3.bar(selected_models, accuracy_gap, color="salmon")
ax3.set_ylabel("Accuracy Gap")
ax3.set_title("Difference Between Final Training and Validation Accuracy")
st.pyplot(fig3)

# ------------------------------------------------------------------------------
# 4. Loss vs. Accuracy Scatter Plot (if Loss Data Available)
# ------------------------------------------------------------------------------
if any("loss" in history and "val_loss" in history for history in histories.values()):
    st.subheader("Validation Loss vs. Accuracy")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for model in selected_models:
        if model in histories and "loss" in histories[model] and "val_loss" in histories[model]:
            final_val_loss = histories[model]["val_loss"][-1]
            final_val_acc = histories[model]["val_accuracy"][-1]
            ax4.scatter(final_val_loss, final_val_acc, label=model, s=100)
            ax4.text(final_val_loss, final_val_acc, f" {model}", fontsize=9)
    ax4.set_xlabel("Final Validation Loss")
    ax4.set_ylabel("Final Validation Accuracy")
    ax4.set_title("Validation Loss vs. Accuracy")
    ax4.legend(fontsize="small")
    st.pyplot(fig4)
else:
    st.info("Validation loss data is not available for the scatter plot.")

# ------------------------------------------------------------------------------
# 5. Model Performance Summary Table
# ------------------------------------------------------------------------------
st.subheader("Model Performance Summary")

# Build a list of dictionaries for each model
performance_data = []
for model in selected_models:
    if model in histories:
        entry = {
            "Model": model,
            "Train Accuracy": histories[model]["accuracy"][-1],
            "Val Accuracy": histories[model]["val_accuracy"][-1],
        }
        # Include loss data if available; otherwise, use None or a placeholder
        if "loss" in histories[model] and "val_loss" in histories[model]:
            entry["Train Loss"] = histories[model]["loss"][-1]
            entry["Val Loss"] = histories[model]["val_loss"][-1]
        else:
            entry["Train Loss"] = None
            entry["Val Loss"] = None
        performance_data.append(entry)
    else:
        performance_data.append({
            "Model": model,
            "Train Accuracy": None,
            "Val Accuracy": None,
            "Train Loss": None,
            "Val Loss": None
        })

# Convert to a DataFrame and display as a table
performance_df = pd.DataFrame(performance_data)
st.table(performance_df)

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", [ "📊 Benchmarks & Stats","🏠 Home", "🧠 Model Architecture", "🔍 Make Predictions"])

# Redirect based on selection

if page == "🏠 Home":
    st.switch_page("Home.py")
elif page == "🧠 Model Architecture":
    st.switch_page("pages/1_Model Architecture.py")
elif page == "🔍 Make Predictions":
    st.switch_page("pages/3_Inference.py")
