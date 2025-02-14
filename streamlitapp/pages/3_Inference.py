import streamlit as st  
import numpy as np 
import tensorflow as tf  
from PIL import Image  
import pickle
import cv2 
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
st.set_page_config(page_title="Deep Learning App", layout="wide")

# Encode the image to base64
background_image_base64 = encode_image("background.jpg")  # Replace with your image path

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

st.title("🖼️ Model Inference")

st.write("Upload an image and select a model to get predictions.")

# Model selection dropdown
model_option = st.selectbox("Choose a Model:", ["ResNet", "MobileNet", "VGG16", "InceptionV3", "Xception", "EfficientNet","NAS Model", "Own Model 1", "Own Model 2"])

# Load model based on selection
MODEL_PATHS = {
    "ResNet": "Models/resnet_model.pkl",
    "MobileNet": "Models/mobilenet_model.pkl",
    "VGG16": "Models/vgg_model.pkl",
    "InceptionV3": "Models/Inception_model.pkl",
    "Xception": "Models/xception_model.pkl",
    "EfficientNet": "Models/EfficientNet_model.pkl",
    "NAS Model": "Models/nas_model.pkl",
    "Own Model 1": "Models/own1_model.pkl",
    "Own Model 2": "Models/own2_model.pkl"
}

# Load model
with open(MODEL_PATHS[model_option], 'rb') as file:
    model = pickle.load(file)

CLASS_NAMES = [
    'Arsenal', 'Aston Villa', 'Brentford', 'Brighton', 'Burnley',
    'Chelsea', 'Crystal Palace', 'Everton', 'Leeds', 'Leicester City',
    'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle',
    'Norwich', 'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves'
]


photo_option = st.selectbox("What type of photo will you upload?", ["Logo", "Jersey"])
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "webp"])


def preprocess_logo_image(image_input):
    if hasattr(image_input, "read"):
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to decode the image. Please check the file.")
    else:
        image = image_input
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    image = cv2.resize(image, (140, 140))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def classify_logo(model, image):
    processed_img = preprocess_logo_image(image)
    predictions = model.predict(processed_img)
    print("Raw Model Output:", predictions)
    softmax_probs = tf.nn.softmax(predictions).numpy()
    predicted_class_index = np.argmax(softmax_probs)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    return predicted_class_name


# ---------- Handling LOGO Images ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert the image to a NumPy array
    image_cv = np.array(image)

    if photo_option == "Logo":
        predicted_class= classify_logo(model, image_cv)  
        st.write(f'Predicted class for Logo: **{predicted_class}**')


    # ---------- Handling JERSEY Images ----------
    elif photo_option == "Jersey":
        def preprocess_image(image):
            image = cv2.resize(image, (140, 140))
            image = image.astype('float32') / 255.0
            image = np.expand_dims(image, axis=0)
            return image
        def sharpen_image(image):
            sharpening_kernel = np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]])
            sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
            return sharpened_image
        def extract_and_crop_top_n_contours(image, n=1):
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            edges = cv2.Canny(blurred_image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cropped_images = []

            for i in range(min(n, len(sorted_contours))):
                x, y, w, h = cv2.boundingRect(sorted_contours[i])
                cropped_img = image[y:y+h, x:x+w]
                resized_img = cv2.resize(cropped_img, (140, 140))
                sharpened_img = sharpen_image(resized_img)
                processed_img = preprocess_image(sharpened_img)      
                cropped_images.append(processed_img)
            return cropped_images
        def classify_images(model, processed_images):
            predictions = []
            for img in processed_images:
                pred = model.predict(img)  
                predicted_class_index = np.argmax(pred) 
                predictions.append(predicted_class_index)
            return predictions
        
        cropped_jerseys = extract_and_crop_top_n_contours(image_cv)
        if cropped_jerseys:
            predicted_class_indices = classify_images(model, cropped_jerseys)
            predicted_class_names = [CLASS_NAMES[idx] for idx in predicted_class_indices]

            predicted_class_str = ', '.join(predicted_class_names)
    
            st.write(f'Predicted class for Logo: **{predicted_class_str}**')

# Sidebar navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🔍 Make Predictions","🏠 Home", "🧠 Model Architecture","📊 Benchmarks & Stats", ])

if page == "🏠 Home":
    st.switch_page("Home.py")
elif page == "🧠 Model Architecture":
    st.switch_page("pages/1_Model Architecture.py")
elif page == "📊 Benchmarks & Stats":
    st.switch_page("pages/2_Benchmarks.py")