import streamlit as st 
from PIL import Image 
import base64
import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

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



st.title("📌 Model Architecture")

st.write("This page displays the architecture (topology) of the selected deep learning models.")

# Define available models, their corresponding image paths, and descriptions
model_options = {
    "ResNet": {
        "image": "architectures/resnet_architecture.png",
        "description": """
            ResNet (Residual Network) is a type of convolutional neural network (CNN) that uses skip connections 
            to allow gradients to flow through the network. This helps in training very deep networks effectively.
            
            - Shows the characteristic residual blocks
            - Has multiple conv_block units with skip connections
            - Includes batch normalization and activation functions
            - Shows increasing feature channels with depth
            - Has identity mappings between blocks
        """
    },
    "EfficientNet": {
        "image": "architectures/efficientnet_architecture.png",
        "description": """
            EfficientNet is a family of CNNs that achieve high accuracy while being computationally efficient. 
            It scales up the network width, depth, and resolution using a compound scaling method.
            
            - Shows optimized block structure
            - Has MBConv blocks (Mobile Inverted Bottleneck)
            - Includes squeeze-and-excitation blocks
            - Shows balanced scaling of network components
            - Has compound scaling visible in the architecture
        """
    },
    "Xception": {
        "image": "architectures/xception_architecture.png",
        "description": """
            Xception is an extension of Inception that replaces standard convolution with depthwise separable convolution. 
            This improves performance while maintaining efficiency.
            
            - Shows the entry, middle, and exit flow design
            - Has multiple block units connected with skip connections
            - Uses separable convolutions extensively
            - Shows activation functions after each convolution
            - Includes batch normalization throughout
        """
    },
    "VGG16": {
        "image": "architectures/vgg_architecture.png",
        "description": """
            VGG16 is a deep CNN known for its simplicity and depth, using small 3x3 filters throughout the network. 
            It has been widely used for image classification tasks.
            
            - Shows simple sequential structure
            - Has regular patterns of convolution blocks
            - Includes pooling layers for dimension reduction
            - Shows increasing filter numbers with depth
            - Ends with dense layers
        """
    },
    "MobileNet": {
        "image": "architectures/mobilenet_architecture.png",
        "description": """
            MobileNet is designed for mobile and edge devices, utilizing depthwise separable convolutions to reduce the 
            model size and computation while maintaining accuracy.
            
            - Shows multiple conv_dw (depthwise convolution) and conv_pw (pointwise convolution) layers
            - Has batch normalization (bn) after convolutions
            - Shows ReLU activations throughout
            - Ends with global average pooling and dense layers
            - Clear structure of depthwise separable convolutions pattern
        """
    },
    "NAS Model": {
        "image": "architectures/nas_architecture.png",
        "description": """
            NAS (Neural Architecture Search) Model is a neural network architecture that is automatically designed 
            using reinforcement learning or evolutionary algorithms. It can discover novel and efficient architectures.
            
            - Shows an automatically discovered architecture
            - Has cast_to_float32 and normalization at input
            - Includes convolution, pooling, and dense layers
            - Shows dropout for regularization
            - Leads to a classification head
        """
    },
    "Custom Model 1": {
        "image": "architectures/own1_architecture.png",
        "description": """
            Custom Model 1 is a user-defined architecture tailored for specific tasks or datasets, allowing for flexibility 
            in design and implementation.
            
            - Shows a simpler structure with conv2d layers, max pooling, and dense layers
            - Includes dropout layers and activation functions
            - Use flattening operations before dense layers
            - Has clear linear paths from input to output
        """
    },
    "Custom Model 2": {
        "image": "architectures/own2_architecture.png",
        "description": """
            Custom Model 2 is another user-defined architecture that may incorporate unique layers or configurations 
            based on project requirements.
            
            - Has a similar pattern to Custom Model 1 but with different layer configurations
            - Includes dropout layers and activation functions
            - Use flattening operations before dense layers
            - Has clear linear paths from input to output
        """
    },
}



# Create a dropdown for model selection
selected_model = st.selectbox("Select a model architecture:", list(model_options.keys()))

# Display the description of the selected model above the image
st.write(model_options[selected_model]["description"])

# Load and display the selected architecture image
architecture_img = Image.open(model_options[selected_model]["image"])

# Define the custom CSS
st.markdown("""
    <style>
        .center-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 900px;
        }
    </style>
""", unsafe_allow_html=True)
image_base64 = image_to_base64(architecture_img)
# Display the image using HTML with the 'center-image' class
st.markdown(f'''
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{image_base64}" class="center-image" />
        <div class="center-caption">{selected_model} Topology</div>
    </div>
''', unsafe_allow_html=True)



with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to:", ["🧠 Model Architecture","🏠 Home",  "📊 Benchmarks & Stats", "🔍 Make Predictions"])

# Redirect based on selection

if page == "🏠 Home":
    st.switch_page("Home.py")
elif page == "📊 Benchmarks & Stats":
    st.switch_page("pages/2_Benchmarks.py")
elif page == "🔍 Make Predictions":
    st.switch_page("pages/3_Inference.py")
