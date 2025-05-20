import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Function to generate a chaotic sequence using the logistic map
def logistic_map(r, x, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)  # Logistic map formula
        sequence.append(x)
    return np.array(sequence)

# Function to encrypt an image using chaotic sequence
def encrypt_image(image, x0, r=3.99):
    M, N = image.shape  # Get image dimensions
    num_pixels = M * N  # Total number of pixels
    
    # Generate chaotic sequence
    chaotic_seq = logistic_map(r, x0, num_pixels)
    chaotic_seq = (chaotic_seq * 255).astype(np.uint8).reshape(M, N)
    
    # Encrypt the image using XOR operation
    encrypted_image = cv2.bitwise_xor(image, chaotic_seq)
    return encrypted_image, chaotic_seq

# Function to decrypt an image using the same chaotic sequence
def decrypt_image(encrypted_image, chaotic_seq):
    decrypted_image = cv2.bitwise_xor(encrypted_image, chaotic_seq)  # XOR again to retrieve original image
    return decrypted_image

# Streamlit page configuration
st.set_page_config(page_title="Image Encryption & Decryption", page_icon="🔒", layout="centered")

# Custom styling for buttons
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stDownloadButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        /* Remove number input arrows */
        input[type=number] {
            -moz-appearance: textfield;
        }
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

# UI Title and Description
st.title("🔐 Image Encryption & Decryption using Logistic Map")
st.write("Secure your images with chaotic encryption")

# File upload functionality
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"], help="Choose an image to encrypt and decrypt")

# Ensure 'x0' is initialized in session state
if "x0" not in st.session_state:
    st.session_state["x0"] = "0.56789"  # Keep it as a string to match text_input

x0 = st.text_input("🔢 Enter initial value (x0)", value=st.session_state["x0"])

# Convert x0 to float safely
try:
    x0 = float(x0)
except ValueError:
    st.error("Invalid input for x0. Please enter a numeric value.")

r = st.slider("🎚 Select r value", min_value=3.5, max_value=4.0, value=3.99, step=0.01)

if uploaded_file:
    with st.spinner("Processing image..."):
        # Convert uploaded image to grayscale and resize
        image = Image.open(uploaded_file).convert("L")
        image = np.array(image)
        image = cv2.resize(image, (256, 256))
        
        st.image(image, caption="🖼 Original Image", use_container_width=True, channels="L")
        
        # Encrypt and decrypt the image
        encrypted_image, chaotic_seq = encrypt_image(image, x0, r)
        decrypted_image = decrypt_image(encrypted_image, chaotic_seq)
        
        # Display encrypted and decrypted images
        col1, col2 = st.columns(2)
        with col1:
            st.image(encrypted_image, caption="🔏 Encrypted Image", use_container_width=True, channels="L")
        with col2:
            st.image(decrypted_image, caption="🔓 Decrypted Image", use_container_width=True, channels="L")
        
        # Save encrypted and decrypted images locally
        cv2.imwrite("encrypted_image.jpg", encrypted_image)
        cv2.imwrite("decrypted_image.jpg", decrypted_image)
        
        st.success("✅ Image encryption & decryption completed!")
        
        # Download buttons for encrypted and decrypted images
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇ Download Encrypted Image", data=open("encrypted_image.jpg", "rb").read(), file_name="encrypted_image.jpg", mime="image/jpeg")
        with col2:
            st.download_button("⬇ Download Decrypted Image", data=open("decrypted_image.jpg", "rb").read(), file_name="decrypted_image.jpg", mime="image/jpeg")
