import streamlit as st
import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import io

# Directory to save student images
if not os.path.exists('student_images'):
    os.makedirs('student_images')

st.title("Student Face Upload")

# Function to save the uploaded image
def save_uploaded_image(uploaded_file, name, index):
    try:
        # Read the file as bytes
        bytes_data = uploaded_file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(bytes_data, np.uint8)
        
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        student_folder = os.path.join('student_images', name)
        if not os.path.exists(student_folder):
            os.makedirs(student_folder)
        
        # Generate a filename based on index
        file_name = f"{name}_{index}.jpg"
        file_path = os.path.join(student_folder, file_name)
        
        cv2.imwrite(file_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return file_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

# Form to upload student images
with st.form("upload_form"):
    name = st.text_input("Student Name")
    uploaded_files = st.file_uploader("Choose files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    submit_button = st.form_submit_button("Upload")

if submit_button:
    if name and uploaded_files:
        successful_uploads = 0
        for index, uploaded_file in enumerate(uploaded_files):
            file_path = save_uploaded_image(uploaded_file, name, index)
            if file_path:
                successful_uploads += 1
                st.success(f"Image uploaded successfully: {file_path}")
            else:
                st.error(f"Failed to upload image {index + 1}")
        
        st.info(f"Uploaded {successful_uploads} out of {len(uploaded_files)} images for {name}.")
    else:
        st.error("Please provide a name and upload at least one image.")