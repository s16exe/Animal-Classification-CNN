import streamlit as st
import time  # Added for simulating program execution
import tensorflow as tf
import numpy as np
from test import predict_image
import json
# App title
def get_animal_info(animal_name: str) -> dict:

    file_path = 'description.json'
    with open(file_path, 'r') as file:
        animal_info = json.load(file)
    
    # Convert the animal name to lowercase to allow for case-insensitive search
    animal_name = animal_name.lower()
    
    # Search for the animal in the dictionary
    if animal_name in animal_info:
        return animal_info[animal_name]
    else:
        return {"error": f"Information for '{animal_name}' not found."}

st.title("Animal Identification")

# App description

st.write("Animal Identification and Information Web App with AI Nature Behaviour")

# File uploader
uploaded_file = st.file_uploader("Upload Animal Picture", type=["png", "jpg", "jpeg"])

# Display the uploaded file
if uploaded_file is not None:
    # Progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    # Circle buffering while running other program
    with st.spinner('Processing image, please wait...'):
        # Save uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        
        # Run image classification
        # time.sleep(2)
        predicted_class, confidence = predict_image("temp_image.jpg")
        
    desc = get_animal_info(predicted_class)
    st.success('Processing complete!')
    st.markdown("""
    <style>
    .custom-text {
        font-size: 64px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"**Predicted Class:** {predicted_class}")
    st.markdown(f"**Confidence:** {confidence:.2f}")


    st.markdown(f"**Description:** {desc['info']}") 
    st.markdown(f"**Lifespan:** {desc['lifespan']}")
    st.markdown(f"**Diet:** {desc['diet']}")
