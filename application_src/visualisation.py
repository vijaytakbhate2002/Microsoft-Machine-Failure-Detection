import streamlit as st
import os
from PIL import Image 
from application_src.demo_data import ShowData
import logging

def visual():   
    """
    Function for Data Visualization in Streamlit.

    This function fetches images from the specified directory and displays them in a Streamlit app for data visualization.

    It shows images using the 'st.image' method with column width adjustment.

    Parameters:
        None

    Returns:
        None
    """
    logging.debug("Data visualization initiated...")

    st.subheader('Data Visualization', divider='rainbow')
    ShowData().data_showroom()

    # Load images
    image_folder = "application_src\\graphs"
    image_paths = os.listdir(image_folder)
    # Display images
    for image_path in image_paths:
        image = Image.open(os.path.join(image_folder, image_path))
        st.image(image, use_column_width=True)

    logging.info("Data visualization completed.")
