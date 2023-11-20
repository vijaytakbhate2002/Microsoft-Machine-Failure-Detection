import streamlit as st

def show_contact():
    st.subheader('Contact Details', divider='rainbow')
    
    # Display contact details
    st.write("Name: Vijay Dipak Takbhate")
    st.write("Email: vijaytakbhate20@gmail.com")
    st.write("Phone: +91 8767363681")

    # Load the PDF and convert it to images
    with st.expander("Vijay's Resume"):
        st.image('static\\Resume.jpg')
