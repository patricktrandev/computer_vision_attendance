import streamlit as st

st.set_page_config(page_title="Attendance System", layout='centered')
st.header("Attendance System using Face recognition")

with st.spinner("Loading Models and conecting to Redis...."):
    import face_rec

st.success("Model loaded sucessfully!")
st.success("Redis db is ready...")