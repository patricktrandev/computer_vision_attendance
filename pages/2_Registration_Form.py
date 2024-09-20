import streamlit as st
import numpy as np
from Home import face_rec
import cv2 as cv
from streamlit_webrtc import webrtc_streamer
import av


st.subheader("Registration Form")

#init registrationform class
registration_form= face_rec.RegistrationForm()


#collect person name and role
#FORM
person_name=st.text_input(label="Name", placeholder="First and Last Name")
role= st.selectbox(label="Select Your Role", options=('Student','Teacher'))
#COLLECT FACIAL EMBEDDING




def video_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img, embd=registration_form.get_embedding(img)

    #save data into local computer
    if embd is not None:
        with open('face_embd.txt', mode='ab') as f:
            np.savetxt(f,embd)

    #save to redis

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_callback)



#SAVE DATA TO REDIS


if st.button('Submit'):
    st.info(f'Person Name: {person_name} -- Role: {role}' ,icon="ℹ️")

    return_val=registration_form.save_data_redis(person_name,role)
    if return_val==True:
        st.success(f"{person_name} register sucessfully!")
    elif return_val=="name_false":
        st.error('Name cannot be empty!')
    elif return_val =='file_false':
        st.error('File not found. Please refresh your page!')


    
