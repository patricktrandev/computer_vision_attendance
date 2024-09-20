import streamlit as st

import pandas as pd
import numpy as np
import cv2 as cv
import time
from streamlit_webrtc import webrtc_streamer
import av

from Home import face_rec
st.subheader("Real time Prediction ")

#retrieve db
with st.spinner("Retrieving data from Redis...."):
    name='company:register'
    redis_db=face_rec.retrieve_data(name)
    st.dataframe(redis_db)


#GET REAL TIME PREDICTION
#time setting
waitTime= 3

setTime= time.time()
realtimepred= face_rec.RealTimePred()




def video_frame_callback(frame):
    global setTime
    
    img = frame.to_ndarray(format="bgr24") #3d array
    #predict real time with frame
    pred_frame=realtimepred.face_prediction(img, redis_db, 'facial_features',['Name','Role'], thresh=0.5)

    #set time to reset saving logs
    time_now= time.time()
    difftime= time_now-setTime

    # current_log= realtimepred.get_current_logs()
    # if(current_log[0]!="Unknown" & current_log[0]!=""):
    #     print(current_log)
    #     realtimepred.saveLogs_redis()
    #     setTime=time.time()
    # check=False
    # current_log= realtimepred.get_current_logs()
    # if difftime>= waitTime:
    #     check= True
    # elif (current_log[0]!=""):
    #     check=True
    #     print(True)


    if difftime >= waitTime:
        
        realtimepred.saveLogs_redis()
        setTime=time.time()
        msg="Save data to redis..."
        print(msg)
        

    return av.VideoFrame.from_ndarray(pred_frame, format="bgr24")


webrtc_streamer(key="realTimePrediction", video_frame_callback=video_frame_callback)



# cap = cv.VideoCapture(0)

# while(True):
#     ret, frame = cap.read()
#     if ret==False:
#         print('cannot read camera')
#         break
        
#     pred_frame=face_rec.face_prediction(frame, retrieve_df, 'facial_features',['Name','Role'], thresh=0.5)
#     cv.imshow('frame', frame)
#     cv.imshow('prediction', pred_frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()