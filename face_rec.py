import numpy as np
import pandas as pd
import cv2 as cv
import time
import os
from datetime import datetime
import redis

from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#connect redis db
host ='redis-19076.c240.us-east-1-3.ec2.cloud.redislabs.com'
portnumber= 19076
password='vpT497BSPNF8DA5yhmBf9hvZJ4X4dlIr'

r=redis.StrictRedis(host=host, port=portnumber, password=password)

#retrieve db
def retrieve_data(name):
    
    retrieve_db= r.hgetall(name)

    #get data
    retrieve_series= pd.Series(retrieve_db)

    #conver byte data to string
    retrieve_series= retrieve_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index= retrieve_series.index
    index= list(map(lambda x: x.decode(), index))

    retrieve_series.index= index
    retrieve_series

    #convert to dataframe
    retrieve_df= retrieve_series.to_frame().reset_index()
    retrieve_df.columns=['name_role', 'facial_features']

    #splir name and role and add to column
    retrieve_df[['Name', 'Role']]=retrieve_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_df[['Name','Role','facial_features']]


#config insight face
app_sc=FaceAnalysis(name="buffalo_sc", root="./insightface/buffalo_sc/buffalo_sc", providers=['CPUExecutionProvider'])
app_sc.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

#ML Search algorithm
def ml_search_algorithm(df,feature_col,test_vector,name_role=['Name','Role'], thresh=0.5):
    df_ref= df.copy()
    X_list=df[feature_col].tolist()
    x= np.asarray(X_list)
    cosine_sml=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    cosine_sml_arr= np.array(cosine_sml).flatten()
    df_ref['cosine']= cosine_sml_arr
    df_filter= df_ref.query(f'cosine >= {thresh}')
    
    
    if len(df_filter)>0:
        df_filter.reset_index(drop=True, inplace=True)
        argmax=df_filter['cosine'].argmax()
        name, role=df_filter.loc[argmax][name_role]
    else:
        name='Unknown'
        role='Unknown'
    return name, role

#display image
def display(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#save logs every 1 minute
class RealTimePred:
    def __init__(self):
        self.logs= dict(name=[],role=[],current_time=[])
    def reset_dict(self):
        self.logs=dict(name=[],role=[],current_time=[])
    def get_current_logs(self):
        #create dataframe logs
        dataframe= pd.DataFrame(self.logs)
        #drop duplicates
        dataframe.drop_duplicates('name', inplace=True)
        #push data to redis db
        #encode data
        name_list= dataframe['name'].tolist()
        role_list= dataframe['role'].tolist()
        return name_list


    def saveLogs_redis(self):
        #create dataframe logs
        dataframe= pd.DataFrame(self.logs)
        #drop duplicates
        dataframe.drop_duplicates('name', inplace=True)
        #push data to redis db
        #encode data
        name_list= dataframe['name'].tolist()
        role_list= dataframe['role'].tolist()
        ctime_list= dataframe['current_time'].tolist()

        encoded_data=[]
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name != "Unknown":
                concat_str= f"{name}@{role}@{ctime}"
                encoded_data.append(concat_str)
        
        if len(encoded_data)>0:
            
            r.lpush("attendance:logs",*encoded_data)

        self.reset_dict()




    def face_prediction(self,img,df_compress,feature_col,name_role=['Name','Role'], thresh=0.5):
        #extract time
        current_time=str(datetime.now())

        #extract feature
        res_i=app_sc.get(img)
        i_copy= img.copy()
        
        for i in res_i:
            x1,y1,x2,y2= i['bbox'].astype(int)
            ebds= i['embedding']
            person_name, person_role= ml_search_algorithm(df_compress, feature_col, test_vector=ebds,name_role=name_role,thresh=thresh)
            #print(person_name, person_role)
            #draw rectangle
            if person_name=="Unknown":
                color=(0,0,255)
            else:
                color=(255,0,0)
            cv.rectangle(i_copy, (x1,y1),(x2,y2),(0,255,0),2)
            text_gen= person_name
            cv.putText(i_copy,text_gen,(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            cv.putText(i_copy,current_time,(x1,y2+10),cv.FONT_HERSHEY_SIMPLEX,0.7,color,2)

            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)
        
        return i_copy
    

class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    def get_embedding(self,frame):
        results= app_sc.get(frame, max_num=1)
        ebd=None
        
        for i in results:
            self.sample+=1
            x1,y1,x2,y2= i['bbox'].astype(int)
            ebd= i['embedding']
            #ebd_list.append(ebd)
            text=f'samples={self.sample}'
            cv.putText(frame,text,(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
            cv.rectangle(frame, (x1,y1),(x2,y2),(0,255,0))
        return frame, ebd
    
    def save_data_redis(self,name, role):
        if name is not None:
            if name.strip()!="":
                key=f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        #check .txt
        if 'face_embd.txt' not in os.listdir():
            return 'file_false'


        #load txt file
        x_arr=np.loadtxt('face_embd.txt', dtype=np.float32)

        #convert to array
        rcv_sample= int(x_arr.size/512)
        x_arr= x_arr.reshape(rcv_sample,512)
        x_arr=np.asarray(x_arr)

        #call means embd
        x_mean= x_arr.mean(axis=0)
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes= x_mean.tobytes()
        # save to db

        r.hset(name='company:register',key=key,value=x_mean_bytes)

        os.remove('face_embd.txt')

        self.reset()

        return True

