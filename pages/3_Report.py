import streamlit as st

from Home import face_rec
st.subheader("Report")

#retrieve logs data and show report
name='attendance:logs'
def load_logs(name,end=-1):
    #retrieve data from redis
    logs_list=face_rec.r.lrange(name, start=0, end=end)
    return logs_list

#tabs to show info
tab1, tab2=st.tabs(['Logs', 'Register Data'])
with tab1:
        
    if(st.button("Refresh Logs")):
        st.write(load_logs(name=name))
with tab2:

    if(st.button("Refresh Data")):
        with st.spinner("Retrieving data from Redis...."):
            name_db='company:register'
            redis_db=face_rec.retrieve_data(name_db)
            st.dataframe(redis_db[['Name','Role']])
