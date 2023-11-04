import streamlit as st
import pickle
import numpy as np


# lr = pickle.load(open('lr_model_28Oct.pkl','rb'))
model = pickle.load(open('dt_model.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

# rf = pickle.load(open('rf_model_28Oct.pkl','rb'))


st.title('Laptop Price Prediction')
st.header('Fill the details to predict laptop Price')


company = st.selectbox('Company',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('Ram in (GB)',[8, 16, 4,2, 12,6, 32,24,64])
weight = st.number_input('Weight(in kg)')
touchscreen = st.selectbox('Touchscreen',['No','Yes'])  # actual value is [0,1]
ips = st.selectbox('IPS',['No','Yes'])              # actual value is [0,1]
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
hdd = st.selectbox('HDD(GB)', [0,  500, 1000, 2000,   32,  128])
ssd = st.selectbox('SSD(GB)',[128, 0,256,512,32,64,1000,1024,16,768,180,240,8])
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())


if st.button('Predict Laptop Price'):
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=='Yes':
        ips=1
    else:
        ips=0
    test = np.array([company,type,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])
    test = test.reshape([1,11])

    st.success(model.predict(test)[0])