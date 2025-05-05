import streamlit as st
import pickle
import numpy as np 

#import the model 

pipe= pickle.load(open('pipe.pkl','rb'))
df= pickle.load(open('df.pkl','rb'))

st.title("Mobile Price Predictor")

#Brand 
Brand = st.selectbox('Brand',df['Brand'].unique())

#type of Mobile
Model = st.selectbox('Mobile Category',df['Model'].unique())

#Color
color = st.selectbox('Color Choice',df['color'].unique())

#RAM
Mobile_RAM= st.selectbox('RAM (in GB)',[2,3,4,8,12,16])

#ROM
ROM= st.selectbox('ROM (in GB)',[2,3,4,8,12,16,32,64,128,256,1024])

#Display_Inch
Display_Inch=st.number_input('Display size in Inch')

#Mobile_Processor
Mobile_Processor=st.selectbox('Processor Choice',df['Mobile_Processor'].unique())

#Primary_rear_Camera
Primary_rear_Camera = st.selectbox('Primary Rear Camera (in MB)',[1.3,2,5,8,32,48,50,64,108,120,130,200])

#Secondary_rear_Camera
Secondary_rear_Camera = st.selectbox('Secondary Rear Camera (in MB)',[0,2,5,8,10,13,48,50])

#Number of rear Cameras
Number_of_rear_Cameras = st.selectbox('Number_of_rear_Cameras',[1,2,3,4])

#Front Camera
Front_Camera = st.selectbox('Front Camera (in MB)',[0,2,3,5,8,10,13,16,20,32,42,50])

#Warranty
Warranty_Available = st.selectbox('Warrenty ',['Yes','No'])

#AI_lens
AI_lens = st.selectbox('AI lens ',['Yes','No'])

#AI_lens
Front_Dual_Camera = st.selectbox('Front Dual Camera ',['Yes','No'])

#Battery
Battery_Capacity = st.selectbox('Battery size',[800,1000,2000,3000,4000,5000])

#Discount %
Discount_Percentage=st.number_input('Discount in Percentage')

#if st.button('Predict Price'):

#    if Warranty_Available == 'Yes':
#        Warranty_Available = 1
 #   else:

#    if AI_lens == 'Yes':
#        AI_lens = 1
#    else:
#        AI_lens=0  
#    
#    if Front_Dual_Camera == 'Yes':

#        Front_Dual_Camera = 1
#    else:
#        Front_Dual_Camera = 0  

    
#    query = np.array([Battery_Capacity,Discount_Percentage,Brand,Model,color,Mobile_RAM, ROM, Display_Inch, Mobile_Processor,Primary_rear_Camera, Secondary_rear_Camera,Number_of_rear_Cameras,AI_lens,Front_Camera,Front_Dual_Camera, Warranty_Available ])
#    query = query.reshape(1,16)
#    st.title(np.exp(pipe.step3.predict(query)))

if st.button('Predict Price'):

    # Convert categorical Yes/No to binary
    Warranty_Available = 1 if Warranty_Available == 'Yes' else 0
    AI_lens = 1 if AI_lens == 'Yes' else 0
    Front_Dual_Camera = 1 if Front_Dual_Camera == 'Yes' else 0

    # Create input array
    query = np.array([
        Battery_Capacity,
        Discount_Percentage,
        Brand,
        Model,
        color,
        Mobile_RAM,
        ROM,
        Display_Inch,
        Mobile_Processor,
        Primary_rear_Camera,
        Secondary_rear_Camera,
        Number_of_rear_Cameras,
        AI_lens,
        Front_Camera,
        Front_Dual_Camera,
        Warranty_Available
    ]).reshape(1, -1)

    # Predict directly (no np.exp needed)
    predicted_price = np.exp(pipe.predict(query)[0])
    st.title(f"Predicted Price: ₹{predicted_price:,.2f}")
