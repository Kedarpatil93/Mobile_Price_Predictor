import streamlit as st
import pickle
import numpy as np 

#import the model 
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load pipeline and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Page config and header
st.set_page_config(page_title="Mobile Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📱 Mobile Price Predictor</h1>", unsafe_allow_html=True)

# Optional: App logo
st.image("https://cdn-icons-png.flaticon.com/512/3845/3845813.png", width=100)

# UI Styling
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stSelectbox > label {
            font-weight: bold;
        }
        .stSlider > label {
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    Brand = st.selectbox('Brand', df['Brand'].unique())
    Model = st.selectbox('Mobile Category', df['Model'].unique())
    color = st.selectbox('Color Choice', df['color'].unique())
    Mobile_RAM = st.selectbox('RAM (in GB)', [2,3,4,8,12,16])
    ROM = st.selectbox('ROM (in GB)', [2,3,4,8,12,16,32,64,128,256,512,1024])
    Display_Inch = st.number_input('Display size in Inch', min_value=3.0, max_value=7.5, step=0.1)

with col2:
    Mobile_Processor = st.selectbox('Processor Choice', df['Mobile_Processor'].unique())
    Primary_rear_Camera = st.selectbox('Primary Rear Camera (in MB)', [1.3,2,5,8,32,48,50,64,108,120,130,200])
    Secondary_rear_Camera = st.selectbox('Secondary Rear Camera (in MB)', [0,2,5,8,10,13,48,50])
    Number_of_rear_Cameras = st.selectbox('Number of Rear Cameras', [1,2,3,4,6])
    Front_Camera = st.selectbox('Front Camera (in MB)', [0,2,3,5,8,10,13,16,20,32,42,50])
    Battery_Capacity = st.selectbox('Battery Capacity (mAh)', [800,1000,2000,3000,4000,5000])
    Discount_Percentage = st.slider('Discount (%)', 0, 100, 10)

# Additional binary options
Warranty_Available = st.radio("Warranty Available", ["Yes", "No"], index=0, horizontal=True)
AI_lens = st.radio("AI Lens Enabled", ["Yes", "No"], index=0, horizontal=True)
Front_Dual_Camera = st.radio("Front Dual Camera", ["Yes", "No"], index=1, horizontal=True)

# Prediction
if st.button('🔍 Predict Price'):
    with st.spinner('Predicting...'):

        # Encode binary inputs
        Warranty_Available = 1 if Warranty_Available == 'Yes' else 0
        AI_lens = 1 if AI_lens == 'Yes' else 0
        Front_Dual_Camera = 1 if Front_Dual_Camera == 'Yes' else 0

        # Create query
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

        # Predict and show result
        predicted_price = np.exp(pipe.predict(query)[0])
        st.success(f"🎯 Predicted Price: ₹{predicted_price:,.2f}")

# Optional chart
if st.checkbox('📊 Show RAM vs Price Chart'):
    df_sample = df[['Mobile_RAM', 'Price']].dropna()
    fig, ax = plt.subplots()
    ax.scatter(df_sample['Mobile_RAM'], np.exp(df_sample['Price']), alpha=0.6, color='green')
    ax.set_xlabel("RAM (in GB)")
    ax.set_ylabel("Price (₹)")
    ax.set_title("RAM vs Price Distribution")
    st.pyplot(fig)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ by Kedar</p>", unsafe_allow_html=True)
