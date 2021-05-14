import streamlit as st
import plotly.express as px
import numpy as np
import pickle
from load_data import load_iris as li
import time


 
# with st.spinner(text='In progress'):
#     time.sleep(5)
#     st.success('Done')

file = st.file_uploader('File uploader')

st.color_picker('pick a color')

st.title("My Awesome Flower Predictor")
st.header("We predict Iris types")
st.subheader("No joke")

# load data
df_iris = li()


st.plotly_chart(px.scatter(df_iris, 'sepal_width', 'sepal_length'))


show_df = st.checkbox("Do you want to see the data?")

if show_df: 
    df_iris

# get user flower input

s_l = st.number_input('Input the Sepal Length')
s_w = st.number_input('Input the Sepal Width')
p_l = st.number_input('Input the Petal Length')
p_w = st.number_input('Input the Petal Width')

user_values = np.array([s_l, s_w, p_l, p_w])

# load model

with open('saved-iris-model-2.pkl', 'rb') as f:
    model = pickle.load(f)

with st.echo():
    # this is my code
    prediction = model.predict(user_values.reshape(1, -1))
    prediction

# type(prediction)

# t = type(prediction)
# t

# st.write(type(prediction))

st.header(f'The model predicts: {prediction[0]}')

# st.balloons()



col1, col2, col3 = st.beta_columns(3)
with col1:
    'I am printing things'
with col2:
    df_iris
with col3:
    st.subheader("cool stuff")