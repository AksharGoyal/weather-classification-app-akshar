import pandas as pd
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import streamlit as st
import joblib
import plotly.express as px
from PIL import Image
from keras.models import load_model

model = load_model('./weather_class_model.keras')
# model = joblib.load("./weather_class_model.joblib", mmap_mode=None)
decoder = joblib.load("./weather_encoder_model.joblib", mmap_mode=None)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Weather Classification", page_icon= '☁️', layout="wide")
st.markdown('# Weather Classification App')
st.markdown('## by Akshar Goyal')
labels = decoder.inverse_transform(list(range(5)))

def main():
    uploaded_file = st.file_uploader('Upload an image (make sure it has some kind of weather)', accept_multiple_files=False, type=['png', 'jpeg', 'jpg'])
    if uploaded_file:
        st.write("Uploaded file name:", uploaded_file.name)
        img = Image.open(uploaded_file)
        st.image(img, caption=uploaded_file.name, width=150)
        img_array = np.array(img)
        extension = uploaded_file.name.split('.')[-1]
        # st.write('File extension is ',extension)
        _, encoded_image = cv2.imencode( '.'+extension, img_array)
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        new_img_array = img_to_array(cv2.resize(decoded_image, (150, 150)))
        new_img_scaled = np.stack([new_img_array])/255
        # new_img = np.stack(cv2.resize(img_array, (150, 150)))/255 # resize them for the sake of consistency
        y = np.round(model.predict(new_img_scaled).reshape(-1),5)
        index = np.argmax(y)
        data = {'Weather':labels, 'Probability':y}
        df = pd.DataFrame(data)
        fig = px.bar(df, x='Weather', y='Probability', title='Weather Classification', text='Probability', text_auto='.2%', color='Probability', color_continuous_scale='Blues')
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update(layout_coloraxis_showscale=False)
        st.plotly_chart(fig)
        st.markdown(f'## The weather is :blue[{labels[index]}]')
        
if __name__ == '__main__':
    main()
    