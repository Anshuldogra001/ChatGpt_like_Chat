import streamlit as st
import json
from keras.utils import pad_sequences
import numpy as np
import keras
from sklearn.preprocessing import LabelEncoder
import colorama
import random
import pickle
import base64
colorama.init()

with open("intents.json") as file:
    data = json.load(file)

def res(tag):
    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "I'm not sure how to respond to that."

def chat():
    model = keras.models.load_model('chat_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 20

    st.title("ChatBot Interface")

    # Chat container
    chat_container = st.empty()

    user_input = st.text_input("You:", "")

    if user_input.lower() == "quit":
        st.stop()

    if user_input:
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                            truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        bot_response = res(tag)

        # Concatenate messages with line breaks
        chat_container.text(f"You: {user_input}")
        chat_container.markdown(f"Bot: {bot_response}")

if __name__ == '__main__':
    st.sidebar.title("ChatBot Options")
    st.sidebar.info("This is a simple chatbot using Streamlit.")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app is a demonstration of a chatbot using Streamlit. "
        "The chatbot is trained to respond to specific patterns."
    )

    st.sidebar.title("ChatBot Controls")
    st.sidebar.text("Type your message in the text box and press Enter to chat.")
    st.sidebar.text("Type 'quit' to exit the chat.")

    chat()


    @st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    img_bg = get_img_as_base64("back_img_2.jpg")
    img_sidebar = get_img_as_base64("back_img.jpg")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{img_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("data:image/png;base64,{img_sidebar}");
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
