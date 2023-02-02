import streamlit as st 
import os
from packages.chatbot.train import train



def train_ui():
    st.title("Train Chatbot Model")
    st.write("This is a tool to train a chatbot model.")

    intents = st.file_uploader("Upload your training data", type="json")

    #save intents file to temp folder
    if intents is not None:
        #Train the model
        trainor = train(intents)
        train_button = st.button("Train Model")
        if train_button:
            trainor.train()
            #Download Button
            st.download_button("Download Model File", data=trainor.model_file, file_name="data.pth", mime="application/octet-stream")


def sidebar():
    st.sidebar.title("NLP Toolset")
    st.sidebar.write("Please choose a tool to use:")
    options = ['Populate Database', 'Train Chatbot Model']
    choice = st.sidebar.selectbox("Tool", options)

    #Logic for each tool
    if choice == 'Populate Database':
        pass
    elif choice == 'Train Chatbot Model':
        train_ui()

def main():
    st.title("NLP Toolset")
    st.write("This is a toolset for NLP tasks like population your databse, training a chatbot model, etc.") 

    sidebar()

if __name__ == "__main__":
    main()