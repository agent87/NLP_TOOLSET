import streamlit as st
import pandas as pd
import sqlalchemy as db


def process_credentials(credentials):
    return 'postgresql+psycopg2' + credentials[8:]

def read_csv(csv_file):
    #show csv file on streamlit
    df = pd.read_csv(csv_file)
    return df

def populate_database(credentials, df):
    #create engine
    engine = db.create_engine(process_credentials(credentials))
    #create connection
    connection = engine.connect()
    #create table with name chats
    df.to_sql('chats', con=connection, if_exists='replace', index=False)

    #close connection
    connection.close()

    
def app():
    st.title("Populate Database")

    st.header("Provide your database credentials")
    #Credentials
    credentials = st.text_input("Credentials")
    #submit csv file
    st.header("Submit your csv file")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if st.button("Populate Database"):
        df = read_csv(csv_file)
        populate_database(credentials, df)
        st.success("Database populated successfully")



if __name__ == "__main__":
    app()