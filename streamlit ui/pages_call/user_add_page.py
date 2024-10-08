import streamlit as st
from io import BytesIO
import zipfile
import os
from azure.storage.blob import BlobServiceClient
import datetime
import databricks.sql as dbsql
import uuid
# from Home import home
from src.login_call import UserData


def user_add():
    if st.session_state.login_flag:
        if st.session_state.login_user_type=="Admin":
            with st.form("plot_form"):
                email_id = st.text_input("Email ID",placeholder="xyz.d@affine.ai")
                user_pass=st.text_input("Password", placeholder="admin@123")
                company_name = st.selectbox("Company Name",options=["Affine1","Affine2"], placeholder="Affine")
                col1, col2 = st.columns(2)
                dept_name=col1.selectbox("Department Name",options=["kids wear","mens wear"], placeholder="kids waer")
                user_level=col2.selectbox("User Level",options=["Admin","General"], placeholder="kids waer")
                access_at=datetime.datetime.now()
                # Use st.form_submit_button to create a button for generating insight
                add_user = col1.form_submit_button(
                    "Add User")
                
            if add_user:
                user_metadata={"Email ID":email_id,
                                "Company Name":company_name,
                                "Department Name":dept_name,
                                "Access At":access_at,
                                "User Level":user_level,
                                "User Password":user_pass
                                }        
                # upload_user_details(user_metadata)
                UserData().register(user_metadata)
        else:
            st.error("You are not have admin access to add a new user.")
    else:
        with st.sidebar:
             st.info("Please Login first")