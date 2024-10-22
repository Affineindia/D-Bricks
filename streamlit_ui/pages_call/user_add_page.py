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
    """Function to add a new user"""
    # Check if the user is logged in
    if st.session_state.login_flag:
         # Check if the logged-in user is an admin
        if st.session_state.login_user_type=="Admin":
             # Create a form for user input
            with st.form("plot_form"):
                # Input for email ID
                email_id = st.text_input("Email ID",placeholder="xyz.d@affine.ai")
                # Input for password, hidden
                user_pass=st.text_input("Password", placeholder="admin@123")
                # Dropdown for company selection
                company_name = st.selectbox("Company Name",options=["Affine1","Affine2"], placeholder="Affine")
                col1, col2 = st.columns(2) # Create two columns for layout
                # Dropdown for department selection
                dept_name=col1.selectbox("Department Name",options=["kids wear","mens wear"], placeholder="kids waer")
                # Dropdown for user level selection
                user_level=col2.selectbox("User Level",options=["Admin","General"], placeholder="kids waer")
                # Record the current datetime for access
                access_at=datetime.datetime.now()
                # Create a button to submit the form and add a user
                add_user = col1.form_submit_button(
                    "Add User")
                
            if add_user: # If the add user button is pressed
                # Register the new user using UserData class
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
            st.error("You are not have admin access to add a new user.") # Show error message
    else:
        with st.sidebar:
             st.info("Please Login first") # Inform the user to log in