import streamlit as st
import databricks.sql as dbsql
from src.login_call import UserData
import streamlit as st



def login():
    st.markdown("""
    <div style='text-align: center; margin-top:-60px; margin-bottom: 100px;margin-left: -50px;'>
    <h2 style='font-size: 60px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    AssetFind AI
    </span>
    <span style='font-size: 60%;'>
    <sup style='position: relative; top: 5px; color:white ;'>by Affine</sup> 
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True) 
    col1,col2,col3=st.columns([0.5,1,1])
    with col2.form("login_form"):
        ## Input User Data
        email_id = st.text_input("Email ID",placeholder="xyz.d@affine.ai")
        user_pass=st.text_input("Password", placeholder="admin@123",type="password")
        col4, col5 = st.columns(2)
        login = col4.form_submit_button("Login")
        # register_user = col2.form_submit_button("Register User")
   
    if login:
        st.session_state.login_flag, st.session_state.login_user_type=UserData().login(email_id,user_pass)

        if st.session_state.login_flag:
            st.session_state.login_user=email_id
            col2.success("Login successfully")
        else:
            st.session_state.login_user=None
            col2.error("Failed to login, please try again later")

       
           
        