import streamlit as st
import databricks.sql as dbsql
from src.login_call import UserData
import streamlit as st



def login():
    """Streamlit UI Login Page"""
    
    ## Login Page Header
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

    # Split the page and display the login form centered on the page.
    col1,col2,col3=st.columns([0.5,1,1])

    ## login page
    with col2.form("login_form"):
        ## Input User Data
        email_id = st.text_input("Email ID",placeholder="xyz.d@affine.ai")
        user_pass=st.text_input("Password", placeholder="admin@123",type="password")
        col4, col5 = st.columns(2)
        login = col4.form_submit_button("Login")
        # register_user = col2.form_submit_button("Register User")

    # Check if the user has clicked the login button or triggered the login process
    if login:
        st.session_state.login_flag, st.session_state.login_user_type=UserData().login(email_id,user_pass)

        if st.session_state.login_flag: # If login is successful
            # Store the logged-in user's email in session state
            st.session_state.login_user=email_id
            # Show success message
            col2.success("Login successfully")
        else: # If login fails
            # Reset the login user in session state to None
            st.session_state.login_user=None 
            # Show error message    
            col2.error("Failed to login, please try again later")

       
           
        