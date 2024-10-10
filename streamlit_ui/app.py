from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
from pages_call.upload_page import upload_and_run
from pages_call.user_add_page import user_add
from pages_call.login_page import login
from pages_call.intellisearch_page import intellisearch
from pages_call.view_tags_page import view_tag



st.set_page_config(layout="wide")

def clear_session(key):
    print(key)
    st.session_state.clear()


if "login_flag" not in st.session_state:
    st.session_state.login_flag=False

if "login_user" not in st.session_state:
    st.session_state.login_user=None

if "login_user_type" not in st.session_state:
    st.session_state.login_user_type=None


### page - intellitag
if "file_data" not in st.session_state:
    st.session_state.file_data=None

if "file_data_flag" not in st.session_state:
    st.session_state.file_data_flag=False

if "img_str" not in st.session_state:
    st.session_state.img_str=[]

if "image_paths" not in st.session_state:
    st.session_state.image_paths=[]

## view tags
if "tag_data" not in st.session_state:
    st.session_state.tag_data=pd.DataFrame()

if "img_url_tag" not in st.session_state:
    st.session_state.img_url_tag=[]



# Use Streamlit to create a sidebar with multiple flow options
with st.sidebar:
    # Define an option menu with flow choices
    choose = option_menu("", ["Login","Add User","Upload","Image Gallery", "View Tags"],
                            icons=["house",'list-task', 'cloud-upload', 'bar-chart-steps',
                                'eye', 'database-down'],
                            menu_icon="app-indicator", default_index=0,orientation="vertical",
                            styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#fa2a54"},
    },
    # key="1",
    # on_change=clear_session,
    )
    if st.session_state.login_user!=None:
        st.text("User ::"+st.session_state.login_user)


# Depending on the selected flow, call the respective function
if choose == "Upload":
    upload_and_run()

elif choose == "Login":
    login()

elif choose == "Add User":
    user_add()

elif choose == "Image Gallery":
    intellisearch()

elif choose == "View Tags":
    view_tag()

else:
    st.write('yet not written')