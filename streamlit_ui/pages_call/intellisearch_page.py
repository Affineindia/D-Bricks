import streamlit as st
from azure.storage.blob import BlobServiceClient
from PIL import Image
import io
import databricks.sql as dbsql
import pandas as pd
import base64
import requests
import json
import datetime
import time
from src.databricks_job_run import DatabrickJob
from src.data_load import AzureStorage,DatabrickSqlTable


def intellisearch():
    st.markdown("""
    <div style='text-align: center; margin-top:-50px; margin-bottom: 20px;margin-left: -50px;'>
    <h2 style='font-size: 60px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    Intelli-Search
    </span>
    <span style='font-size: 60%;'>
    <sup style='position: relative; top: 5px; color:white ;'>by Affine</sup> 
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True) 
    col1,col2,col3=st.columns([1,1,1])
    if st.session_state.login_flag:
        # with st.sidebar:
        text_input=col1.text_input("Text search ::")
        uploaded_file=col2.file_uploader("Image Search ::",type=["png", "jpg", "jpeg", "gif", "bmp", "tiff"])
        button=col1.button("Search")
        if button:
            if uploaded_file is not None:
                # image = Image.open(uploaded_file)
                # st.image(image, caption="Uploaded Image", use_column_width=True)
                file_name=uploaded_file.name
                blob_name=AzureStorage().search_upload_blob(file_name,uploaded_file)
                ## image Search
                try:
                    if not text_input:
                        text_input=""
                    # job_id=367189284162575
                    job_id=st.secrets.credentials.intellisearch_job_id
                    data = {"job_id": job_id,
                            "notebook_params": {"img_input": blob_name,
                                            "text_input": text_input,
                                            "uploaded_by":st.session_state.login_user
                                            }
                            }
                    run_id,_=DatabrickJob().job_runs(job_id,
                                            data)
                    print("Run ID::",run_id)
                    output=DatabrickJob().get_job_result(run_id)
                    filter_ids=eval(output["notebook_output"]["result"])["ids"]
                    if filter_ids=="No list found in the text.":
                        filter_ids=[]
                except:
                    filter_ids=[]
            else:
                ## Text Search
                try:
                    img_base64=""
                    job_id=st.secrets.credentials.intellisearch_job_id
                    data = {
                        "job_id": job_id,
                        "notebook_params": {
                                            "img_input": img_base64,
                                            "text_input": text_input,
                                            "uploaded_by":st.session_state.login_user
                                            }
                            }
                    run_id,_=DatabrickJob().job_runs(job_id,data)
                    print("Run ID::",run_id)
                    output=DatabrickJob().get_job_result(run_id)
                    filter_ids=eval(output["notebook_output"]["result"])["ids"]
                    if filter_ids=="No list found in the text.":
                        filter_ids=[]
                except:
                    filter_ids=[]
            
            print(filter_ids)
            # st.text(filter_ids)
        else:
            filter_ids=[]    #['c314cc8975b911efb3eec025a5494578', 'b37d5e3871a711ef801c14857ffeb9ff', '76eaa2f471be11ef9b2f14857ffeb9ff'] #[]

        # if len(st.session_state.ids_1)==0:#st.session_state.file_data_flag==False:
            ### metadata table
        st.session_state.file_data=DatabrickSqlTable().filter_data(st.session_state.login_user,filter_ids)
        # st.session_state.ids_1=st.session_state.file_data['id'].to_list()
        st.session_state.file_data_flag=True
            # st.dataframe(st.session_state.file_data)
        # elif st.session_state.ids_1!=:
        #     pass
            # st.dataframe(st.session_state.file_data)

        if st.session_state.file_data_flag:
            list_of_images=st.session_state.file_data['file_path'].to_list()
            list_of_ids=st.session_state.file_data['id'].to_list()
            list_of_img_name=st.session_state.file_data['image_name'].to_list()
            list_of_uploaded_by=st.session_state.file_data['created_by'].to_list()
            list_of_uploaded_at=st.session_state.file_data['upload_time'].to_list()
            list_of_tag_generated=st.session_state.file_data['tag_generated'].to_list()
            print(" length list_of_ids ::",len(list_of_ids))
            ## len of list of images
            # len_of_img_url_state=len(st.session_state.img_str) # 2
            # len_of_img_url=len(list_of_images) # 3
            list_of_url=[url for url in list_of_images ] # if url not in st.session_state.image_paths
            print("List of URL:",len(list_of_url))
            columns = st.columns(4)
            st.session_state.selected_images=[]
            # if len_of_img_url_state!=len_of_img_url:
            img_str=[]
            for img_path in list_of_url:
                img_str_1=AzureStorage().read_image(img_path)
                st.session_state.image_paths.append(img_path)
                img_str.append(img_str_1)
                # st.session_state.img_str.append(img_str_1)
            print("img_str ::",len(img_str))
            if len(img_str)!=0: #st.session_state.img_str
                image_counter = 0
                for n,img_str in enumerate(img_str): #st.session_state.img_str
                    with columns[image_counter % 4]:
                        # if st.checkbox("", key=list_of_ids[n]):
                        #     st.session_state.selected_images.append(list_of_ids[n])
                        # st.image(img, caption=img_path)
                        # img_html = f'''
                        # <div style="text-align:flex;padding-top:10px">
                        #     <img src="data:image/png;base64,{img_str}" style="height:200px;width:200px;border:black 2px solid;">
                        # </div>
                        # '''
                        # st.markdown(img_html, unsafe_allow_html=True)
                        st.markdown("""
                                <style>
                                .hover-container {
                                    position: relative;
                                    display: inline-block;
                                    margin: 10px;
                                }
                                .hover-container .hover-overlay {
                                    display: none;
                                    position: absolute;
                                    top: 250px;
                                    left: 300px;
                                    transform: translate(-50%, -50%);
                                    background-color: rgba(0, 0, 0, 0.8);
                                    color: white;
                                    padding: 20px;
                                    width: 400px;
                                    height: 250px;
                                    box-sizing: border-box;
                                    border-radius: 5px;
                                    z-index: 1000;
                                    overflow: auto;
                                }
                                .hover-container:hover .hover-overlay {
                                    display: block;
                                }
                                .hover-container img {
                                    display: block;
                                    width: 200px;
                                    height: 200px;
                                    border-radius: 12px;
                                }
                                .hover-container .hover-overlay table {
                                width: 100%;
                                border-collapse: collapse;
                                font-size: 15px; /* Small font size */
                                
                            }
                                </style>
                            """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="hover-container">
                            <figure style="text-align: center;">
                            <img src="data:image/png;base64,{img_str}" style="height:200px;width:200px;border:black 0.2px solid;">
                            </figure>
                            <div class="hover-overlay">
                                <table style="margin-left:-25px">
                                    <tr>
                                        <th>ID</th>
                                        <td>{list_of_ids[n]}</td>
                                    </tr>
                                    <tr>
                                        <th>Image Name</th>
                                        <td>{list_of_img_name[n]}</td>
                                    </tr>
                                    <tr>
                                        <th>Uploaded By</th>
                                        <td>{list_of_uploaded_by[n]}</td>
                                    </tr>
                                    <tr>
                                        <th>Uploaded At</th>
                                        <td>{list_of_uploaded_at[n]}</td>
                                    </tr>
                                    <tr>
                                        <th>Tag Generated</th>
                                        <td>{list_of_tag_generated[n]}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    image_counter += 1
                # st.text(st.session_state.selected_images)
                # view_flag=True #st.button("View",)
                # if view_flag:
                #     view_tag()
    else:
        with st.sidebar:
            st.info("Please Login first")

