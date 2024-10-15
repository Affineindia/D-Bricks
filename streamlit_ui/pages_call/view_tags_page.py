import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
import numpy as np
import databricks.sql as dbsql
import pandas as pd
import ast
import streamlit_shadcn_ui as ui
import io
import databricks.sql as dbsql
import pandas as pd
import base64
from azure.storage.blob import BlobServiceClient
from src.data_load import DatabrickSqlTable,AzureStorage
from collections import OrderedDict

def view_tag():
    st.markdown("""
    <div style='text-align: center; margin-top:-50px; margin-bottom: 20px;margin-left: -50px;'>
    <h2 style='font-size: 60px; font-family: Courier New, monospace;
                    letter-spacing: 2px; text-decoration: none;'>
    <img src="https://acis.affineanalytics.co.in/assets/images/logo_small.png" alt="logo" width="70" height="60">
    <span style='background: linear-gradient(45deg, #ed4965, #c05aaf);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    Intelli-Tags
    </span>
    <span style='font-size: 60%;'>
    <sup style='position: relative; top: 5px; color:white ;'>by Affine</sup> 
    </span>
    </h2>
    </div>
    """, unsafe_allow_html=True) 
    if st.session_state.login_flag:
        ## Fetch the tags data
        st.session_state.tag_data=DatabrickSqlTable().featch_keywords_data(st.session_state.login_user)
        # st.session_state.user_upload_flag_view_tags=False

        ids_list=st.session_state.tag_data['id'].to_list()
        img_url_list=st.session_state.tag_data['file_path'].to_list()
        list_of_img_name=st.session_state.tag_data['image_name'].to_list()
        list_of_uploaded_by=st.session_state.tag_data['created_by'].to_list()
        list_of_uploaded_at=st.session_state.tag_data['upload_time'].to_list()
        list_of_model_type=st.session_state.tag_data['model_type'].to_list()
        output_dict={}
        col1, col2 = st.columns([0.5, 1], gap="large")
        for n,i in enumerate(st.session_state.tag_data['final_predictor'].to_list()):
            list_of_dict=ast.literal_eval(i[i.find("["):i.find("]")+1])
            # for dict1 in list_of_dict:
            #     dict1["Model Type"]=list_of_model_type[n] 
            img_name=list_of_img_name[n]   # id=ids_list[n]
            output_dict[img_name]={"img_url":img_url_list[n],
                            "id":ids_list[n],
                            "image_name":list_of_img_name[n],
                            "uploaded_by":list_of_uploaded_by[n],
                            "uploaded_at":list_of_uploaded_at[n],
                            "model_type":list_of_model_type[n],
                            "tags":list_of_dict}

        ## select id
        selected_id=col1.selectbox("Select the Image ::",list_of_img_name) #  ids_list
        on_change=None
        if selected_id!=on_change:
            st.session_state.output_list_dict_1=output_dict[selected_id]
            st.session_state.selected_url=AzureStorage().read_image(st.session_state.output_list_dict_1['img_url'])
            st.session_state.output_list_dict=st.session_state.output_list_dict_1['tags']
            on_change=selected_id

        with col1:
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
                                    top: 450px;
                                    left: 200px;
                                    transform: translate(-50%, -50%);
                                    background-color: rgba(0, 0, 0, 0.8);
                                    color: white;
                                    padding: 20px;
                                    width: 400px;
                                    height: 235px;
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
                                    height: 250px;
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
                            <img src="data:image/png;base64,{st.session_state.selected_url}" style="height:500px;width:400px;border:black 0.2px solid;">
                            <div class="hover-overlay">
                                <table style="margin-left:-25px">
                                    <tr>
                                        <th>ID</th>
                                        <td>{st.session_state.output_list_dict_1["id"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Image Name</th>
                                        <td>{st.session_state.output_list_dict_1["image_name"]}</td> 
                                    </tr>
                                    <tr>
                                        <th>Uploaded By</th>
                                        <td>{st.session_state.output_list_dict_1["uploaded_by"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Uploaded At</th>
                                        <td>{st.session_state.output_list_dict_1["uploaded_at"]}</td>
                                    </tr>
                                    <tr>
                                        <th>Model Type</th>
                                        <td>{st.session_state.output_list_dict_1["model_type"]}</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        def merge_dicts_in_order(dict1, dict2):
            # Start with dict1 to preserve its order
            merged_dict = OrderedDict(dict1)
            for key, value2 in dict2.items():
                value1 = dict1.get(key, "")
                if key in merged_dict:
                    # If both values are non-empty and not the same, concatenate them
                    if value1 and value2 and value1 != value2:
                        merged_dict[key] = f"{value1}, {value2}"
                    else:
                        # Otherwise, keep the non-empty value or an empty string if both are empty
                        merged_dict[key] = value1 or value2
                else:
                    # Add keys from dict2 that are not in dict1
                    merged_dict[key] = value2
            return merged_dict

        st.session_state.product_dict = {}
        st.session_state.Product_listed = []
        for dict1 in st.session_state.output_list_dict:
                for keys in dict1.keys():
                    if keys=="products category": # topwear 
                        if dict1[keys] not in st.session_state.Product_listed:
                            st.session_state.product_dict[dict1[keys]] = dict1
                            st.session_state.Product_listed.append(dict1[keys])
                        else:
                            final_dict=merge_dicts_in_order(st.session_state.product_dict[dict1[keys]],dict1)
                            st.session_state.product_dict[dict1[keys]]=final_dict
        # st.markdown(st.session_state.product_dict)
        if len(st.session_state.Product_listed) != 0:
            print("------------------------------Option tabs------------------------")
            print(st.session_state.Product_listed)
            print(st.session_state.product_dict)
            print("-----------------------------------------------------------------")
            with col2:
                select_tabs = ui.tabs(options=st.session_state.Product_listed,
                                    default_value=st.session_state.Product_listed[0], key="main_tabs")
                if select_tabs in st.session_state.Product_listed:
                    index = st.session_state.Product_listed.index(select_tabs)
                    if select_tabs == st.session_state.Product_listed[index]:
                        product_dict1 = {"Categories": [], "Attributes": []}
                        for key,value in st.session_state.product_dict[select_tabs].items():
                            if value!='N/A' and key!="products category":
                                product_dict1['Categories'].append(key)
                                product_dict1['Attributes'].append(value)
                            else:
                                print(f"{key} :: {value}")
                        ui.table(data=pd.DataFrame(
                            (product_dict1)))
                else:
                    with col2:
                        st.error("Please uplaod the image again")

    else:
        with st.sidebar:
            st.info("Please Login first")

