## Import all the dependencies
import streamlit as st
import pandas as pd
from src.databricks_job_run import DatabrickJob
from src.data_load import AzureStorage,DatabrickSqlTable
import ast



def intellisearch():
    """`Streamlit UI IntelliSearch Page` - This page includes both `Semantic Search (AI Search)` and `Tag Search` functionalities."""
    ## Page Header
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
    ## split the display into two columns 
    col1,col2=st.columns([1,1])

    ## if user login flag is `True` then will display the page
    if st.session_state.login_flag:

        ## select appropriate search options
        with st.sidebar:
            st.divider()
            search_option=st.radio("",["AI Search","TagSearch"],horizontal=True)
        
        ## if user select the `AI Search` option
        if search_option=="AI Search":
            ## input user query
            text_input=col1.text_input("Text search ::")
            ## Input user query image
            uploaded_file=col2.file_uploader("Image Search ::",type=["png", "jpg", "jpeg", "gif", "bmp", "tiff","webp","jfif"])
            ## Click on the `Search` button to excute the search
            button=col1.button("Search")
            if button:
                ## Store the image in blob storage if the user uploads an image for query
                if uploaded_file is not None:
                    file_name=uploaded_file.name
                    blob_name=AzureStorage().search_upload_blob(file_name,uploaded_file)
                    ## Approach -1 :: image Search and image + text search
                    try:
                        if not text_input:
                            text_input=""
                        ## Load the intellisearch job id from streamlit secrets
                        job_id=st.secrets.credentials.intellisearch_job_id

                        ## input notebbok parameter to run the job 
                        data = {"job_id": job_id,
                                "notebook_params": {"img_input": blob_name,
                                                "text_input": text_input,
                                                "uploaded_by":st.session_state.login_user
                                                }
                                }
                        ## Pass the parameters and return the run ID of the job    
                        run_id,_=DatabrickJob().job_runs(job_id,
                                                data)
                        print("Run the job successfully..")
                        ## Retrieve the results of a job by its run ID using the Databricks API.
                        output=DatabrickJob().get_job_result(run_id)
                        # Evaluate the result of the notebook output to extract the "ids" from the output.
                        # If the result is "No list found in the text.", initialize filter_ids as an empty list.
                        filter_ids=eval(output["notebook_output"]["result"])["ids"]
                        if filter_ids=="No list found in the text.":
                            filter_ids=[]
                    except:
                        filter_ids=[]
                else:
                    ## Approach 2 :: Text Search
                    try:
                        # For text-only search, initialize the `image_base64` variable as an empty string.
                        img_base64=""
                        ## Load the intellisearch job id from streamlit secrets
                        job_id=st.secrets.credentials.intellisearch_job_id

                        ## input notebbok parameter to run the job 
                        data = {
                            "job_id": job_id,
                            "notebook_params": {
                                                "img_input": img_base64,
                                                "text_input": text_input,
                                                "uploaded_by":st.session_state.login_user
                                                }
                                }
                        ## Pass the parameters and return the run ID of the job 
                        run_id,_=DatabrickJob().job_runs(job_id,data)
                        print("Run the job successfully..")
                        ## fetch the output results using the run id
                        output=DatabrickJob().get_job_result(run_id)
                        # Evaluate the result of the notebook output to extract the "ids" from the output.
                        # If the result is "No list found in the text.", initialize filter_ids as an empty list.
                        filter_ids=eval(output["notebook_output"]["result"])["ids"]
                        if filter_ids=="No list found in the text.":
                            filter_ids=[]
                    except:
                        filter_ids=[]
                print(filter_ids)
            else:
                filter_ids=[]    
            
            # Filter the Table by using the unique ids of assets
            st.session_state.file_data=DatabrickSqlTable().filter_data(st.session_state.login_user,filter_ids)
            st.session_state.file_data_flag=True

            if st.session_state.file_data_flag:
                # Extract various columns from the session state file data and print the lengths of the lists, then create 4 layout columns in Streamlit.
                list_of_url=st.session_state.file_data['file_path'].to_list()
                list_of_ids=st.session_state.file_data['id'].to_list()
                list_of_img_name=st.session_state.file_data['image_name'].to_list()
                list_of_uploaded_by=st.session_state.file_data['created_by'].to_list()
                list_of_uploaded_at=st.session_state.file_data['upload_time'].to_list()
                list_of_tag_generated=st.session_state.file_data['tag_generated'].to_list()
                print(" length list_of_ids ::",len(list_of_ids))
                print("List of URL:",len(list_of_url))
                columns = st.columns(4)
                
                ## Read the images and store them in dictionary
                for img_path in list_of_url:
                    if img_path not in st.session_state.images_url.keys():
                        img_str_1=AzureStorage().read_image(img_path)
                        st.session_state.images_url[img_path]=img_str_1

                ## Get the images sequienclly from the session dictionary
                img_str=[st.session_state.images_url[url] for url in list_of_url]
                print("img_str ::",len(img_str))
                
                ## at least 1 image data should present in the `img_str` dict
                if len(img_str)!=0: 
                    image_counter = 0

                    ## Assets images should visualize on ui
                    for n,img_str in enumerate(img_str): 
                        with columns[image_counter % 4]:
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
        else:
            ## Keyword Based Search
            file_data=DatabrickSqlTable().featch_keywords_data(st.session_state.login_user)

            ## List of tags dictionaries
            all_tags_list=[]
            for n,data in enumerate(file_data[['final_predictor','id']].iterrows()):
                ## List of tags dictionaries
                final_tags_list=data[1]['final_predictor']
                ## unique id's of assets
                id=data[-1]['id']
                list_of_dict=ast.literal_eval(final_tags_list[final_tags_list.find("["):final_tags_list.find("]")+1])
                ## adding the unique id of assets to all generated tag list
                for dict1 in list_of_dict:
                    dict1['id']=id
                all_tags_list.extend(list_of_dict)
            
            ## Dataframe of all products generated tags
            df=pd.DataFrame(all_tags_list)

            with st.sidebar:
                ## Finding unique values in each column and creating a dictionary
                unique_values_dict = {col: df[col].unique().tolist() for col in df.columns}
                ## Apply the filter on the Product Category column
                filter_products_types=st.multiselect("Product Category",unique_values_dict['products category'])

                if filter_products_types:
                    df=df.loc[df['products category'].isin(filter_products_types)]

                ## Finding unique values in each column and creating a dictionary
                unique_values_dict = {col: df[col].unique().tolist() for col in df.columns}
                ## Apply the filter on the Products column
                filter_products=st.multiselect("Products",unique_values_dict['Products'])

                if filter_products:
                    df=df.loc[df['Products'].isin(filter_products)]
                
                ## Finding unique values in each column and creating a dictionary
                unique_values_dict = {col: df[col].unique().tolist() for col in df.columns}
                ## Apply the filter on the gender column
                filter_gender=st.multiselect("Gender",unique_values_dict['Gender'])

                if filter_gender:
                    df=df.loc[df['Gender'].isin(filter_gender)]

            ids=df['id'].unique()
            df2=file_data.loc[file_data['id'].isin(ids)]
            
            list_of_url=df2['file_path'].to_list()
            list_of_ids=df2['id'].to_list()
            list_of_img_name=df2['image_name'].to_list()
            list_of_uploaded_by=df2['created_by'].to_list()
            list_of_uploaded_at=df2['upload_time'].to_list()
            list_of_tag_generated=df2['tag_generated'].to_list()
            print(" length list_of_ids ::",len(list_of_ids))
            print("List of URL:",len(list_of_url))
            columns = st.columns(4)
            
            ## Read the images and store them in dictionary
            for img_path in list_of_url:
                if img_path not in st.session_state.images_url.keys():
                    img_str_1=AzureStorage().read_image(img_path)
                    st.session_state.images_url[img_path]=img_str_1

            img_str=[st.session_state.images_url[url] for url in list_of_url]
            print("img_str ::",len(img_str))

            if len(img_str)!=0:
                image_counter = 0
                for n,img_str in enumerate(img_str): 
                    with columns[image_counter % 4]:
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
    else:
        ## If user has not logged in, then below message will be displayed.
        with st.sidebar:
            st.info("Please Login first")

