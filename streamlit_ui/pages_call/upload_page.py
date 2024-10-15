import streamlit as st
from io import BytesIO
import zipfile
import os
import datetime
from src.databricks_job_run import DatabrickJob
from src.data_load import AzureStorage

def upload_and_run():
    if st.session_state.login_flag:
        # Streamlit app
        st.title("Upload a Files")
        # model="GPT-4o"
        model=st.selectbox("Select Model ::",['gpt-4o','llava'])
        batch_name=st.text_input("Batch Name ::",max_chars=20)
        batch_name=batch_name+"-" +str(datetime.datetime.now())
        uploaded_files = st.file_uploader("Choose a ZIP file", type=["zip","png", "jpg", "jpeg", "gif", "bmp", "tiff","webp","jfif"],accept_multiple_files=True)
        button=st.button("Upload")
        created_at=datetime.datetime.now()
        if button:
            # st.session_state.user_upload_flag_view_tags=True
            if uploaded_files is not None:
                run_flag_list=[]
                for uploaded_file in uploaded_files:
                    try:
                        print("Zip File ::",uploaded_file.name.endswith(".zip"))
                        if uploaded_file.name.endswith(".zip"):
                            # Unzip the file in memory
                            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                                for file_info in zip_ref.infolist():
                                    with zip_ref.open(file_info) as file:
                                        # Read file content
                                        file_content = file.read()
                                        
                                        # Extract the folder name and file name from the ZIP file structure
                                        _, file_name = os.path.split(file_info.filename)
                                        folder_name=batch_name
                                        print("folder_name ::",folder_name)
                                        print("file_name ::",file_name)

                                        if file_name:
                                            # Create a file-like object to upload to blob storage
                                            file_like_object = BytesIO(file_content)
                                            file_metadata={"Image Name":file_name,
                                            "Batch Name":batch_name,
                                            "Uploaded By":st.session_state.login_user,                                        #"sanket.bodake@affine.ai",
                                            "Uploaded At":datetime.datetime.now(),
                                            "TAG GENERATE At":datetime.datetime.now(),
                                            "created_at":created_at,
                                            "TAG PROCESSING TIME":"NA",
                                            "Tag Generated":"No",
                                            "model": model,
                                            "review_status" :False,
                                            "model_failure" : False,
                                            "user_feedback":False}
                                            # Upload to Azure Blob Storage with the folder structure
                                            # run_flag=upload_to_blob(folder_name, file_name, file_like_object.getvalue(),file_metadata)
                                            run_flag=AzureStorage().upload_to_blob(folder_name, file_name, file_like_object.getvalue(),file_metadata)
                                            run_flag_list.append(run_flag)

                        else:
                            folder_name=batch_name #"images_uploads"
                            file_name=uploaded_file.name
                            file_size = uploaded_file.size
                            
                            if file_size < 20 * 1024 * 1024:
                                file_metadata={"Image Name":file_name,
                                                "Batch Name":batch_name,
                                                "Uploaded By":st.session_state.login_user,                                        #"sanket.bodake@affine.ai",
                                                "Uploaded At":datetime.datetime.now(),
                                                "TAG GENERATE At":datetime.datetime.now(),
                                                "created_at":created_at,
                                                "TAG PROCESSING TIME":"NA",
                                                "Tag Generated":"No",
                                                "model":model ,
                                                "review_status" :False,
                                                "model_failure" : False,
                                                "user_feedback":False}

                                # Upload to Azure Blob Storage with the folder structure
                                run_flag=AzureStorage().upload_to_blob(folder_name, file_name, uploaded_file,file_metadata)
                                run_flag_list.append(run_flag)

                            else:
                                st.warning("The uploaded image size is greater than 20 MB.")
                                run_flag_list.append(False)

                    except Exception as e:
                        st.error(e)

                false_count=run_flag_list.count(False)
                if len(run_flag_list)!=false_count:
                    ## Keyword extraction
                    job_id=st.secrets.credentials.Keyword_extraction_job_id # 1062717292802339

                    tag_generated_at=str(datetime.datetime.now())
                    print()
                    print(" Time ::",tag_generated_at)
                    data = {
                        "job_id": job_id,
                        "notebook_params": {
                                            "uploadedby": st.session_state.login_user,
                                            "generatedat": tag_generated_at,
                                            }
                            }       
                    run_id,_=DatabrickJob().job_runs(job_id,data)
                    # output_data=get_job_result(run_id)
                    # if output_data:
                    #     ## EMBEDDING
                    #     job_id=896975992217485
                    #     run_id,_=job_runs_upload(job_id,st.session_state.login_user)
                else:
                    print("No need to run")
    else:
        with st.sidebar:
            st.info("Please Login first")


