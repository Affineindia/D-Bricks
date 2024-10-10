from src.processing import Processing
from azure.storage.blob import BlobServiceClient
import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64
import pandas as pd

class DatabrickSqlTable(Processing):

    def __init__(self):
        super().__init__()

    def check_img_name(self,created_by):
        """Check image already exists in the database"""
        query_img = f"SELECT image_name FROM intellitag_catalog.intellitag_dbx.file_metadata where created_by='{created_by}'"
        with self.sql_connection.cursor() as cursor:
            cursor.execute(query_img)
            results = cursor.fetchall()
        img_name_list=[row.image_name for row in results]
        return img_name_list

    def upload_file_metadata(self,file_metadata):
        query=f"SHOW TABLES IN intellitag_catalog.intellitag_dbx"
        
        # Create the table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS intellitag_catalog.intellitag_dbx.file_metadata (
        id STRING,
        image_name STRING,
        batch_name STRING,
        file_path STRING,
        image_url STRING,
        created_by STRING,
        upload_time TIMESTAMP,
        created_at TIMESTAMP,
        tag_generated STRING,
        tag_generated_at TIMESTAMP,
        tag_processing_time STRING,
        model STRING,
        review_status BOOLEAN,
        model_failure BOOLEAN,
        user_feedback BOOLEAN,
        CONSTRAINT PM_KEY3 PRIMARY KEY (id)
        )
        """
        #CONSTRAINT unique_image_id PRIMARY KEY (image_id)
        with self.sql_connection.cursor() as cursor:
            cursor.execute(create_table_query)

        # unique id
        img_id=self.generate_unique_id()

        img_name_list=self.check_img_name(st.session_state.login_user)
    
        # print(img_name_list)
        if file_metadata['Image Name'] not in img_name_list:
                                                                        
            query = f"""
            INSERT INTO intellitag_catalog.intellitag_dbx.file_metadata (id ,
                                                                        image_name ,
                                                                        batch_name ,
                                                                        file_path ,
                                                                        image_url ,
                                                                        created_by ,
                                                                        upload_time ,
                                                                        created_at ,
                                                                        tag_generated ,
                                                                        tag_generated_at ,
                                                                        tag_processing_time ,
                                                                        model ,
                                                                        review_status ,
                                                                        model_failure ,
                                                                        user_feedback)
                                                                VALUES ('{img_id}',
                                                                        '{file_metadata['Image Name']}',
                                                                        '{file_metadata['Batch Name']}',
                                                                        '{file_metadata['File Path']}',
                                                                        '{file_metadata['image_url']}',
                                                                        '{file_metadata['Uploaded By']}', 
                                                                        '{file_metadata['Uploaded At']}', 
                                                                        '{file_metadata['created_at']}',
                                                                        '{file_metadata['Tag Generated']}',
                                                                        '{file_metadata['TAG GENERATE At']}',
                                                                        '{file_metadata['TAG PROCESSING TIME']}',
                                                                        '{file_metadata['model']}',
                                                                        '{file_metadata['review_status']}',
                                                                        '{file_metadata['model_failure']}',
                                                                        '{file_metadata['user_feedback']}')
                                                                """
            # Execute the query with parameters
            with self.sql_connection.cursor() as cursor:
                cursor.execute(query)

            return True
        else:
            return False

    def featch_keywords_data(self,uploaded_by):
        # query=f"select * from intellitag_catalog.intellitag_dbx.file_metadata where uploaded_by='{uploaded_by}'"
        query=f"""SELECT t1.id,t1.image_name,t1.file_path,t1.created_by,t1.upload_time,t2.final_predictor
            FROM (
            SELECT *
            FROM intellitag_catalog.intellitag_dbx.file_metadata
            WHERE created_by='{uploaded_by}' and tag_generated = "Yes"
            ) t1
            JOIN intellitag_catalog.intellitag_dbx.result_logs t2
            ON t1.id = t2.id"""
        
        with self.sql_connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        # Get the column names from the cursor description
        
        df=pd.DataFrame(result,columns=columns)
        
        return df

    def filter_data(self,uploaded_by,filter_ids):
        query=f"select * from intellitag_catalog.intellitag_dbx.file_metadata where created_by='{uploaded_by}'"
        print(self.sql_connection.cursor())
        with self.sql_connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        print("Featch filter data...",columns)
        df=pd.DataFrame(result,columns=columns)
        # Get the column names from the cursor description
        if len(filter_ids)!=0:
            df=df[df['id'].isin(filter_ids)]
            df['id'] = pd.Categorical(df['id'], categories=filter_ids, ordered=True)
            df = df.sort_values('id').reset_index(drop=True)
            print("SHAPE ::",df.shape)
            return df
        else:
            print("SHAPE ::",df.shape)
            return df

class AzureStorage(DatabrickSqlTable):

    def __init__(self):
        super().__init__()
        __azure_connection_string=os.getenv('AZURE_CONNECTION_STRING') or st.secrets.credentials.AZURE_CONNECTION_STRING
        self.container_name=os.getenv('CONTAINER_NAME') or st.secrets.credentials.CONTAINER_NAME
        self.blob_service_client = BlobServiceClient.from_connection_string(__azure_connection_string)
        print("Container Name:", self.container_name)
        
    def upload_to_blob(self,folder_name, file_name, data,file_metadata):
        "This is the intial upload"
        # Sanitize the folder name and file name
        sanitized_folder_name = self.sanitize_blob_name(folder_name)
        sanitized_file_name = self.sanitize_blob_name(file_name)
        
        # Form the full blob name
        blob_name = f"Intellitag_Uploads/{sanitized_folder_name}/{sanitized_file_name}"
        file_metadata["File Path"]=blob_name
        file_metadata["image_url"]= "/dbfs/mnt/my_intellitag_mount"+blob_name
        save_flag=self.upload_file_metadata(file_metadata)
        if save_flag:
            if blob_name:  # Ensure the blob name is not empty
                blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
                blob_client.upload_blob(data, overwrite=True)
                st.success(f"Uploaded {file_name} to Azure Blob Storage")
                return True
        else:
            st.error(f"Uploaded {file_name} file already exist with same name.")
            return False

    def search_upload_blob(self,file_name, data):
        
        # Sanitize the folder name and file name
        # sanitized_folder_name = sanitize_blob_name(folder_name)
        sanitized_file_name = self.sanitize_blob_name(file_name)
        
        # Form the full blob name
        blob_name = f"intillisearch/{sanitized_file_name}"
        if blob_name:  # Ensure the blob name is not empty
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_name)
            blob_client.upload_blob(data, overwrite=True)
            print(" Search Image path ::", blob_name)
            return blob_name

    def read_image(self,blob_path):

        container_client = self.blob_service_client.get_container_client(self.container_name)
        
        # Download the blob data
        blob_client = container_client.get_blob_client(blob_path)
        blob_data = blob_client.download_blob().readall()
        
        # Convert blob data to Image
        image = Image.open(io.BytesIO(blob_data))
        # fixed_height=200
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        # width_percent = (fixed_height / float(image.size[1]))
        # fixed_width = int((float(image.size[0]) * float(width_percent)))
        # resized_image = image.resize((fixed_width, fixed_height), Image.ANTIALIAS)
        return img_str
