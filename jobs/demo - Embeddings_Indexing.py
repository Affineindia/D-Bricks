# Databricks notebook source
# MAGIC %md
# MAGIC ##### Import All Required Packages
# MAGIC ---------------------------

# COMMAND ----------

# import necessary library
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from msrest.authentication import CognitiveServicesCredentials
from transformers import AutoModel, AutoProcessor
#####################################################
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
from pyspark.sql import SparkSession
###########################################
from datetime import datetime, timedelta
from PIL import Image
import pandas as pd
import numpy as np
import requests
import time
import logging
import re
import torch

##############################################################
from databricks.vector_search.client import VectorSearchClient
from transformers import AutoModel, AutoProcessor
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time 
import torch

# COMMAND ----------

# Azure Cognitive Services OCR API information
ai_vision_subscription_key = dbutils.secrets.get(scope="dbx-us-scope",key="ai-vision-subscription-key") 
ai_vision_endpoint =dbutils.secrets.get(scope="dbx-us-scope",key="ai-vision-endpoint") 
api_version = "2023-02-01-preview"

# Azure Storage account details
storage_account_key =dbutils.secrets.get(scope="dbx-us-scope",key="storage-account-access-key") 
storage_account_name ="adlsusdldev02" 
storage_container_name ="intellitag"

# COMMAND ----------

file_metadata="File_Metadata"
table_name = "Result_Logs"
results_embedding_table = "intellitag_catalog.intellitag_dbx.result_table_embedded_azure"

# COMMAND ----------

## inputs using databricks api
uploaded_by= dbutils.widgets.get("uploadedby")
# uploaded_by="sanket.bodake@affine.ai"

# filter the data
query=f"""SELECT t1.id,t1.image_name,t1.file_path,t1.created_by,t1.upload_time,t2.final_predictor
        FROM (
        SELECT *
        FROM intellitag_catalog.intellitag_dbx.{file_metadata}
        WHERE created_by = '{uploaded_by}' and tag_generated = "Yes"
        ) t1   
        JOIN intellitag_catalog.intellitag_dbx.{table_name} t2
        ON t1.id = t2.id where text_vector_flag = 'No'"""

df = spark.sql(query)
display(df)

# COMMAND ----------

# # Initialize the model and processor for embedding 
# model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
# processor = AutoProcessor.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# COMMAND ----------

# Initialize the Spark session and define the schema
spark = SparkSession.builder.appName("Embedding Processing").getOrCreate()
schema = StructType([
    StructField("id", StringType(), True),
    StructField("created_by", StringType(), True),
    StructField("image_path", StringType(), True),
    StructField("final_predictor", StringType(), True),
    StructField("text_vector_flag", StringType(), True),
    StructField("image_embedding", ArrayType(FloatType()), True),
    StructField("text_embedding", ArrayType(FloatType()), True)
])

# COMMAND ----------

def create_table_if_not_exists(table_name, schema):
    """Create the table if it does not exist."""
    if not spark.catalog.tableExists(table_name):
        # Create an empty DataFrame using the schema
        empty_df = spark.createDataFrame([], schema)
        # Create the table
        empty_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        print(f"Table {table_name} created successfully.")
    else:
        print(f"Table {table_name} already exists.")

create_table_if_not_exists(results_embedding_table, schema)

# COMMAND ----------

def extract_and_format_generated_tags(text):
    """Function to convert final_predictor to text format."""
    match = re.search(r"\{generated_tags=\[(.*)\]\}", text)
    if match:
        generated_tags_text = match.group(1)
        formatted_tags = generated_tags_text.replace("=", ":")
        formatted_tags = formatted_tags.replace("}, {", "\n")
        formatted_tags = formatted_tags.replace("{", "").replace("}", "")
        return formatted_tags
    else:
        return "No generated_tags found in the provided text."

# COMMAND ----------

# Initialize Computer Vision client
computervision_client = ComputerVisionClient(
    ai_vision_endpoint, CognitiveServicesCredentials(
        ai_vision_subscription_key)
)

# COMMAND ----------

def text_embedding(prompt):
    """
    Text embedding using Azure Computer Vision 4.0
    """
    version = "?api-version=" + api_version + "&modelVersion=latest"
    vec_txt_url = f"{ai_vision_endpoint}/computervision/retrieval:vectorizeText{version}"
    headers = {"Content-type": "application/json",
               "Ocp-Apim-Subscription-Key": ai_vision_subscription_key}

    payload = {"text": prompt}
    response = requests.post(vec_txt_url, json=payload, headers=headers)

    if response.status_code == 200:
        text_emb = response.json().get("vector")
        return text_emb

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# COMMAND ----------

# Function to get image embedding
def get_image_embeddings(blob_name):
    cogSvcsEndpoint = ai_vision_endpoint
    cogSvcsApiKey = ai_vision_subscription_key

    # # Connect to Azure Storage
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)

    # Get a reference to the blob
    blob_client = blob_service_client.get_blob_client(
        container=storage_container_name, blob=blob_name)

    # # Generate a SAS token for the blob
    # sas_token = blob_client.generate_shared_access_signature(permission="r", expiry=datetime.utcnow() + timedelta(hours=1))

    # Generate a SAS token for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=storage_container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1),
    )
    # Construct the URL with the SAS token
    blob_url = f"{blob_client.url}?{sas_token}"
    url = f"{cogSvcsEndpoint}/computervision/retrieval:vectorizeImage"
    params = {
        "api-version": api_version
    }
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": cogSvcsApiKey
    }
    data = {
        "url": blob_url
    }
    # print(f"Calling Computer Vision API for blob: {blob_name}")

    response = requests.post(url, params=params, headers=headers, json=data)

    # results = response.json()

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        logging.error(f"Error: {response.status_code}, {response.text}")
        response.raise_for_status()

    embeddings = response.json()["vector"]
    return embeddings

# COMMAND ----------

def process_and_store_embeddings(df, uploaded_by, schema,embedding_model):
    """Function to iterate through each row in the DataFrame, generate embeddings, and append to the Delta table."""
    for row in df.collect():
        image_path = "/dbfs/mnt/my_intellitag_mount/" + row.file_path
        predictor_text = row.final_predictor
        input_text = extract_and_format_generated_tags(predictor_text)
        input_image = Image.open(image_path)
        if embedding_model == "jinaai":
            text_emb, image_emb = generate_embeddings(input_text, input_image)

        elif embedding_model == "azure":
            text_emb = text_embedding(input_text)
            image_emb = get_image_embeddings(row.file_path)


        if text_emb is not None and image_emb is not None:
            embed_data = {
                'id': row.id,
                'created_by': uploaded_by,
                'image_path': row.file_path,
                'final_predictor': row.final_predictor,
                'text_vector_flag': 'Yes',
                'text_embedding': text_emb,
                'image_embedding': image_emb
            }
        # return embed_data
            data_df = spark.createDataFrame([embed_data], schema=schema)
            data_df.write.format("delta").mode("append").saveAsTable(results_embedding_table)
            
            # Perform the update using SQL
            spark.sql(f"""
                UPDATE intellitag_catalog.intellitag_dbx.{table_name}
                SET text_vector_flag = 'Yes'
                WHERE id = '{row.id}'
            """)


# COMMAND ----------

# display(df)

# COMMAND ----------

process_and_store_embeddings (df, uploaded_by, schema,"azure")

# COMMAND ----------

# MAGIC %md
# MAGIC ### indexing

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vector Search Endpoint connection & creation

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="intellisearch"
text_embedding_index_name = "intellitag_catalog.intellitag_dbx.intellisearch_dbx_text_embedding_index"
image_embedding_index_name = "intellitag_catalog.intellitag_dbx.intellisearch_dbx_image_embedding_index"

# COMMAND ----------

def endpoint_exists( endpoint_name):
    vsc = VectorSearchClient()
    # Check if the endpoint exists
    if len(vsc.list_endpoints())==0:
        return False 
    else: 
        endpoints = vsc.list_endpoints()['endpoints']
        for endpoint in endpoints:
            print(endpoint["name"])
            if endpoint["name"]== endpoint_name:
                return True
    return False

def get_endpoint_status(endpoint_name):
    vsc = VectorSearchClient()
    endpoints = vsc.list_endpoints()['endpoints']

    print(endpoints)
    for endpoint in endpoints:
        if endpoint["name"] == endpoint_name:
            return endpoint["endpoint_status"]["state"]
    raise Exception(f"Endpoint {endpoint_name} not found.")

# COMMAND ----------

def create_endpoint_if_not_exists_and_sync(VECTOR_SEARCH_ENDPOINT_NAME,source_table_fullname ):
    # Initialize Vector Search Client
    vsc = VectorSearchClient()
    if not endpoint_exists(VECTOR_SEARCH_ENDPOINT_NAME):
        print("Endpoint Not exit")
        vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
        time.sleep(70)

        spark.sql(f"ALTER TABLE {source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        endpoint = vsc.get_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME)
        print(f"Endpoint created {VECTOR_SEARCH_ENDPOINT_NAME}")
        print(endpoint)
        # print(endpoint_exists(VECTOR_SEARCH_ENDPOINT_NAME))
        # time.sleep(60)
        # Create Delta Sync Index for Text Embeddings

        vsc.create_delta_sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=text_embedding_index_name,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=1024,  # Ensure this matches your text embedding size
            embedding_vector_column="text_embedding"
        )
        print("Create Delta Sync Index for Text Embeddings")
        # Create Delta Sync Index for Image Embeddings
        vsc.create_delta_sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=image_embedding_index_name,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_dimension=1024,  # Ensure this matches your image embedding size
            embedding_vector_column="image_embedding"
        )
        print("Create Delta Sync Index for Image Embeddings")
           # Assuming new embeddings are added to the source table, trigger the sync process
        while True:
            if vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, text_embedding_index_name).describe()['status']['detailed_state']=="ONLINE_NO_PENDING_UPDATE" and vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, image_embedding_index_name).describe()['status']['detailed_state']=="ONLINE_NO_PENDING_UPDATE" :
            # Wait for index to come online. Expect this command to take several minutes.
                # Sync Text Embedding Index
                vsc.get_index(
                    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
                    index_name=text_embedding_index_name
                ).sync()
                print("Sync Text Embedding Index")
                # Sync Image Embedding Index
                vsc.get_index(
                    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
                    index_name=image_embedding_index_name
                ).sync()
                print("Sync Image Embedding Index")
                break
            else:
                print("Waiting for index to come online")
                time.sleep(10)
            
    else:
        print("Endpoint Already exist.")
        while True:
            if vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, text_embedding_index_name).describe()['status']['detailed_state']=="ONLINE_NO_PENDING_UPDATE" and vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, image_embedding_index_name).describe()['status']['detailed_state']=="ONLINE_NO_PENDING_UPDATE" :
                # Assuming new embeddings are added to the source table, trigger the sync process
                # Sync Text Embedding Index
                vsc.get_index(
                    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
                    index_name=text_embedding_index_name
                ).sync()
                print("Sync Text Embedding Index")
                # Sync Image Embedding Index
                vsc.get_index(
                    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
                    index_name=image_embedding_index_name
                ).sync()
                print("Sync Image Embedding Index")
                break
            else:
                print("Waiting for index to come online")
                time.sleep(10)


create_endpoint_if_not_exists_and_sync(VECTOR_SEARCH_ENDPOINT_NAME,results_embedding_table)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


