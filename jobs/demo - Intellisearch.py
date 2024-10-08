# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch==0.22 
# MAGIC %pip install timm
# MAGIC %pip install einops
# MAGIC #azure custom vision embedding
# MAGIC %pip install msrest
# MAGIC %pip install azure-cognitiveservices-vision-computervision==0.9.0
# MAGIC # %restart_python
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# import necessary library
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
# from transformers import AutoModel, AutoProcessor
# Azure custom vision
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings,generate_blob_sas,BlobSasPermissions
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from datetime import datetime, timedelta
from PIL import Image
import logging
import ast
import torch
import base64
import requests
import io
import time
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

storage_account_name = "adlsusdldev02"
storage_account_access_key =dbutils.secrets.get(scope="dbx-us-scope",key="storage-account-access-key") 
container_name = "intellitag"
mount_point = "/mnt/my_intellitag_mount"

VECTOR_SEARCH_ENDPOINT_NAME="intellisearch"
text_embedding_index_name="intellitag_catalog.intellitag_dbx.intellisearch_dbx_text_embedding_index"
image_embedding_index_name ="intellitag_catalog.intellitag_dbx.intellisearch_dbx_image_embedding_index"

# COMMAND ----------

# Azure Cognitive Services OCR API information
ai_vision_subscription_key =dbutils.secrets.get(scope="dbx-us-scope",key="ai-vision-subscription-key") 
ai_vision_endpoint =dbutils.secrets.get(scope="dbx-us-scope",key="ai-vision-endpoint")
api_version = "2023-02-01-preview"

# COMMAND ----------

if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source=f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
        mount_point=mount_point,
        extra_configs={f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
    )

# COMMAND ----------

# # Initialize the model and processor for embedding
# model = AutoModel.from_pretrained('jinaai/jina-clip-v1', 
#                                   cache_dir="/Volumes/intellitag_catalog/models/llms/test/jina-clip-v1",
#                                   trust_remote_code=True)
                                  
# processor = AutoProcessor.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)

# COMMAND ----------

# #Jinaai embedding model
# def jinaai_generate_embeddings(query):
#     try:
#         # Check if the input is a string (text) or an image (e.g., NumPy array or PIL image)
#         if isinstance(query, str):
#             print("Process text input and generate embeddings")
#             # Process text input and generate embeddings
#             inputs = processor(text=query, return_tensors="pt", padding=True)
#             with torch.no_grad():
#                 text_embedding = model.get_text_features(inputs['input_ids'])
#             # Convert text embedding to a list of floats
#             return text_embedding.squeeze().cpu().numpy().astype(float).tolist()
        
#         else:
#             print("Process image input and generate embeddings")
#             # Process image input and generate embeddings
#             inputs = processor(images=query, return_tensors="pt")
#             with torch.no_grad():
#                 image_embedding = model.get_image_features(inputs['pixel_values'])
#             # Convert image embedding to a list of floats
#             return image_embedding.squeeze().cpu().numpy().astype(float).tolist()
    
#     except Exception as e:
#         print(f"Error generating embeddings: {e}")
#         return None


# COMMAND ----------

#azure custom vision embedding
# Initialize Computer Vision client
computervision_client = ComputerVisionClient(
    ai_vision_endpoint, CognitiveServicesCredentials(
        ai_vision_subscription_key)
)
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
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_access_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)

    # Get a reference to the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)

    # # Generate a SAS token for the blob
    # sas_token = blob_client.generate_shared_access_signature(permission="r", expiry=datetime.utcnow() + timedelta(hours=1))

    # Generate a SAS token for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
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

def get_context(question,flag,VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname,uploaded_by,thereshold=0.3):
    if flag=="input_is_text":
        # embeddings=generate_embeddings(question)
        embeddings=text_embedding(question)
        print("text_embedding")
    elif flag=="input_is_image":
        embeddings=get_image_embeddings(question)
        print("image_embedding")
    else:
        print("please check input provided")
    vsc = VectorSearchClient()
    index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
    results=index.similarity_search(
        # query_text=question,
        query_vector=embeddings,
        columns=["id","image_path", "final_predictor"],
        filters={"created_by": (uploaded_by)},
        num_results=10)
    docs = results.get('result', {}).get('data_array', [])
    
    return docs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text to Image

# COMMAND ----------

input_img = dbutils.widgets.get("img_input")
text_input=dbutils.widgets.get("text_input")
print("###############################")
print(text_input)
uploaded=dbutils.widgets.get("uploaded_by")
if (text_input!="" and input_img!="") or input_img!="":
    flag="input_is_image"
    print("input is image")
    context=get_context(question=input_img,flag=flag,
                        VECTOR_SEARCH_ENDPOINT_NAME=VECTOR_SEARCH_ENDPOINT_NAME,vs_index_fullname=image_embedding_index_name,
                        uploaded_by=uploaded)
    ids=[i[0] for i in context]
    image_list =[entry[1] for entry in context]
elif text_input!="":
    flag="input_is_text"
    print("input is text")
    print("Text Search ::",text_input)
    print("Uploaded By ::",uploaded)
    context=get_context(
        question=text_input,flag=flag,
        VECTOR_SEARCH_ENDPOINT_NAME=VECTOR_SEARCH_ENDPOINT_NAME,
        vs_index_fullname=image_embedding_index_name,
        uploaded_by=uploaded)
    ids=[i[0] for i in context]
    image_list =[entry[1] for entry in context]

else:
    pass


results={"ids":ids}

print(results)

# COMMAND ----------

for img in image_list:
    img_path = "/dbfs/mnt/my_intellitag_mount/" + img
    image = Image.open(img_path)
    display(image)

# COMMAND ----------

id_list=[ context[i][0]for i in range (len(context))]
generated_tags_list=[ context[i][2] for i in range (len(context))]
# image_file_path_list=[ context[i][1] for i in range (len(context))]

# COMMAND ----------

# # You can now use `csv_content` wherever needed
import pandas as pd
from io import StringIO
# Create a DataFrame from the two lists
context_df = pd.DataFrame({
    'id': id_list,
    'generated_tags': generated_tags_list
})

# Convert the DataFrame to CSV string
csv_buffer = StringIO()
context_df.to_csv(csv_buffer, index=False)

# Store the CSV content in a variable
csv_content = csv_buffer.getvalue()

# Print the CSV content (Optional)
# print(csv_content)

# COMMAND ----------

from openai import AzureOpenAI
##########################################################################################################
openai_api_version = "2024-06-01"
openai_api_key =dbutils.secrets.get(scope="dbx-us-scope",key="azure-openai-llm-api-key") 
openai_azure_endpoint=dbutils.secrets.get(scope="dbx-us-scope",key="azure-openai-llm-base-url") 
model_deployment_name="gpt4odeployment" #"gpt-4o-05-13"
##########################################################################################################
client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    azure_endpoint=openai_azure_endpoint
)

# Function to get most relevant images using GPT4V model from filtered images
def get_input_image_tag(blob_name,text_input=""):
    cogSvcsEndpoint = ai_vision_endpoint
    cogSvcsApiKey = ai_vision_subscription_key

    # # Connect to Azure Storage
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_access_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)

    # Get a reference to the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=blob_name)

    # # Generate a SAS token for the blob
    # sas_token = blob_client.generate_shared_access_signature(permission="r", expiry=datetime.utcnow() + timedelta(hours=1))

    # Generate a SAS token for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1),
    )
    # Construct the URL with the SAS token
    blob_url = f"{blob_client.url}?{sas_token}"
    tags="""Focus on the clothing item(s) being highlighted which could be either topwear, bottomwear, or both. Use the following attributes to describe topwear: Products, Color, Gender, Pattern, Silhouette, Neckline, Sleeve Length, Sleeve Style, Fabric, Brand, Occasion, Fit Type, and Top Wear Length. For bottomwear, describe using these attributes: Products, Color, Gender, Pattern, Bottom Style, Bottom Length, Waist Rise, Fabric, Brand, Occasion, and Fit Type."""
    prompt=f"""Describe the product in the given images. Use the following tags:{tags}"""
    
    response = client.chat.completions.create(
        model=model_deployment_name, #"gpt-4v",
        messages=[{ "role": "system", "content": "You are a helpful assistant."},
                  { "role": "user", "content": [{"type": "text","text": prompt},
                  {"type": "image_url","image_url": {"url":blob_url}}] } ], 
        # max_tokens=1024,
        max_tokens=4096,
        n=1,
        temperature=0,
    )
    
    answer = response.choices[0].message.content
    # print(f"Answer: {answer}\n")
    return answer


# COMMAND ----------

if input_img!="" and text_input!="":
   print("if")
   input_img_description=get_input_image_tag(input_img,text_input)
   concatenated_text_input="Adhere strictly to the following user-specific requirements to generate recommendation. Recommend only products that fully satisfy all the user-specific requirements without exception. \n  user-specific requirements:"+text_input+".\nThe user input image description:\n "+input_img_description

elif input_img!="":
    print("elif1")
    input_img_description=get_input_image_tag(input_img)
    concatenated_text_input=input_img_description
elif text_input!="":
    print("elif2")
    concatenated_text_input=text_input
else:
    pass


# COMMAND ----------

# Function to get most relevant images using GPT4V model from filtered images
def llm_filter(query,input_csv):

    template = """
    You will be working with a CSV file containing columns labeled 'id'and 'generated_tags'. Your task is to utilize this data to generate personalized product recommendations based on specific user input.User input might contains text describing user-specific requirements for the product they are looking for and/or description of products, use these for recommending the products. Follow these detailed instructions to ensure your recommendations are accurate and relevant:
    
    1.Utilize the detailed data in the 'id' and 'generated_tags' columns to recommend the similar product. Focus mainly on 'products category','Products and 'Gender'.

    2.Prioritize relevance and suitability of products based on the user's specific needs, rather than relying solely on keyword matching from the query.

    3.Recommend products even if the image does not feature a person If gender is not specified.

    4.Sort the recommendations in descending order of relevance and suitability.
    
    5.** Give recommendations in list format only, strictly follow the format given in the below example. Remember that each entry in the list should be enclosed in "". Do not add duplicate product id in the recommendation list. **
    Output must be in the following format as stated in the example below-
    Recommendations:["id1", "id2",........]

    csv data : {input_csv}

    user input : {question}

    Response:
    """
    
    prompt = template.format(input_csv=input_csv, question=query)
    response = client.chat.completions.create(
        model=model_deployment_name, #"gpt-4v",
        messages=[ {"role": "system", "content": "You are a recommendation assistant."},
        {"role": "user", "content": f"{prompt}"}],
        # max_tokens=1024,
        max_tokens=4096,
        n=1,
        temperature=0,
    )
    
    answer = response.choices[0].message.content
    # print(f"Question: {query}\n")
    # print(f"Answer: {answer}\n")
    # Find the positions of the opening [ and closing ]
    start = answer.find('[')
    end = answer.find(']')

    # Extract the substring between the brackets
    if start != -1 and end != -1:
        list_content = answer[start:end+1]  # Include the closing ]
        
        # Convert the extracted string into a list using ast.literal_eval
        extracted_id_list = ast.literal_eval(list_content)
        return {"ids":extracted_id_list}
  
    else:
        print("No list found in the text.")

        return {"ids":answer}


# COMMAND ----------

retrived_images_id=llm_filter(concatenated_text_input,csv_content)
retrived_images_id

# COMMAND ----------

dbutils.notebook.exit(retrived_images_id)
