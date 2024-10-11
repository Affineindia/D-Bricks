# Databricks notebook source
# MAGIC %md
# MAGIC The notebook is setting up the environment to deploy custom multimodal LLMs, such as `LLAVA` and `Llama 3.2`, by downloading them from `Hugging Face`. It registers the LLMs in the `Unity Catalog` and tracks their versions using `MLflow`, while serving the models on a large GPU compute instance.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Install Required Packages

# COMMAND ----------

## llava
# %pip install --upgrade transformers==4.39.0 mlflow>=2.11.3 tensorflow==2.16.1 accelerate==0.23.0 bitsandbytes>=0.41.3 
# %pip install azure-storage-blob
# dbutils.library.restartPython()

## llama vision
%pip install --upgrade transformers mlflow>=2.11.3 tensorflow==2.16.1 accelerate>=0.26.0 bitsandbytes>=0.41.3 
%pip install azure-storage-blob
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Hugging Face Auth
# MAGIC Logging into Huggingface for Model Access

# COMMAND ----------

from huggingface_hub import notebook_login

# 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
# Login to Huggingface to get access to the model
notebook_login()

# COMMAND ----------

!nvidia-smi

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import All Required Packages

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
from transformers import AutoProcessor, LlavaForConditionalGeneration
import mlflow
import pandas as pd
from transformers import BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import requests
import base64
from io import BytesIO
import io

import numpy as np
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# COMMAND ----------

# DBTITLE 1,Inferred Signature
signature = infer_signature(
    model_input=np.array(
        [
            ["<input_img>", "<prompts>", "<connection_string>", "<storage_container_name>", "<temperature>", "<max_tokens>"],
             ["<input_img>", "<prompts>", "<connection_string>", "<storage_container_name>", "<temperature>", "<max_tokens>"],
              ]
        ),
    model_output=np.array(
                        [
                            ["<Sample output 1>"], 
                            ["<Sample output 2>"]
                        ]
    ),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Llava v1.6 7B Model Registry

# COMMAND ----------

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        from azure.storage.blob import BlobServiceClient
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import mlflow
        from transformers import BitsAndBytesConfig
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch
        from PIL import Image
        import requests
        import base64
        from io import BytesIO
        import io

        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        # processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16, 
            # quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16),  ## int4
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) ,                                        ## int8
            low_cpu_mem_usage=True,
            device_map="auto",
            # cache_dir="/Volumes/intellitag_catalog/models/llms/test/llava-hf"
            )
        
        # self.model.to("cuda:0")

    def predict(self, context, model_input):
        processor = self.processor
        model = self.model
        results=[]
        #  ["<input_img>", "<prompts>", "<connection_string>", "<storage_container_name>", "<temperature>", "<max_tokens>"],
        for mi in model_input:
            image_input = mi[0]
            prompt = mi[1]
            connection_string = mi[2]
            storage_container_name =mi[3]
            temperature= float(mi[4])
            max_tokens=int(mi[5])
            print("Temperature ::",temperature)
            print("Max Tokens ::",max_tokens)

            def get_image_from_blob(connection_string,storage_container_name,blob_name):

                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                # Get a reference to the container
                container_client = blob_service_client.get_container_client(storage_container_name)
                
                # Get a reference to the blob (image)
                blob_client = container_client.get_blob_client(blob_name)
                
                # Download blob data (image)
                blob_data = blob_client.download_blob().readall()
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(blob_data))
                # display(image)
                return image
            
            # image=base64_to_image(image_input)
            image=get_image_from_blob(connection_string,storage_container_name,image_input)

            inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
            # Generate and store response
            output = model.generate(**inputs, 
                                    max_new_tokens=max_tokens, 
                                    temperature=temperature, 
                                    do_sample=False)
            
            result = (processor.decode(output[0], skip_special_tokens=True))
            results.append(result)
        return results

# COMMAND ----------

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "model", 
        python_model=Model(),
        pip_requirements=['transformers==4.39.0','bitsandbytes>=0.41.3','accelerate==0.23.0','mlflow==2.11.3', 'tensorflow', 'torch', 'Image', 'requests',"bytesbufio","pillow","azure-storage-blob"],
        signature=signature
        )
    run_id = mlflow.active_run().info.run_id

# COMMAND ----------

import mlflow
catalog_name = "intellitag_catalog"
schema_name = "models"
model_name = "v4_affine_llava"

mlflow.set_registry_uri("databricks-uc")

model_version_obj = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=f"{catalog_name}.{schema_name}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### LLama 3.2 11B  Model Registry

# COMMAND ----------

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        from azure.storage.blob import BlobServiceClient
        import mlflow
        from transformers import BitsAndBytesConfig
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        import torch
        from PIL import Image
        import requests
        import base64
        from io import BytesIO
        import io
        
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True) ,
            low_cpu_mem_usage=True,   
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # self.model.to("cuda:0")

    def predict(self, context, model_input):
        processor = self.processor
        model = self.model
        results=[]
        #  ["<input_img>", "<prompts>", "<connection_string>", "<storage_container_name>", "<temperature>", "<max_tokens>"],
        for mi in model_input:
            image_input = mi[0]
            prompt = mi[1]
            connection_string = mi[2]
            storage_container_name =mi[3]
            temperature= float(mi[4])
            max_tokens=int(mi[5])
            print("Temperature ::",temperature)
            print("Max Tokens ::",max_tokens)

            def get_image_from_blob(connection_string,storage_container_name,blob_name):

                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                # Get a reference to the container
                container_client = blob_service_client.get_container_client(storage_container_name)
                
                # Get a reference to the blob (image)
                blob_client = container_client.get_blob_client(blob_name)
                
                # Download blob data (image)
                blob_data = blob_client.download_blob().readall()
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(blob_data))
                # display(image)
                return image
            
            # image=base64_to_image(image_input)
            image=get_image_from_blob(connection_string,storage_container_name,image_input)

            inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
            # Generate and store response
            output = model.generate(**inputs, 
                                    max_new_tokens=max_tokens, 
                                    temperature=temperature, 
                                    do_sample=False)
            
            result = (processor.decode(output[0], skip_special_tokens=True))
            results.append(result)
        return results

# COMMAND ----------

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        "model", 
        python_model=Model(),
        pip_requirements=['transformers','bitsandbytes>=0.41.3','accelerate>=0.26.0','mlflow==2.11.3', 'tensorflow', 'torch', 'Image', 'requests',"bytesbufio","pillow","azure-storage-blob"],
        signature=signature
        )
    run_id = mlflow.active_run().info.run_id

# COMMAND ----------

import mlflow
catalog_name = "intellitag_catalog"
schema_name = "models"
model_name = "v1_affine_llama_vision"

mlflow.set_registry_uri("databricks-uc")


model_version_obj = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=f"{catalog_name}.{schema_name}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC -----------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inferencing 

# COMMAND ----------

tags = """products category:TOPWEAR,BOTTOMWEAR
TOPWEAR--

Products:Shirt,Jumpsuit,Dungarees,Sleepwear,Sweatshirt,Innerwear,Dress,Top,T-shirt,Polo Collar T-shirt,Sweater/Cardigan,Hoodie,Jackets,Raincoat,LongCoat,Blazer,Suits (2-piece or 3-piece Men),Kurta,Kurta Set,Maxi Dress,Showerproof,vest.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colorblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Silhouette:A-line,Peplum,Balloon,Fit and Flare,Sheath,Bodycon,Shirt Style,Jumper,Wrap,Kaftan.
Neckline:Turtle/High Neck,Round Neck,Square Neck,Halter Neck,Scoop Neck,V neck Boat Neck,Polo Collar,Open-Collar,Crew Neck,Tie-Up Neck,Boat Neck,Shirt Neck, pointed collar, mandarin collar.
Sleeve Length:Sleeveless,Half Sleeves,Long Sleeves,Three-Quarter Sleeves.
Sleeve Style:Batwing Sleeves,Bell Sleeves,Flared Sleeves,Balloon Sleeves,Puffed Sleeves,Cold Sleeves,Shoulder Sleeves,Regular Sleeves,Slit Sleeves,Roll Up Sleeves,No Sleeves,Flutter Sleeve.
Fabric:Cotton,Polyester,Leather,Denim,Silk,Wool,Corduroy,Fleece,Schiffli,Terry,Crepe,Net,Georgette.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work,winters.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
Top Wear Length:Midi,Maxi,Mini,Above Knee,Cropped,Regular,hip length, waist length.


BOTTOMWEAR--

Products:Trouser,Jeans,Shorts,Trackpants,Joggers,Cargos,Skirts, dress pants.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colorblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Bottom Style:Flared,Straight,Loose Fit,A-Line,Peplum,Skinny Fit,Pencil, Slim,wide leg.
Bottom Length:Above Knee,Below Knee,Knee Length,Midi,Mini,Ankle,Maxi,Regular length,full length.
Waist Rise:High-Rise,Low-Rise,Mid-Rise.
Fabric:Cotton,Chambray,Polyester,Leather,Denim,Corduroy,Silk,Wool,Fleece,Velvet.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
"""

metadata_json = {"generated_tags": [
                                {
                                "products category": "string",
                                "Products": "string",
                                "Color": "string",
                                "Gender": "string",
                                "Pattern": "string",
                                "Silhouette": "string",
                                "Neckline": "string",
                                "Sleeve Length": "string",
                                "Sleeve Style": "string",
                                "Fabric": "string",
                                "Brand": "string",
                                "Occasion": "string",
                                "Fit Type": "string",
                                "Top Wear Length": "string",
                                },
                                 {
                                "products category": "string",
                                "Products": "string",
                                "Color": "string",
                                "Gender": "string",
                                "Pattern": "string",
                                "Bottom Style": "string",
                                "Waist Rise": "string",
                                "Bottom Length": "string",
                                "Fabric": "string",
                                "Brand": "string",
                                "Occasion": "string",
                                "Fit Type": "string",
                                }
                                ]
                }

predictor_system_message = f"""[INST] <image>
You will receive an image_path as input.
1. Analyze the provided catalog image and identify all the topwear and/or bottomwear items that are in focus and fully captured within the image.
2. For each identified clothing item, extract the following metadata:
	{tags}
3. Organize the metadata for each identified clothing item in a structured JSON format, with each item represented as a dictionary containing the specified keys.
4. If the color recognition tags provided do not cover all the colors present in the image, feel free to use additional color descriptors as necessary.
5. Ensure that the gender and material composition type are correctly mapped based on the identified products.
6. Carefully match the provided tags with the appropriate metadata entities, and do not include any additional comments in the JSON output.
7. If the image contains clothing items that are not fully captured or are out of focus, do not include them in the output.
8. If the image does not contain any topwear or bottomwear items, or if the provided tags do not match the contents of the image, respond with an empty JSON array.


<thinking>
Review the provided image and tags, and extract the relevant metadata for each identified clothing item according to the instructions above. Organize the metadata in a structured JSON format, ensuring that all required fields are populated accurately.
</thinking>

Answer in the below format and DO NOT explain anything else:

Format:
 {str(metadata_json)}
  [/INST]"""


# COMMAND ----------

import requests
import json
import os


img_url="Intellitag_Testing/Semi-Formal/Men/img2.jpg"

connection_string=dbutils.secrets.get(scope="dbx-us-scope",key="storage-account-connection-string") 
storage_container_name="intellitag"

temperature=0.1
max_tokens=510

data = {
   "inputs": [(img_url, predictor_system_message,connection_string,storage_container_name,temperature,max_tokens)]
 }

API_TOKEN = dbutils.secrets.get(scope="dbx-us-scope",key="databricks-PAT") 

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(url="https://adb-8566293101107632.12.azuredatabricks.net/serving-endpoints/affine_llava_v8/invocations",
                         json=data, 
                         headers=headers)

print(json.dumps(response.json()))

# COMMAND ----------

import ast

results=eval(json.dumps(response.json()))['predictions'][0]
eval(results[results.find("[/INST]")+8:].replace("```json","").replace("```","").strip())
