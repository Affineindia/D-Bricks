# Databricks notebook source
# MAGIC %md
# MAGIC ### Install All Modules

# COMMAND ----------

# MAGIC %pip install --quiet -r requirements.txt
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import all Required Packages

# COMMAND ----------

from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import autogen
######################################################
# Import necessary libraries for PySpark
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
############
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ast
import datetime
import time
import os
import shutil

# COMMAND ----------

# MAGIC %md
# MAGIC ### Env. Variables

# COMMAND ----------

storage_account_name = "adlsusdldev02"
storage_account_access_key =dbutils.secrets.get(scope="dbx-us-scope",key="storage-account-access-key") 
container_name = "intellitag"
mount_point = "/mnt/my_intellitag_mount"
database_name = "intellitag_catalog.intellitag_dbx"
table_name = "Result_Logs"
file_metadata="File_Metadata"

# COMMAND ----------

model="gpt4odeployment"
api_key=dbutils.secrets.get(scope="dbx-us-scope",key="azure-openai-llm-api-key") 
base_url=dbutils.secrets.get(scope="dbx-us-scope",key="azure-openai-llm-base-url") 
api_type= "azure"
api_version= "2024-02-01"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Azure Storage Mount

# COMMAND ----------

if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
    dbutils.fs.mount(
        source=f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
        mount_point=mount_point,
        extra_configs={f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### List down all the files from the specified storage path

# COMMAND ----------


def list_jpg_files_recursively(directory):
    files = []
    # List all items in the directory
    items = dbutils.fs.ls(directory)
    
    for item in items:
        if item.isDir():
            # print("-------------------Dir-----------------")
            # print(item.name)
            # If the item is a directory, recurse into it
            files.extend(list_jpg_files_recursively(item.path))
        elif item.path.endswith(".jpg"):
            # If the item is a .jpg file, add it to the list
            # print(len(item))
            files.append(item.path.replace("dbfs:/","/dbfs/")) #.replace("dbfs:/mnt/my_intellitag_mount/","")
    
    return files

directory_to_search = mount_point+"/Intellitag_Uploads" 

#+ "/Intellitag_Testing"  #"/Origamis Intellitag Testing"
# Get the list of all .jpg files
# jpg_files = list_jpg_files_recursively(directory_to_search)
# jpg_files

# COMMAND ----------

# MAGIC %md
# MAGIC #### Agent Set-up

# COMMAND ----------

llm_config_azure = [
    {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "api_type": api_type,
        "api_version": api_version,
        "temperature":0.0
    }
]

llm_config = {"config_list": llm_config_azure}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-defined Tags

# COMMAND ----------

tags = """products category:TOPWEAR,BOTTOMWEAR
TOPWEAR--

Products:Shirt,Jumpsuit,Dungarees,Sleepwear,Sweatshirt,Innerwear,Dress,Top,T-shirt,Polo Collar T-shirt,Sweater/Cardigan,Hoodie,Jackets,Raincoat,LongCoat,Blazer,Suits (2-piece or 3-piece Men),Kurta,Kurta Set,Maxi Dress,Showerproof.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colourblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Silhouette:A-line,Peplum,Balloon,Fit and Flare,Sheath,Bodycon,Shirt Style,Jumper,Wrap,Kaftan.
Neckline:Turtle/High Neck,Round Neck,Square Neck,Halter Neck,Scoop Neck,V neck Boat Neck,Polo Collar,Open-Collar,Crew Neck,Tie-Up Neck,Boat Neck,Shirt Neck.
Sleeve Length:Sleeveless,Short Sleeves,Long Sleeves,Three-Quarter Sleeves.
Sleeve Style:Batwing Sleeves,Bell Sleeves,Flared Sleeves,Balloon Sleeves,Puffed Sleeves,Cold Sleeves,Shoulder Sleeves,Regular Sleeves,Slit Sleeves,Roll Up Sleeves,No Sleeves,Flutter Sleeve.
Fabric:Cotton,Polyster,Leather,Denim,Silk,Wool,Corduroy,Fleece,Schiffli,Terry,Crepe,Net,Georgette.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work,winters.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
Top Wear Length:Midi,Maxi,Mini,Above Knee,Cropped,Regular.


BOTTOMWEAR--

Products:Trouser,Jeans,Shorts,Trackpants,Joggers,Cargos,Skirts.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colourblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Bottom Style:Flared,Loose Fit,A-Line,Peplum,Skinny Fit,Pencil.
Bottom Length:Above Knee,Below Knee,Knee Length,Midi,Mini,Ankle,Maxi,Regular length.
Waist Rise:High-Rise,Low-Rise,Mid-Rise.
Fabric:Cotton,Chambray,Polyster,Leather,Denim,Corduroy,Silk,Wool,Fleece,Velvet.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ##### A) User Proxy Agent

# COMMAND ----------

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Once the task is completed, answer 'TERMINATE-AGENT'",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE-AGENT" in msg["content"].lower(),
    code_execution_config=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### B) Predictor Agent

# COMMAND ----------

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
                                }
                                ]
                }

predictor_system_message = f"""
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
"""


predictor = MultimodalConversableAgent(
    name="predictor", system_message=predictor_system_message, llm_config=llm_config
)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### C) Evaluator Agent

# COMMAND ----------

sample_response_evaluator = {
    "generated_tags_with_critic": [
        [{"tag": "Products", "value": "T-shirt", "score": 1, "critic_message": ""},
        {"tag": "Color", "value": "Blue", "score": 1, "critic_message": ""},
        {"tag": "Pattern","value": "Solid","score": 0,"critic_message": "The pattern is actually striped."}]
    ],
}


evaluator = MultimodalConversableAgent(
    name="evaluator",
    system_message=f"""You are an evaluator agent. Your task is to evaluate the accuracy of the tags generated for a given image.
    You will receive an image and generated tags as inputs.
    Analyze the image thoroughly and for each of the generated tag, give a score of 0 or 1. 0 if incorrect, 1 if it is correct.
    If the score is 0 for a given tag, also give a critic message for that particular tag.
    Finally answer the image and your generated tags with score and message. Follow the example response given below
    
    Refer to the pre-defined tags and values. Your evaluation must be based on this.
    pre-defined tags: {tags}

    If your evaluation score for all the tags are 1 and all tags are correct, then must write 'ALL-GOOD' at last.
    Sample response format: 
    {sample_response_evaluator}""",
    llm_config=llm_config,
)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### D) Terminator Agent

# COMMAND ----------

terminator_system_message = f"""

You need to answer with 'max-3-tries'. Do NOT add any introductory phrase or do NOT explain anything else.

"""

tip_message = "\nIf you do your BEST WORK, I'll tip you $100!"

terminator = autogen.AssistantAgent(
    name="terminator",
    system_message=terminator_system_message + tip_message,
    human_input_mode="NEVER",
    llm_config=llm_config,
)

# COMMAND ----------

def check_name_occurrences(data, name_value, no_of_iters):
    count = sum(1 for entry in data if entry.get("name") == name_value)
    # print('count: ', count)
    return count >= no_of_iters


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    # print('messages: ', messages)
    # print('groupchat_messages', messages)
    if last_speaker is user_proxy:
        # init -> retrieve
        return predictor
    if last_speaker is predictor:
        return evaluator

    if last_speaker is evaluator:
        if "all-good" in messages[-1]["content"].lower():
            None
        elif check_name_occurrences(messages, "evaluator", 3):
            return terminator
        else:
            return predictor

groupchat = autogen.GroupChat(
    agents=[user_proxy, predictor, evaluator, terminator],
    messages=[],
    max_round=50,
    speaker_selection_method=state_transition,
)


manager_system_message = """
"""

manager = autogen.GroupChatManager(
    groupchat=groupchat, system_message=manager_system_message, llm_config=llm_config
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Delete Cache Folder
# MAGIC --------------------------------

# COMMAND ----------

def delete_Cach_folder():
      
    # Define the directory name
    dir_name = ".cache"
    
    # # Get the current working directory
    current_dir = os.getcwd()
    
    # # Construct the full path to the directory
    dir_path = os.path.join(current_dir, dir_name)
    
    # # Check if the directory exists
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        # Remove the directory and its contents
        shutil.rmtree(dir_path)
        print(f"The directory '{dir_name}' has been deleted.")
    else:
        print(f"The directory '{dir_name}' does not exist.")
 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Create `result_sample table` , if not exist
# MAGIC ---------------------------------------------------------------

# COMMAND ----------

## code to create result table
# 2. Initialize Spark Session (Databricks automatically initializes a Spark session for you, 
# but this step is needed if running elsewhere)
spark = SparkSession.builder.appName("Create Empty Table").getOrCreate()

# 3. Define the schema for your table
schema = StructType([
    StructField("id", StringType(), True),
    StructField("image_path", StringType(), True),
    StructField("evaluator", StringType(), True),
    StructField("predictor", StringType(), True),
    StructField("final_predictor", StringType(), True),
    StructField("text_vector_flag", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("cost", StringType(), True),
    StructField("model_type", StringType(), True)
])
# model type

# 4. Create an empty DataFrame using the schema
empty_df = spark.createDataFrame([], schema)

# Table name and database

full_table_name = f"{database_name}.{table_name}"

# 5. Check if the table exists
if not spark.catalog.tableExists(full_table_name):
    # 6. If the table does not exist, create it
    empty_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_table_name)
    print(f"Table {full_table_name} created successfully.")
else:
    print(f"Table {full_table_name} already exists.")

# 7. Verify that the table exists and is empty
# spark.sql(f"SELECT * FROM {full_table_name}").show()  # Should show no rows if empty


# COMMAND ----------

# MAGIC %md  
# MAGIC ##### Get Final Dictionary 
# MAGIC ---------------------------------------------------------------

# COMMAND ----------

def get_final_dict(row,result, model_type):
    if model_type.lower()=="gpt-4o":
        print("GPT-40 Results Formating...")
        try:
            final_dict={}
            final_dict['id']=row.id
            final_dict['image_path']=row.file_path
            final_dict['model_type']=model_type
            eval_dict={}
            for num,ev in enumerate([history for history in result.chat_history if history['name']=="evaluator"]):
                eval_dict[num]=eval(ev['content'].replace("```json","").replace("```","").replace("ALL-GOOD",""))

            predict_dict={}
            for num,pr in enumerate([history for history in result.chat_history if history['name']=="predictor"]):
                # predictor_results=pr['content'][pr['content'].find("["):pr['content'].rfind("]")+1].replace("```json","").replace("```","")
                # predict_dict[num]=ast.literal_eval(predictor_results)
                predict_dict[num]=eval(pr['content'].replace("```json","").replace("```",""))

            final_dict['evaluator']=str(eval_dict)
            final_dict['predictor']=str(predict_dict)
            final_dict['final_predictor']=str(predict_dict[max(predict_dict.keys())])
            final_dict['summary']=result.summary
            final_dict['cost']=result.cost
            # flag will be True if text embedding is done
            final_dict['text_vector_flag']="No"
            return final_dict
        except Exception as e:
            print(e)
            return None
        
    elif model_type.lower()=="llava":
        print("LLava Results Formating...")
        try:
            final_dict={}
            final_dict['id']=row.id
            final_dict['image_path']=row.file_path
            final_dict['model_type']=model_type
            eval_dict={}
            predict_dict={}
            final_dict['evaluator']=str(eval_dict)
            final_dict['predictor']=str(predict_dict)
            final_dict['final_predictor']=result
            final_dict['summary']=""
            final_dict['cost']=""
            # flag will be True if text embedding is done
            final_dict['text_vector_flag']="No"
            return final_dict
        except Exception as e:
            print(e)
            return None



# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Save Final Dict to Table
# MAGIC ---------------------------------------------------------------

# COMMAND ----------

def save_final_dict_to_table(final_dict):      
    try:  
        # 3. Create a DataFrame from the data_dict using the schema of the existing table
        data_df = spark.createDataFrame([final_dict], schema=schema)

        # 4. Insert the data into the existing Delta table
        data_df.write.format("delta").mode("append").saveAsTable(full_table_name)
        return True
    except:
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLava Vision LLM Model 

# COMMAND ----------

topwear_tags = """products category:TOPWEAR
Products:Shirt,Jumpsuit,Dungarees,Sleepwear,Sweatshirt,Innerwear,Dress,Top,T-shirt,Polo Collar T-shirt,Sweater/Cardigan,Hoodie,Jackets,Raincoat,LongCoat,Blazer,Suits (2-piece or 3-piece Men),Kurta,Kurta Set,Maxi Dress,Showerproof.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colourblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Silhouette:A-line,Peplum,Balloon,Fit and Flare,Sheath,Bodycon,Shirt Style,Jumper,Wrap,Kaftan.
Neckline:Turtle/High Neck,Round Neck,Square Neck,Halter Neck,Scoop Neck,V neck Boat Neck,Polo Collar,Open-Collar,Crew Neck,Tie-Up Neck,Boat Neck,Shirt Neck.
Sleeve Length:Sleeveless,Short Sleeves,Long Sleeves,Three-Quarter Sleeves.
Sleeve Style:Batwing Sleeves,Bell Sleeves,Flared Sleeves,Balloon Sleeves,Puffed Sleeves,Cold Sleeves,Shoulder Sleeves,Regular Sleeves,Slit Sleeves,Roll Up Sleeves,No Sleeves,Flutter Sleeve.
Fabric:Cotton,Polyster,Leather,Denim,Silk,Wool,Corduroy,Fleece,Schiffli,Terry,Crepe,Net,Georgette.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work,winters.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
Top Wear Length:Midi,Maxi,Mini,Above Knee,Cropped,Regular.
"""
topwear_metadata_json = {
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
                        }

# COMMAND ----------

bottomwear_tags = """products category:BOTTOMWEAR
Products:Trouser,Jeans,Shorts,Trackpants,Joggers,Cargos,Skirts.
Color:Pink,Red,Blue,Black,Grey,Navy Blue,Charcoal Grey,White,Green,Olive,Brown,Beige,Khaki,Cream,Maroon,Off White,Grey Melange,Teal,Coffee Brown,Pink,Mustard,Purple,Rust,Sea Green,Burgundy,Turquoise Blue,Taupe,Silver,Mauve,Orange,Yellow,Multi,Lavender,Tan,Peach,Magenta,Fluorescent Green,Coral,Copper.
Gender:Men,Women,Girls,Boys.
Pattern:Striped,Checked,Embellished,Ribbed,Colourblocked,Dyed,Printed,Embroidered,Self-Design,Solid,Graphic,Floral,Polka Dots,Camouflage,Animal,Self-Design,Ombre.
Bottom Style:Flared,Loose Fit,A-Line,Peplum,Skinny Fit,Pencil.
Bottom Length:Above Knee,Below Knee,Knee Length,Midi,Mini,Ankle,Maxi,Regular length.
Waist Rise:High-Rise,Low-Rise,Mid-Rise.
Fabric:Cotton,Chambray,Polyster,Leather,Denim,Corduroy,Silk,Wool,Fleece,Velvet.
Brand:Mango,Puma,Adidas,Nike,Calvin Klein,Lacoste,Fred Perry,Brooks Brothers,GAP,Levis,GANT,Superdry,Tommy Hilfiger,H&M,Zara,Louis Phillipe,Polo Raulph Lauren,Guess,Gucci,Prada,Versace,Aeropostale,Abercrombie & Fitch,DKNY,Michael Kors,Coach,Fendi.
Occasion:Casual,Formal,Parties/Evening,Sports,Maternity,Ethnic,Everyday,Work.
Fit Type:Slim Fit,Oversized,Regular,Skinny Fit,Loose Fit.
"""
bottomwear_metadata_json = {
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
                          

# COMMAND ----------

import requests
import json
import os
 
topwear_llava_system_message = f"""[INST] <image>
You will receive an image_path as input.
1. Analyze the provided catalog image and identify all the topwear items that are in focus and fully captured within the image.
2. For each identified clothing item, extract the following metadata:
    {topwear_tags}
3. Organize the metadata for each identified clothing item in a structured JSON format, with each item represented as a dictionary containing the specified keys.
4. If the color recognition tags provided do not cover all the colors present in the image, feel free to use additional color descriptors as necessary.
5. Ensure that the gender and material composition type are correctly mapped based on the identified products.
6. Carefully match the provided tags with the appropriate metadata entities, and do not include any additional comments in the JSON output.
7. If the image contains clothing items that are not fully captured or are out of focus, do not include them in the output.
8. If the image does not contain any topwear items, or if the provided tags do not match the contents of the image, respond with an empty JSON array.
 
 
<thinking>
Review the provided image and tags, and extract the relevant metadata for each identified clothing item according to the instructions above. Organize the metadata in a structured JSON format, ensuring that all required fields are populated accurately.If the image does not contain any topwear items, must return empty dictionary.
</thinking>
 
Answer in the below format and DO NOT explain anything else:
 
Format:
 {str(topwear_metadata_json)}
  [/INST]"""

# COMMAND ----------

import requests
import json
import os


bottomwear_llava_system_message = f"""[INST] <image>
You will receive an image_path as input.
1. Analyze the provided catalog image and identify all the bottomwear items that are in focus and fully captured within the image.
2. For each identified clothing item, extract the following metadata:
    {bottomwear_tags}
3. Organize the metadata for each identified clothing item in a structured JSON format, with each item represented as a dictionary containing the specified keys.
4. If the color recognition tags provided do not cover all the colors present in the image, feel free to use additional color descriptors as necessary.
5. Ensure that the gender and material composition type are correctly mapped based on the identified products.
6. Carefully match the provided tags with the appropriate metadata entities, and do not include any additional comments in the JSON output.
7. If the image contains clothing items that are not fully captured or are out of focus, do not include them in the output.
8. If the image does not contain any bottomwear items, or if the provided tags do not match the contents of the image, respond with an empty JSON array.
 
 
<thinking>
Review the provided image and tags, and extract the relevant metadata for each identified clothing item according to the instructions above. Organize the metadata in a structured JSON format, ensuring that all required fields are populated accurately.If the image does not contain any bottomwear items, must return empty dictionary.
</thinking>
 
Answer in the below format and DO NOT explain anything else:
 
Format:
 {str(bottomwear_metadata_json)}
  [/INST]"""


# COMMAND ----------

def llava_call(img_url,llava_system_message, databricks_llm_url, databricks_pat, temperature=0.1, max_tokens=4000):
    """
    Sends a request to the Databricks LLM (Large Language Model) API to generate a response
    based on the provided image URL and other parameters, then extracts and returns the processed response.

    Args:
        img_url (str): The URL of the image to be used as input for the model.
        databricks_llm_url (str): The endpoint URL of the Databricks LLM.
        databricks_pat (str): Personal Access Token (PAT) for authenticating with the Databricks API.
        temperature (float, optional): The temperature parameter to control the creativity of the response. Defaults to 0.1.
        max_tokens (int, optional): The maximum number of tokens allowed in the model's response. Defaults to 510.

    Returns:
        str: The processed text output from the LLM response, with specific markers and code blocks removed.
    """

    # Prepare the data payload with input parameters including the image URL, system message, and temperature
    data = {
        "inputs": [(img_url,
                    llava_system_message,
                    connection_string,
                    storage_container_name,
                    temperature, max_tokens)]
    }

    # Set up headers, including the authorization token for Databricks API
    headers = {"Context-Type": "text/json",
               "Authorization": f"Bearer {databricks_pat}"}

    # Make a POST request to the Databricks LLM API with the provided URL, data, and headers
    response = requests.post(url=databricks_llm_url,
                             json=data, 
                             headers=headers)

    # Parse the JSON response into a Python object and extract the predictions text
    text = ast.literal_eval(json.dumps(response.json()))["predictions"][0]

    # Define the marker that indicates where the actual response starts
    marker = '[/INST]'
    position = text.find(marker)

    # If the marker is found, extract the text after the marker and clean it by removing code blocks
    if position != -1:
        result = text[position + len(marker):].strip().replace("```json", "").replace("```", "")
    else:
        # If the marker is not found, return an empty string
        result = ""

    # Return the final cleaned result
    return result


# COMMAND ----------

def llava_response(img_url):        
    result={"generated_tags":[]}
    msg="""[INST] <image> Find out the apparel in focus present in the given image, from below list:["topwear", "bottomwear"].
    example:["topwear"],["bottomwear"],["topwear", "bottomwear"], []
    output:
    [/INST]"""
    item_list=llava_call(img_url=img_url,
                                llava_system_message=msg,
                                databricks_llm_url=llava_base_url,
                                databricks_pat=API_TOKEN)
    for item in ast.literal_eval(item_list):
        if item=="topwear":
            llava_system_message=topwear_llava_system_message
        elif item=="bottomwear":
            llava_system_message=bottomwear_llava_system_message
    # for llava_system_message in [topwear_llava_system_message,bottomwear_llava_system_message]:
        result_=llava_call(img_url=img_url,
                        llava_system_message=llava_system_message,
                        databricks_llm_url=llava_base_url,
                        databricks_pat=API_TOKEN
                        )
        print(result_)
        if result_:
            result['generated_tags'].append(ast.literal_eval(result_))
    return result

# COMMAND ----------

uploadedby = dbutils.widgets.get("uploadedby")
date_str=dbutils.widgets.get("generatedat")
tag_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
# uploadedby = "sanket.bodake@affine.ai"
# date_str="12:3444"
# tag_datetime = "12:44"

connection_string=dbutils.secrets.get(scope="dbx-us-scope",key="storage-account-connection-string") 
storage_container_name="intellitag"
API_TOKEN =dbutils.secrets.get(scope="dbx-us-scope",key="databricks-PAT") 
llava_base_url="https://adb-8566293101107632.12.azuredatabricks.net/serving-endpoints/affine_llava_v8/invocations"
tag_flag='No'


df=spark.sql(f"SELECT * FROM intellitag_catalog.intellitag_dbx.{file_metadata} WHERE created_by=='{uploadedby}' and tag_generated=='{tag_flag}'")

display(df)
# final_dict_list=[]
for row in df.collect():

    #--------------------- Load and display the image----------------------------
    image_path="/dbfs/mnt/my_intellitag_mount/"+row.file_path
    model_type= row.model

    if model_type.lower()=="gpt-4o":
    
        #-----------------------Run the Agent ---------------------------------------
        img_path_format = f"image_path:   <img {image_path}>"
        result = user_proxy.initiate_chat(
            manager, message=img_path_format, summary_method="reflection_with_llm"
        )
 
        #----------------------Get final dict --------------------------
        final_dict=get_final_dict(row,result,model_type)

    elif model_type.lower()=="llava":
        result=llava_response(img_url=row.file_path)
        final_dict=get_final_dict(row,result,model_type)
        # final_dict_list.append(final_dict)
       
     #----------------------save final dict to table --------------------------
    if final_dict:
        saved_flag=save_final_dict_to_table(final_dict)

    #-------------------------------------------------------------------------------
    if saved_flag:
        # Perform the update using SQL
        spark.sql(f"""
            UPDATE intellitag_catalog.intellitag_dbx.{file_metadata}
            SET tag_generated = 'Yes',
                tag_generated_at='{tag_datetime}'
            WHERE id = '{row.id}'
        """)
        
    # delete cache folder
    delete_Cach_folder()
    time.sleep(30)

# COMMAND ----------

dbutils.notebook.exit("Done")

# COMMAND ----------

# import csv

# def convert_dict_to_csv(data, csv_filename):
#     # Check if the data is not empty
#     if data is not None and len(data) > 0:
#         # Extract the keys from the first dictionary in the list to use as column headers
#         keys = data[0].keys()
        
#         # Open the CSV file in write mode
#         with open(csv_filename, mode='w', newline='') as file:
#             # Create a DictWriter object with the extracted keys as the fieldnames (columns)
#             writer = csv.DictWriter(file, fieldnames=keys)
            
#             # Write the header (column names)
#             writer.writeheader()
            
#             # Write all the dictionaries in the list as rows in the CSV
#             writer.writerows(data)
        
#         print(f"Data has been successfully written to {csv_filename}.")
#     else:
#         print("The provided data list is empty.")

# # Specify the CSV file name
# csv_filename = "llava_output_top_bottom1.csv"

# # Call the function to convert the list of dictionaries into a CSV
# convert_dict_to_csv(final_dict_list, csv_filename)


# COMMAND ----------

# msg="""[INST] <image> Find out the apparel in focus present in the given image, from below list:["topwear", "bottomwear"].
# example:["topwear"],["bottomwear"],["topwear", "bottomwear"], []
# output:
# [/INST]"""
# item_list=llava_call(img_url="llava_testing/Buy Grey Fusion Fit Mens Cotton Trouser Online _ Tistabene 28.png",
#                             llava_system_message=msg,
#                             databricks_llm_url=llava_base_url,
#                             databricks_pat=API_TOKEN)
# for item in ast.literal_eval(item_list):
#     print(item)

# COMMAND ----------

# import requests
# import json
# import os

# metadata_json = {"generated_tags": [
#                                 {
#                                 "products category": "string",
#                                 "Products": "string",
#                                 "Color": "string",
#                                 "Gender": "string",
#                                 "Pattern": "string",
#                                 "Silhouette": "string",
#                                 "Neckline": "string",
#                                 "Sleeve Length": "string",
#                                 "Sleeve Style": "string",
#                                 "Fabric": "string",
#                                 "Brand": "string",
#                                 "Occasion": "string",
#                                 "Fit Type": "string",
#                                 "Top Wear Length": "string",
#                                 },
#                                  {
#                                 "products category": "string",
#                                 "Products": "string",
#                                 "Color": "string",
#                                 "Gender": "string",
#                                 "Pattern": "string",
#                                 "Bottom Style": "string",
#                                 "Waist Rise": "string",
#                                 "Bottom Length": "string",
#                                 "Fabric": "string",
#                                 "Brand": "string",
#                                 "Occasion": "string",
#                                 "Fit Type": "string",
#                                 }
#                                 ]
#                 }
 
# llava_system_message = f"""[INST] <image>
# You will receive an image_path as input.
# 1. Analyze the provided catalog image and identify all the topwear and/or bottomwear items that are in focus and fully captured within the image.
# 2. For each identified clothing item, extract the following metadata:
#     {tags}
# 3. Organize the metadata for each identified clothing item in a structured JSON format, with each item represented as a dictionary containing the specified keys.
# 4. If the color recognition tags provided do not cover all the colors present in the image, feel free to use additional color descriptors as necessary.
# 5. Ensure that the gender and material composition type are correctly mapped based on the identified products.
# 6. Carefully match the provided tags with the appropriate metadata entities, and do not include any additional comments in the JSON output.
# 7. If the image contains clothing items that are not fully captured or are out of focus, do not include them in the output.
# 8. If the image does not contain any topwear or bottomwear items, or if the provided tags do not match the contents of the image, respond with an empty JSON array.
 
 
# <thinking>
# Review the provided image and tags, and extract the relevant metadata for each identified clothing item according to the instructions above. Organize the metadata in a structured JSON format, ensuring that all required fields are populated accurately.
# </thinking>
 
# Answer in the below format and DO NOT explain anything else:
 
# Format:
#  {str(metadata_json)}
#   [/INST]"""



# def llava_call(img_url, databricks_llm_url, databricks_pat, temperature=0.1, max_tokens=4000):
#     """
#     Sends a request to the Databricks LLM (Large Language Model) API to generate a response
#     based on the provided image URL and other parameters, then extracts and returns the processed response.

#     Args:
#         img_url (str): The URL of the image to be used as input for the model.
#         databricks_llm_url (str): The endpoint URL of the Databricks LLM.
#         databricks_pat (str): Personal Access Token (PAT) for authenticating with the Databricks API.
#         temperature (float, optional): The temperature parameter to control the creativity of the response. Defaults to 0.1.
#         max_tokens (int, optional): The maximum number of tokens allowed in the model's response. Defaults to 510.

#     Returns:
#         str: The processed text output from the LLM response, with specific markers and code blocks removed.
#     """
#     # Prepare the data payload with input parameters including the image URL, system message, and temperature
#     data = {
#         "inputs": [(img_url,
#                     llava_system_message,
#                     connection_string,
#                     storage_container_name,
#                     temperature, max_tokens)]
#     }

#     # Set up headers, including the authorization token for Databricks API
#     headers = {"Context-Type": "text/json",
#                "Authorization": f"Bearer {databricks_pat}"}

#     # Make a POST request to the Databricks LLM API with the provided URL, data, and headers
#     response = requests.post(url=databricks_llm_url,
#                              json=data, 
#                              headers=headers)

#     # Parse the JSON response into a Python object and extract the predictions text
#     text = ast.literal_eval(json.dumps(response.json()))["predictions"][0]

#     # Define the marker that indicates where the actual response starts
#     marker = '[/INST]'
#     position = text.find(marker)

#     # If the marker is found, extract the text after the marker and clean it by removing code blocks
#     if position != -1:
#         result = text[position + len(marker):].strip().replace("```json", "").replace("```", "")
#     else:
#         # If the marker is not found, return an empty string
#         result = ""

#     # Return the final cleaned result
#     return result


# COMMAND ----------

# MAGIC %md
# MAGIC 1. ALL tag ok, but still ALL-GOOD Not written in evaluator.
# MAGIC 2. Predictor format 
