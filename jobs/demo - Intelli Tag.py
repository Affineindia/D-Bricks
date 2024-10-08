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

### List Available Mounts
# dbutils.fs.mounts()

#### List Files in a Directory
# display(dbutils.fs.ls(mount_point + "/Origamis Intellitag Testing/*/*/*.jpg"))

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

directory_to_search = mount_point+"/Intellitag_Uploads" #+ "/Intellitag_Testing"  #"/Origamis Intellitag Testing"
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

uploadedby = dbutils.widgets.get("uploadedby")
date_str=dbutils.widgets.get("generatedat")
tag_datetime = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')

tag_flag='No'

df=spark.sql(f"SELECT * FROM intellitag_catalog.intellitag_dbx.{file_metadata} WHERE created_by=='{uploadedby}' and tag_generated=='{tag_flag}'")

display(df)

for row in df.collect():
    #--------------------- Load and display the image----------------------------
    image_path="/dbfs/mnt/my_intellitag_mount/"+row.file_path
   
    #-----------------------Run the Agent ---------------------------------------
    img_path_format = f"image_path:   <img {image_path}>"
    result = user_proxy.initiate_chat(
        manager, message=img_path_format, summary_method="reflection_with_llm"
    )

    #----------------------Get final dict --------------------------
    final_dict=get_final_dict(row,result,"gpt-4o")

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

# MAGIC %md
# MAGIC 1. ALL tag ok, but still ALL-GOOD Not written in evaluator.
# MAGIC 2. Predictor format 
