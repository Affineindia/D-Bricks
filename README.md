# Assetfind AI- AssetTag & AssetFind Tool 
~ *Databricks Hackathon*


### Description
This project automates the identification and tagging of key product attributes using prompt engineering and Multimodal LLM (e.g. GPT-4O & LLava). It integrates a GenAI-based search engine for searching the right assets (Retrieval Augmented Search) and also GenAI-generated tags-based search, leveraging Azure OpenAI/Databricks LLMs and other Databricks capabilities. Databricks Unity Catalog is used to store and manage data, while scheduled jobs handle automated runs. The output is an interactive dashboard for seamless image uploads and attribute-based search.


### Table of Contents
1. [Folder Structure](#folder-structure)
2. [Technologies Used](#technologies-used)
3. [Technical Architecture](#technical-architecture)
4. [Databricks LLMOPs Architecture](#databricks-llmops-architecture)
5. [Databricks Workflows](#databricks-workflows)
6. [How to Run the Streamlit Application](#how-to-run-the-streamlit-application)


## Folder Structure
Here is the structure of the project:

```plaintext
  <Affine Assetfind AI>/
  │
  ├── jobs/
  │   ├── demo - Embeddings_Indexing
  │   ├── demo - Asset Tag
  │   ├── demo - Asset Find
  │   └── requirements.txt
  │    
  ├── notebook/   
  │   ├── LLM Registry                       # External Hugging Face LLM Registry
  │   └── Evaluation_without_and_with_agent  # custom evaluation matrix defined
  │   
  ├── streamlit ui/  
  │   ├── Dockerfile  # dockerfile to containerize the application
  │   ├── app.py
  │   ├──.dockerignore
  │   ├── pages_call/
  │   │   ├── assetfind_page.py
  │   │   ├── login_page.py
  │   │   ├── upload_page.py
  │   │   ├── user_add_page.py
  │   │   └── view_tags_page.py
  │   │
  │   ├── src/
  │   │   ├── data_load.py
  │   │   ├── databricks_job_run.py
  │       ├── databricks_sql_connector.py
  │       ├── login_call.py
  │       └── processing.py
  │
  ├── .gitignore
  ├── requirements.txt
  └── README.md
```
--------------------------------------------
## Technologies Used
1. **Azure Databricks**: 

      Offers a unified platform for processing, analyzing, and generating insights from data, while facilitating the seamless integration of LLM models and automated workflows.

2. **Databricks Unity Catalog**: 

      Provides a centralized governance layer to manage and secure data across Databricks, ensuring organized storage and retrieval of image attributes and search results.

3. **Databricks Delta Table Indexing**:

      Optimizes data retrieval by enabling faster querying and efficient access to large datasets, crucial for managing image metadata.

4. **Databricks Vector Search Endpoint**: 

      Powers high-performance similarity searches for image and text embeddings, enabling efficient search functionalities.

5. **Databricks Model Registry & Serving**: 

      Manages model tracking and deployment, ensuring seamless scaling and real-time or batch inference for machine learning models.

6. **Azure OpenAI**: 

      Provides large language models (LLMs) to generate product tags and perform text/image-to-image searches using prompt engineering, enabling intelligent and automated search capabilities.
   
7. **Azure Blob Storage**: 

      Stores product images and metadata efficiently, serving as a scalable and secure repository for the image data used in the project.

8. **Azure Key Vault**: 

      Azure Key Vault is a cloud service used to securely store and manage sensitive information like secrets, keys, and certificates for protecting access to applications and resources.


9. **Autogen Agent**: 
              
      An automation tool that streamlines the image tagging and search processes, reducing the need for manual intervention by continuously improving search and tagging results.

10. **GitHub**: 

      Used for version control and collaboration, hosting the project's codebase, ensuring continuous integration, and managing updates to the search engine and tagging framework.


11. **Streamlit**:
  
      Python-based framework used to develop the interactive dashboard, allowing users to upload images, perform keyword-based searches, and view results in real time.

------------------------------------------------------------------------
## Technical Architecture

Below is the technical architecture of AssetTag and AssetFind tool.
![image](https://github.com/user-attachments/assets/9ccece55-4d4c-4cea-b094-3a2736b7ea29)


------------------------------------------------------------------
## Databricks LLMOPs Architecture
![image](https://github.com/user-attachments/assets/964e9ecc-d53f-4e8b-8a50-c52cc140db76)



-----------------------------------------------------------------
## Databricks Workflows

### DAG Flow Overview

##### Overview

The DAG (Directed Acyclic Graph) represents the workflow for the Assetfind AI project, orchestrating various tasks in a sequential and conditional manner to achieve specific data processing, keyword generation, indexing and semantic search objectives. This DAG runs from the databricks api.

##### Task Dependencies

Tasks within the DAG are organized based on dependencies, ensuring that certain tasks are executed only after their prerequisite tasks have completed successfully. The dependencies are defined using the depends_on attribute, specifying the task keys on which each task depends.

##### Task Execution Conditions

Some tasks in the DAG are executed conditionally based on the outcome of previous tasks or external conditions. Conditional execution logic is defined using the `condition_task` attribute, which compares a specific value or expression with the outcome of a preceding task.

##### Task Details

Below is a detailed breakdown of the tasks within the DAG:

1. AssetTag Task (`Asset Tag`):
  - Responsible for generating tags based on the defined taxonomy and prompts, using the `demo - Asset Tag.py` script. The results are saved into the catalog Delta table.
  - This task is executable on the Databricks cluster.
  - Generate the Tags from marketing asset Task (`demo - AssetFind`):

2. Indexes the generated tags and images Task (`Embeddings_Indexing`):
  - Creates the Databricks vector endpoint if it doesn't already exist.
  - Syncs the embedding table with the vector store.
  - This task depends on the successful completion of the AssetTag task.

3. AssetFind Task (`Asset Find`):

  - Runs through the Databricks API.
  - Performs semantic search based on user input data.
  - Provides recommendations based on user inputs using the `demo - Asset Find.py`.



-------------------------------
####  Note

This documentation provides a comprehensive overview of the DAG flow within the Databricks workflow for AssetTag, keyword indexing, and uploaded images as well as AssetFind. Understanding this flow is essential for effectively managing and optimizing task execution to achieve project objectives efficiently.

1. *AssetTag & Indexing of Tags/Images*

![image](https://github.com/user-attachments/assets/c7481533-db5c-45e1-abc3-1b1f521f419e)

2. *AssetFind*

![image](https://github.com/user-attachments/assets/a00526b5-73a4-42c5-802a-a48f662b0492)


## How to Run the Streamlit Application

1. Clone the Repository
  Run the following command to clone the repository:
  ```bash
  git clone https://github.com/sanket-affine/D-Bricks.git
  ```
2. Navigate to the Directory
  Change the directory to the streamlit_ui folder:
```bash
cd streamlit_ui
```
3. Install Requirements
  Install the required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Application
  Start the Streamlit application with the following command:
```bash
streamlit run app.py --global.developmentMode=false
```

## Streamlit UI

1. AssetTag View

![image](https://github.com/user-attachments/assets/d99a4adc-bb0b-43ae-b5c9-8de5e3fb0ee3)

2. AssetFind 

A. Retrieval Augmented Search
![image](https://github.com/user-attachments/assets/c7567cc2-e567-4c2c-9fd8-b12896c81d18)

B. TagSearch
![image](https://github.com/user-attachments/assets/0fcc30d2-83d6-459b-9172-c55d24b9b6a1)







