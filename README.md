# IntelliTag & IntelliSearch Tool ~ Databricks Hackathon
------------------------------

### Description
---------------------------
This project


### Table of Contents
-------------------------------
1. [Folder Structure](#)
2. [Technologies Used](#)
3. [Intellitag flow Diagram](#)
4. [IntelliSearch Flow Diagram](#)
5. [Databricks Workflows](#)
6. [Streamlit UI](#)

1. Folder Structure
------------------------------
Here is the structure of the project:

```plaintext
  <AffineIntelliTag-Search>/
  │
  ├── prompts/
  │   ├── dos_prompt.py
  │   ├── summarize_prompt.py
  │   └── lexical_data.py
  │   
  ├── review_summerization/
  │   ├── drivers_of_sentiments.py
  │   └── summerization.py
  │   
  ├── utility/
  │   ├── azure_open_api.py
  │   └── data_source.py
  │
  ├── Dockerfile
  ├── docker-compose.yml
  ├── crontab
  ├── requirements.txt
  ├── DOS.py
  ├── app.py # streamlit application
  ├── dos_run.py # Driver of sentiment analysis
  ├── summerizer_run.py # Summerizer of reviews
  ├── run.sh
  └── README.md
```



