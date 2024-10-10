import requests
import datetime
import os
import json
from dotenv import load_dotenv
import time
# from src.databricks_sql_connector import DatabrickSqlConnect
import streamlit as st

load_dotenv()

class DatabrickJob():

    def __init__(self) -> None:
        self.workspace_url=os.getenv('workspace_url') or st.secrets["credentials"]["workspace_url"]  
        self.token=os.getenv('access_token') or st.secrets["credentials"]["access_token"] 

    def job_runs(self,job_id:int,input_data:dict):
        input_data['job_id']=job_id
        # Define the API endpoint for running the job
        endpoint = f"{self.workspace_url}/api/2.1/jobs/run-now"

        # Set up the request headers with authentication
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Make the API request to run the job
        response = requests.post(endpoint, headers=headers, data=json.dumps(input_data))

        # Check the response
        if response.status_code == 200:
            run_id = response.json().get("run_id")
            print(f"Job run started successfully with run_id: {run_id}")
            return run_id,id
        else:
            print(f"Failed to start job run: {response.status_code} - {response.text}")


    def get_job_result(self,run_id:int):
        url = f'{self.workspace_url}/api/2.0/jobs/runs/get?run_id={run_id}'
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
        while True:
            response = requests.get(url, headers=headers)
            run_status = response.json()
            if run_status['state']['life_cycle_state'] == 'TERMINATED':
                break
            elif run_status['state']['life_cycle_state'] == 'INTERNAL_ERROR':
                return run_status
            time.sleep(2)
    
        output_url = f'{self.workspace_url}/api/2.0/jobs/runs/get-output?run_id={run_id}'
        output_response = requests.get(output_url, headers=headers)
        output_data = output_response.json()
        return output_data



