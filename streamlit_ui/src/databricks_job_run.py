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
    """
    A class to interact with Databricks Jobs API to run jobs and retrieve job results.
    
    Attributes:
    ----------
    workspace_url : str
        The URL of the Databricks workspace (fetched from environment variables or secrets).
    token : str
        The access token for Databricks API authentication (fetched from environment variables or secrets).
    """

    def __init__(self) -> None:
        """
        Initializes the DatabrickJob class with workspace URL and access token.
        
        The workspace URL and token are fetched from environment variables or 
        Streamlit secrets. If not found in environment variables, the credentials 
        from Streamlit's secrets will be used.
        """
        self.workspace_url=os.getenv('workspace_url') or st.secrets["credentials"]["workspace_url"]  
        self.token=os.getenv('access_token') or st.secrets["credentials"]["access_token"] 

    def job_runs(self,job_id:int,input_data:dict)-> int:
        """
        Triggers a Databricks job run using the provided job ID and input data.

        Parameters:
        ----------
        job_id : int
            The ID of the Databricks job to run.
        input_data : dict
            Additional parameters or configurations required to run the job.

        Returns:
        -------
        int
            The ID of the job run (run_id) if the job is successfully triggered.

        Raises:
        ------
        Exception
            If the job run fails, an error message will be printed with the HTTP status code.
        """
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


    def get_job_result(self,run_id:int)-> dict:
        """
        Retrieves the result of a specific Databricks job run based on the provided run ID.

        Parameters:
        ----------
        run_id : int
            The ID of the Databricks job run to retrieve results for.

        Returns:
        -------
        dict
            A dictionary containing the job run result, including job output and status.

        Raises:
        ------
        Exception
            If the job run encounters an internal error or any other failure, 
            an error message will be returned.
        """
        # Define the API endpoint to get job run status
        url = f'{self.workspace_url}/api/2.0/jobs/runs/get?run_id={run_id}'
        headers = {
                    'Authorization': f'Bearer {self.token}',
                    'Content-Type': 'application/json'
                    }

        # Poll the job run status until it is terminated or encounters an error
        while True:
            response = requests.get(url, headers=headers)
            run_status = response.json()
            if run_status['state']['life_cycle_state'] == 'TERMINATED':
                break
            elif run_status['state']['life_cycle_state'] == 'INTERNAL_ERROR':
                return run_status
            time.sleep(2)

        # Define the API endpoint to get the job run output
        output_url = f'{self.workspace_url}/api/2.0/jobs/runs/get-output?run_id={run_id}'
        output_response = requests.get(output_url, headers=headers)
        output_data = output_response.json()
        return output_data



