import databricks.sql as dbsql
from dotenv import load_dotenv
import os
import streamlit as st

# load_dotenv()

class DatabrickSqlConnect():
    """
    A class to establish a connection to a Databricks SQL endpoint.

    Attributes:
        server_hostname (str): The hostname of the Databricks server.
        http_path (str): The HTTP path for connecting to the Databricks SQL endpoint.
        access_token (str): The access token for authentication.
        sql_connection: The SQL connection object to interact with Databricks SQL.
    """

    def __init__(self):
        """
        Initializes the DatabrickSQL class and establishes a connection to the Databricks SQL endpoint.
        
        Raises:
            Exception: If there is an error during the connection initialization.
        """
        try:
            self.server_hostname = os.getenv('server_hostname') or st.secrets["credentials"]["server_hostname"] # Retrieve the server hostname from the environment variable
            self.http_path = os.getenv('http_path') or st.secrets["credentials"]["http_path"] # Retrieve the HTTP path from the environment variable
            self.access_token = os.getenv('access_token') or st.secrets["credentials"]["access_token"]  # Retrieve the access token from the environment variable
            # Establish a connection to the Databricks SQL endpoint using the retrieved credentials
            self.sql_connection = dbsql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token
            )

        except Exception as e:
            # Print an error message if there is an issue during initialization
            print("Error in DatabrickSQL initialization:", e)




if __name__ == '__main__':
    obj=DatabrickSqlConnect()
    print(obj.sql_connection)
        