import uuid
from src.databricks_sql_connector import DatabrickSqlConnect
import os
import streamlit as st


class Processing(DatabrickSqlConnect):
    """
    This class handles the processing tasks related to blob name sanitization and unique ID generation 
    for images, leveraging the Databrick SQL connection.
    """

    def __init__(self):
        """
        Initializes the Processing class by inheriting from DatabrickSqlConnect, 
        setting up the SQL connection for database operations.
        """
        super().__init__()

    def sanitize_blob_name(self,blob_name):
        """
        Sanitizes the provided blob name by replacing problematic characters 
        (like backslashes, question marks, and hash symbols) with safer characters.

        Parameters:
        ----------
        blob_name : str
            The name of the blob file to sanitize.
        
        Returns:
        -------
        str
            The sanitized blob name with backslashes replaced by forward slashes and 
            any question marks or hash symbols replaced by underscores.
        """
        # Replace any problematic characters with underscores
        return blob_name.replace("\\", "/").replace("?", "_").replace("#", "_")

    def generate_unique_id(self):
        """
        Generates a unique image UUID that is not present in the `file_metadata` table.
        
        This function fetches all the existing IDs from the database, generates a new UUID,
        and ensures the new ID does not conflict with existing ones.

        Returns:
        -------
        str
            A unique UUID string (without hyphens) that is not already in the database.
        """
        # SQL query to select all existing IDs from the `file_metadata` table
        query_id = f"SELECT id FROM intellitag_catalog.intellitag_dbx.file_metadata"
        with self.sql_connection.cursor() as cursor:
            cursor.execute(query_id)
            results = cursor.fetchall()
        ids_list=[row.id for row in results]

        # Keep generating a new UUID until a unique one is found
        while True:
            image_id = str(uuid.uuid1())
            if ids_list==None:
                return image_id.replace("-","") # Return the UUID with hyphens removed
            elif image_id not in ids_list:
                return image_id.replace("-","") # Return the UUID if it's unique


