import uuid
from src.databricks_sql_connector import DatabrickSqlConnect
# from src.databricks_job_run import DatabrickSqlTable

import os
import streamlit as st


class Processing(DatabrickSqlConnect):

    def __init__(self):
        super().__init__()

    def sanitize_blob_name(self,blob_name):
            # Replace any problematic characters with underscores
            return blob_name.replace("\\", "/").replace("?", "_").replace("#", "_")

    def generate_unique_id(self):
        """Verify Image UUID """
        query_id = f"SELECT id FROM intellitag_catalog.intellitag_dbx.file_metadata"
        with self.sql_connection.cursor() as cursor:
            cursor.execute(query_id)
            results = cursor.fetchall()
        ids_list=[row.id for row in results]
        # print(ids_list)
        while True:
            image_id = str(uuid.uuid1())
            if ids_list==None:
                return image_id.replace("-","")
            elif image_id not in ids_list:
                return image_id.replace("-","")


