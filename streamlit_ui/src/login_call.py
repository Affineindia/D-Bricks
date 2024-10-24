from src.databricks_sql_connector import DatabrickSqlConnect
import streamlit as st


class UserData(DatabrickSqlConnect):
    """
    A class used to handle user login operations for the environment.

    Inherits:
    ---------
    DatabrickSQL: Inherits the properties and methods from the DatabrickSQL base class.

    Methods:
    --------
    login(email_id: str, password: str) -> tuple:
        Verifies the email_id and password against the user_table in the database.

    Parameters:
    -----------
    email_id : str
        The email ID of the user trying to log in.
    password : str
        The password associated with the given email ID.
    """

    def __init__(self):
        """Initializes the Login class and establishes a connection using the parent class."""
        super().__init__()

    def login(self, email_id: str, password: str) -> tuple:
        """
        Authenticates a user by checking their email ID and password in the database.

        Parameters:
        -----------
        email_id : str
            The email ID of the user trying to log in.
        password : str
            The password associated with the given email ID.

        Returns:
        --------
        tuple:
            A tuple containing a boolean and user access level. 
            If authentication is successful, returns (True, user_level_access).
            Otherwise, returns (False, None).
        """
        
        # Ensure that email_id and password are strings.
        assert isinstance(email_id, str), "email_id must be a string"
        assert isinstance(password, str), "password must be a string"
        
        # SQL query to check if the email_id and password exist in the user_table.
        __login_query = f"SELECT email_id,user_password,user_level FROM intellitag_catalog.intellitag_dbx.user_table WHERE email_id=='{email_id}' and user_password=='{password}'"

        # Execute the SQL query using the cursor from the SQL connection.
        with self.sql_connection.cursor() as cursor:
            cursor.execute(__login_query)
            __result = cursor.fetchall()  # Fetch all results from the query.

        # Close the SQL connection after executing the query.
        self.sql_connection.close()

        # Create lists of email IDs and passwords from the query results.
        __email_id_list = [row.email_id for row in __result]
        __user_password_list = [row.user_password for row in __result]

        # Check if the result is None, indicating no matching record found.
        if __result is None:
            return False, None
        
        # Check if the provided email_id and password are in the lists retrieved.
        elif email_id in __email_id_list and password in __user_password_list:
            # Get the user level access from the query results and return it.
            __user_level_access = [row.user_level for row in __result][0]
            return True, __user_level_access
        else:
            # If email or password does not match, return False and None.
            return False, None
    
    def check_user(self,email_id:str)-> str:
        """
        Checks if a given email ID exists in the `user_table` of the database.

        Parameters:
        ----------
        email_id : str
            The email ID of the user to check in the database.

        Returns:
        -------
        str
            Returns "Yes" if the email ID exists in the database, otherwise returns "NO".
        """
        # SQL query to select all email IDs from the user_table in the database
        check_query = f"SELECT email_id FROM intellitag_catalog.intellitag_dbx.user_table"

        # Execute the SQL query using the cursor from the SQL connection.
        with self.sql_connection.cursor() as cursor:
            cursor.execute(check_query)
            __results = cursor.fetchall()  # Fetch all results from the query.

        # Extract the list of email IDs from the query result
        result=[row.email_id for row in __results]
        print(result)

        # Check if the result is None or if the provided email_id is not in the result list
        if result==None:
            return "NO"
        elif email_id not in result:
            return "NO"
        else:
            return "Yes"

    def register(self,user_metadata):
        """
        Registers a new user in the database by inserting their metadata into the `user_table`.

        Parameters:
        ----------
        user_metadata : dict
            A dictionary containing the user's details, including:
            - 'Email ID' : str, email of the user
            - 'Company Name' : str, the company name of the user
            - 'Department Name' : str, the department the user belongs to
            - 'Access At' : str, timestamp representing when access was granted
            - 'User Level' : str, the access level of the user (e.g., admin, user)
            - 'User Password' : str, password for the user

        Returns:
        -------
        None
            Displays a success message if the user is registered successfully, or an error if the user already exists.
        """

        # Create the table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS intellitag_catalog.intellitag_dbx.user_table (
        email_id STRING,
        company_name STRING,
        department_name STRING,
        access_at TIMESTAMP,
        user_level STRING,
        user_password STRING,
        CONSTRAINT PM_KEY_1 PRIMARY KEY (email_id)
        )
        """
        # Create the table in the database (if it doesn't already exist)
        with self.sql_connection.cursor() as cursor:
            cursor.execute(create_table_query)
        
        # Check if the user is already registered using their email ID
        user_flag=self.check_user(user_metadata['Email ID'])

        # If the user is not registered, insert their data into the table
        if user_flag=="NO":
            query = f"""
            INSERT INTO intellitag_catalog.intellitag_dbx.user_table (email_id,company_name,department_name,access_at, user_level, user_password)
            VALUES ('{user_metadata['Email ID']}','{user_metadata['Company Name']}', '{user_metadata['Department Name']}', '{user_metadata['Access At']}','{user_metadata['User Level']}','{user_metadata['User Password']}')
            """
            # Execute the query to insert the user's data
            with self.sql_connection.cursor() as cursor:
                cursor.execute(query)

            st.success("User Registered Successfully.")

        else:
            st.error("User Already Registered. Please try again.")


if __name__=="__main__":
    login_obj=UserData()
    email="sanket.bodake@affine.ai"
    password=""
    login_flag,user_access_level=login_obj.login(email,password)
    print(" Login Flag ::",login_flag)
    print(" User Access Level ::",user_access_level)