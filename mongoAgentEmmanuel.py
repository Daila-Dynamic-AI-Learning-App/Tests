import pymongo
from pymongo import MongoClient
import datetime
from bson import ObjectId
from langchain.tools import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import os


os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(temperature=0, model_name="gpt-4")

DB_USER = "tommydebisi"
DB_SECRET = "fd73QuSzzWrQUAPE"
database_name = "daila"
# Replace these with your MongoDB connection details
client = MongoClient(f"mongodb+srv://{DB_USER}:{DB_SECRET}@cluster0.pho2egn.mongodb.net/?retryWrites=true&w=majority")

def retrieve_schema(database_name):
   
    # Connect to the specified database
    db = client[database_name]
    
    # Get the list of collection names in the database
    collection_names = db.list_collection_names()
    
    # Initialize a dictionary to store the schemas for all collections
    all_schemas = {}
    
    # Iterate through each collection to build the schema
    for collection_name in collection_names:
        collection = db[collection_name]
        documents = collection.find({})
        
        # Initialize an empty schema dictionary for the current collection
        schema = {}
        
        for doc in documents:
            for key in doc:
                if key not in schema:
                    schema[key] = type(doc[key]).__name__
        
        # Store the schema for the current collection
        all_schemas[collection_name] = schema
    

    return all_schemas



from pymongo import MongoClient

def perform_insert(insert_data, collection_name):
    """
    Inserts a document into a specified MongoDB collection.

    Parameters:
    - insert_data (dict): The data to be inserted as a dictionary.
    - collection_name (str): The name of the collection to insert the data into.

    Returns:
    str: A message indicating the success or failure of the insertion operation along with the inserted document ID.
    """
    # Hardcoded MongoDB connection string
    connection_string = f"mongodb+srv://{DB_USER}:{DB_SECRET}@cluster0.pho2egn.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(connection_string)

    try:
        # Connect to the specified database
        db = client[database_name]

        # Get the specified collection
        collection = db[collection_name]

        # Perform the insert operation
        insertion_result = collection.insert_one(insert_data)
        inserted_id = insertion_result.inserted_id



        return f"Insertion successful. Inserted document ID: {inserted_id}"
    except Exception as e:
        return f"Insertion failed. Error: {str(e)}"


def perform_extraction(query, collection_name):
    """
    Extracts data from a specified MongoDB collection based on a given query.

    Parameters:
    - query (dict): The query to filter the data to be extracted.
    - collection_name (str): The name of the collection to extract data from.

    Returns:
    list: A list of documents extracted from the collection based on the query.
    """
    try:
        # Connect to the specified database
        db = client[database_name]

        # Get the specified collection
        collection = db[collection_name]

        # Perform the extraction operation
        extracted_data = collection.find(query)
        
        # Process and use the extracted data as needed
        result = [doc for doc in extracted_data]



        return result
    except Exception as e:
        return f"Extraction failed. Error: {str(e)}"


# Example extraction query (replace with your actual query)
query = {"assessment": "Example Assessment"}
collection_name = "assessment"

# Call the function to perform the extraction operation
extracted_data = perform_extraction(query, collection_name)

# Print the extracted data
print(extracted_data)


tool_insert = StructuredTool.from_function(perform_insert)
tool_extract = StructuredTool.from_function(perform_extraction)

# Initialize the agent

# Structured tools are compatible with the STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION agent type.
agent_executor = initialize_agent(
    [tool_insert, tool_extract],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

schema = retrieve_schema(database_name)

print(agent_executor.run(f"Given this database schema {schema} perform the following query to the nonSQL database.  What assessments are there?"))

# Close mongo connection
client.close()