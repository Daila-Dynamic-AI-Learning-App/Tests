import os
import logging
import json
import pymongo
import keys
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from pymongo import MongoClient

# Set OpenAI API key
# This is required for the language model to function
os.environ["OPENAI_API_KEY"] = keys.OPEN_API_KEY

# Connect to MongoDB
# We are using MongoDB as our database. The connection string is constructed from the keys module.
mongo_client = MongoClient(f"mongodb+srv://{keys.DB_USER}:{keys.DB_SECRET}@cluster0.pho2egn.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client[keys.DB_NAME]


def get_mongodb_schema():
    # Get a list of collections in the database
    # This is used to understand the structure of the database
    collections = db.list_collection_names()

    # Create an empty dictionary to store the schema
    # The schema will be a mapping of collection names to their fields
    schema = {}

    # Iterate over each collection
    for collection in collections:
        # Get the collection object
        # This allows us to interact with the collection
        coll = db[collection]

        # Get a sample document from the collection
        # We use this to understand the fields in the collection
        document = coll.find_one()

        # Get the keys (fields) of the document
        # These will be added to the schema
        fields = list(document.keys())

        # Add the collection's fields to the schema
        # This allows us to understand the structure of the collection
        schema[collection] = fields

    # Return the schema
    # This will be used by other functions to generate queries
    return schema

# Tool decorator for performing MongoDB queries
# This function is decorated with the @tool decorator, which means it can be used as a tool in the agent
@tool
def insert_data(input_str: str) -> str:
    """
        performs a mongodb insert operation based on the input_str given

        Args:
            input_str: the input string to generate a prompt for
            collection_name: the name of the collection to query

        Returns:
            the result of the query as a string
    """
    # Get the schema of the database
    # This is used to generate the query
    db_schema = get_mongodb_schema()

    # If we couldn't get the schema, return an error
    if db_schema is None:
        return "Error: Unable to retrieve the database schema."

    # Convert the schema to a string
    # This is required for the language model to generate the query
    schema_str = json.dumps(db_schema, indent=2)
    # Define the template for the prompt
    # This is used by the language model to generate the query
    template = """Given this Mongodb database schema:{schema} \
        Return a MongoDB query to insert data from the database based on this prompt: {input}. \
        and the collection to make the change into. \
        You are only allowed to return the query in MongoDB syntax and the collection name based on the schema provided, nothing else. \
        using the format below: \

        collection_name: <collection-from-the-schema-provided> \
        query: <mongodb-syntax-query-to-pass-into-insert-function> \
        """
    
    # Create a PromptTemplate for generating MongoDB queries
    # This is used by the language model to generate the query
    prompt = PromptTemplate(template=template, input_variables=["schema", "input"])

    # Create a chat model using the GPT-like language model
    # This is used to generate the query
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate MongoDB query using the prompt and user input
    # The query is a string that can be executed by MongoDB
    query = llm_chain.predict(schema=schema_str, input=input_str)

    # Print the query for debugging purposes
    print(query)
    # logging.info(query)
    # collection = db[]
    
    # try:
    #     # Execute the MongoDB query for the given collection
    #     # This will insert the data into the database
    #     result = collection.insert_one(query)
        
    #     # Returns the id of the inserted document
    #     # This can be used to reference the document later
    #     return str(result.inserted_id)

    # # Handle MongoDB errors:
    # # If there is a problem with the query, this will catch it
    # except pymongo.errors.PyMongoError as error:
    #     logging.exception("MongoDB error")
    #     return f"Error: {error}"

    # # Handle other errors:
    # # This will catch any other errors that might occur
    # except Exception as e:
    #     return f"Error: {e}"
    
# Tool decorator for extracting data from the database using SQL
# This function is decorated with the @tool decorator, which means it can be used as a tool in the agent
@tool
def extract_data(input_str: str, ) -> str:
    """
        performs a mongodb find operation based on the input_str given

        Args:
            input_str: the input string to generate a prompt for

        Returns:
            the result of the find query as a string
    """
    # Get the schema of the database
    # This is used to generate the query
    db_schema = get_mongodb_schema()
    if db_schema is None:
        return "Error: Unable to retrieve the database schema."

    # Convert the schema to a string
    # This is required for the language model to generate the query
    schema_str = json.dumps(db_schema, indent=2)


    # Define the template for the prompt
    # This is used by the language model to generate the query
    template = """Given this Mongodb database schema:{schema} \
        Return a MongoDB query to retrieve data from the database based on this prompt: {input}. \
        You are only allowed to return the query in MongoDB syntax, nothing else."""
    
    # Create a PromptTemplate for generating SQL queries
    # This is used by the language model to generate the query
    prompt = PromptTemplate(template=template, input_variables=["schema", "input"])

    # Create a chat model using the GPT-like language model
    # This is used to generate the query
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate MongoDb query using the prompt and user input
    # The query is a string that can be executed by MongoDB
    query = llm_chain.predict(schema=schema_str, input=input_str)

    # logging.info(query)
    # collection = db[collection_name]
    
    # try:
    #     # Execute the MongoDB query for the given collection
    #     # This will retrieve the data from the database
    #     result = collection.find(query)
        
    #     # Returns the object found in the collection as a string
    #     # This can be used to understand the data in the database
    #     return str(result)

    # # Handle MongoDB errors:
    # # If there is a problem with the query, this will catch it
    # except pymongo.errors.PyMongoError as error:
    #     logging.exception("MongoDB error")
    #     return f"Error: {error}"

    # # Handle other errors:
    # # This will catch any other errors that might occur
    # except Exception as e:
    #     return f"Error: {e}"


# Function to set up an agent for MongoDB interactions
# This function initializes an agent that can interact with MongoDB
def AgentMongoDB(input: str) -> None:
    """
        initializes an agent for mongodb interactions

        Args:
            input: the input string to generate a prompt for
    """
    # Define the tools for performing MongoDB queries
    # These tools are used by the agent to interact with the database
    mongo_query_tool = Tool(
        name="MongoDB find query",
        func=extract_data,
        description="Useful for MongoDB find operations",
    )

    mongo_in_tool = Tool(
        name="MongoDB insert query",
        func=insert_data,
        description="Useful for MongoDB insert operations",
    )

    # available custom tools
    # These are the tools that the agent can use
    tools = [mongo_query_tool, mongo_in_tool]

    # Create a chat model
    # This is used by the agent to generate queries
    chat_model = ChatOpenAI(temperature=0, model_name="gpt-4")
    
    # Initialize an agent with tools, chat model, and type
    # The agent can now interact with the database
    agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    # Run the agent on the provided input
    # This will generate a query and execute it
    print(agent.run(input))

# print(get_mongodb_schema())
# Run the agent with the given input as a string
# This is an example of how to use the agent
AgentMongoDB("add a new user with name 'Tommy' and password 'blah' with username of 'kpk'")