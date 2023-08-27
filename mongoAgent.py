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
os.environ["OPENAI_API_KEY"] = keys.OPEN_API_KEY

# Connect to MongoDB
mongo_client = MongoClient(f"mongodb+srv://{keys.DB_USER}:{keys.DB_SECRET}@cluster0.pho2egn.mongodb.net/?retryWrites=true&w=majority")
db = mongo_client[keys.DB_NAME]


def get_mongodb_schema():
    # Get a list of collections in the database
    collections = db.list_collection_names()

    # Create an empty dictionary to store the schema
    schema = {}

    # Iterate over each collection
    for collection in collections:
        # Get the collection object
        coll = db[collection]

        # Get a sample document from the collection
        document = coll.find_one()

        # Get the keys (fields) of the document
        fields = list(document.keys())

        # Add the collection's fields to the schema
        schema[collection] = fields

    # Return the schema
    return schema

# Tool decorator for performing MongoDB queries
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
    db_schema = get_mongodb_schema()

    if db_schema is None:
        return "Error: Unable to retrieve the database schema."

    schema_str = json.dumps(db_schema, indent=2)
    template = """Given this Mongodb database schema:{schema} \
        Return a MongoDB query to insert data from the database based on this prompt: {input}. \
        and the collection to make the change into. \
        You are only allowed to return the query in MongoDB syntax and the collection name based on the schema provided, nothing else. \
        using the format below: \

        collection_name: <collection-from-the-schema-provided> \
        query: <mongodb-syntax-query-to-pass-into-insert-function> \
        """
    
    # Create a PromptTemplate for generating MongoDB queries
    prompt = PromptTemplate(template=template, input_variables=["schema", "input"])

    # Create a chat model using the GPT-like language model
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate MongoDB query using the prompt and user input
    query = llm_chain.predict(schema=schema_str, input=input_str)

    print(query)
    # logging.info(query)
    # collection = db[]
    
    # try:
    #     # Execute the MongoDB query for the given collection
    #     result = collection.insert_one(query)
        
    #     # Returns the id of the inserted document
    #     return str(result.inserted_id)

    # # Handle MongoDB errors:
    # except pymongo.errors.PyMongoError as error:
    #     logging.exception("MongoDB error")
    #     return f"Error: {error}"

    # # Handle other errors:
    # except Exception as e:
    #     return f"Error: {e}"
    
# Tool decorator for extracting data from the database using SQL
@tool
def extract_data(input_str: str, ) -> str:
    """
        performs a mongodb find operation based on the input_str given

        Args:
            input_str: the input string to generate a prompt for

        Returns:
            the result of the find query as a string
    """
    # Prompt template for SQL query generation
    db_schema = get_mongodb_schema()
    if db_schema is None:
        return "Error: Unable to retrieve the database schema."

    # Convert the schema to a string
    schema_str = json.dumps(db_schema, indent=2)


    template = """Given this Mongodb database schema:{schema} \
        Return a MongoDB query to retrieve data from the database based on this prompt: {input}. \
        You are only allowed to return the query in MongoDB syntax, nothing else."""
    
    # Create a PromptTemplate for generating SQL queries
    prompt = PromptTemplate(template=template, input_variables=["schema", "input"])

    # Create a chat model using the GPT-like language model
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate MongoDb query using the prompt and user input
    query = llm_chain.predict(schema=schema_str, input=input_str)

    # logging.info(query)
    # collection = db[collection_name]
    
    # try:
    #     # Execute the MongoDB query for the given collection
    #     result = collection.find(query)
        
    #     # Returns the object found in the collection as a string
    #     return str(result)

    # # Handle MongoDB errors:
    # except pymongo.errors.PyMongoError as error:
    #     logging.exception("MongoDB error")
    #     return f"Error: {error}"

    # # Handle other errors:
    # except Exception as e:
    #     return f"Error: {e}"


# Function to set up an agent for MongoDB interactions
def AgentMongoDB(input: str) -> None:
    """
        initializes an agent for mongodb interactions

        Args:
            input: the input string to generate a prompt for
    """
    # Define the tools for performing MongoDB queries
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
    tools = [mongo_query_tool, mongo_in_tool]

    # Create a chat model
    chat_model = ChatOpenAI(temperature=0, model_name="gpt-4")
    
    # Initialize an agent with tools, chat model, and type
    agent = initialize_agent(tools, chat_model, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    # Run the agent on the provided input
    print(agent.run(input))

# print(get_mongodb_schema())
# Run the agent with the given input as a string
AgentMongoDB("add a new user with name 'Tommy' and password 'blah' with username of 'kpk'")