import os
from motor.motor_asyncio import AsyncIOMotorClient #pymongo driver to connect to databases.
from dotenv import load_dotenv

load_dotenv()

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = "document_automation"

# Singleton client
_client: AsyncIOMotorClient = None 


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None: # if _client is not set then it will get connected using the AsyncIOMotorClient
        _client = AsyncIOMotorClient(MONGODB_CONNECTION_STRING)
    return _client


def get_db():
    return get_client()[DATABASE_NAME]


async def close_client():
    global _client
    if _client: # if _client is set then the client connection is closed
        _client.close()
        _client = None