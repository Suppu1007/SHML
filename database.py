import os
from pymongo import MongoClient

# --- Hardcoded Password (⚠️ INSECURE - REPLACE PLACEHOLDER IN PRODUCTION) ---
# Credentials are now hardcoded into variables as requested.
MONGO_USER = "SHML"
MONGO_PASSWORD = "smart" # <--- REPLACE THIS PLACEHOLDER WITH THE ACTUAL PASSWORD

# --- MongoDB Configuration ---
# Construct the URI using the hardcoded credentials
# mongodb+srv://SHML:smart@cluster0.a6xpenk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
MONGO_URI = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.a6xpenk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "SHML"
COLLECTION_NAME = "smart_1007"  

# --- Initialization ---
users_collection = None # Initialize to None in case of failure

try:
    client = MongoClient(MONGO_URI)
    # The 'ping' command forces an authentication check to verify the connection
    client.admin.command('ping') 
    db = client[DATABASE_NAME]
    users_collection = db[COLLECTION_NAME]
    print("✅ Successfully connected to MongoDB.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not connect to MongoDB. Details: {e}")

# --- Database Operations ---
def get_user_by_email(email):
    """Retrieve a user by email."""
    # Check if the collection object was successfully initialized
    if users_collection is not None:
        return users_collection.find_one({'email': email})
    return None # Return None if database connection failed
    
def create_user(email, hashed_password):
    """Create a new user."""
    if users_collection is not None:
        # Note: In a real MongoDB setup, you might also store profile data here
        user_data = {'email': email, 'password': hashed_password}
        return users_collection.insert_one(user_data)
    return None
