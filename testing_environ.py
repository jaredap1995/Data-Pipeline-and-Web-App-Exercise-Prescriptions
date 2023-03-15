# importing os module 
import os
import pprint
from config import key
import json
  
# Get the list of user's
# environment variables
env_var = os.environ

# Print the list of user's
# environment variables
print("User's Environment variable:")


json_data = json.loads(key)

print(json_data)