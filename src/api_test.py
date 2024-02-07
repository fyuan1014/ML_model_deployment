import requests 
import json 
import pandas as pd 

# Load test data 
df=pd.read_csv('../data/cell2cellholdout.csv') 
# Define the API endpoint 
url = 'http://127.0.0.1:1313/predict' 
# Iterate over rows in the DataFrame 
for index, row in df.tail(5).iterrows(): 
    # Convert the row to a dictionary 
    data_dict = row.to_dict() 
    # Convert the dictionary to a JSON string 
    json_data = json.dumps(data_dict) 
    #print(json_data)
    # Set the request headers 
    headers = {'Content-Type': 'application/json'} 
    # Make the API call 
    response = requests.post(url, headers=headers, data=json_data) 
    # Check the response 
    if response.status_code == 200: 
        result = response.json()
        print(result)
    else:
        print(f"Error: {response.status_code}\n{response.text}")
