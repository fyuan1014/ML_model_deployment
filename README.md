# Machine learning model deployment
This repository shows how you can deploy your trained machine learning model for production as an API with flask, a popular Python web framework. Here is an overview of this repository.

## Codes
All the codes are included into the src folder: 
1. **train.py**: this the code where you can use and adjust to train any machine learning models; 
2. **main.py**: this is the main function to build the score app as an API with flask, which can handle both both single or batch calls. This code uses the port 1313; 
3. **api_test.py**: this code shows an example how you can use the API with Python.

## Data and Model
The folders of data and model are the locations for storing data and trained model (after running the train.py). For simplication, this model training only used several features from the cell churn dataset. All the codes use the open cell churn data from Kaggle (https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom).

## Requirements
Requirements.txt file includes the main Python libraries to be used.

## Command for Running the API app
In a CMD or Powershell terminal, use this command: python main.py to start the API, when you see prints such as "* Running on http://127.0.0.1:1313" from the 1313 port (you may use different ports and see your own port number), you are ready to run the api_test.py, where I just used the tail from the test data for testing.

## Next Step
1. Deployment: With the codes all ready as in src, the next step for real production is to create the Dockerfile to containerize the codes, dependency and requirements, which you can use to create the Docker image and further deploy it as container or run the container with Kubernetes.
2. Model and data monitoring: As model is deployed for production, data and model metrics need to be created for monitoring drifts. 
