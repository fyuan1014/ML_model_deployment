# Scoring app with API
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

class FeatureEngineering:
    def __init__(self, raw_data):
        self.data_origin = raw_data.copy(deep=True)
        self.catVarList = ['ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner','UniqueSubs','ActiveSubs']
        self.numVarList = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls', 'PercChangeRevenues', 'DroppedCalls']
        self.data_origin['UniqueSubs'] = self.data_origin['UniqueSubs'].astype(str)
        self.data_origin['ActiveSubs'] = self.data_origin['ActiveSubs'].astype(str)
        self.data = self.data_origin[self.catVarList+self.numVarList]
    
    def mean_imputation_and_scaling(self,imputer,std_scaler):
        
        columns_to_drop = self.catVarList
        
        imputed_data = pd.DataFrame(imputer.transform(self.data[self.numVarList]), columns=self.numVarList)
        scaled_data = pd.DataFrame(std_scaler.transform(imputed_data), columns=imputed_data.columns)
        
        self.data = pd.concat([scaled_data, self.data[columns_to_drop]], axis=1, sort=False)

    def create_dummies(self, selected_features):

        self.data = pd.get_dummies(self.data, columns=self.catVarList, prefix=self.catVarList)
        for col in selected_features: 
            if col not in self.data.columns: 
                self.data[col] = 0

        return self.data

def predict_single_row(model, features, data):
    X = data[features]
    predicted_proba = model.predict(X)[0]
    predicted_class = 1 if predicted_proba >= 0.5 else 0
    
    # Convert features to a dictionary with serializable values
    features_dict = {str(feature): str(X.iloc[0][feature]) for feature in X.columns}
    # Sort feature names alphabetically
    sorted_feature_names = sorted(features_dict.keys())
    result = {
        'business_outcome': predicted_class,
        'phat': float(predicted_proba),
        'model_inputs': {feature: features_dict[feature] for feature in sorted_feature_names}
    }
    return result

def predict_batch(model, features, data):
    predictions = []
    for _, row in data.iterrows():
        prediction = predict_single_row(model, features, row)
        predictions.append(prediction)
    return predictions

combined_filepath = '../model/combined_model_and_features.pkl'
loaded_model_and_features = joblib.load(combined_filepath)
imputer = loaded_model_and_features['imputer']
scaler = loaded_model_and_features['scaler']
xgb_model = loaded_model_and_features['xgb_model']
selected_features = loaded_model_and_features['selected_features']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert data to DataFrame
        if isinstance(data, list):
            data = pd.DataFrame(data)
        else:
            data = pd.DataFrame([data])

        # 1. Preprocess data
        feature_engineering = FeatureEngineering(data)
        feature_engineering.mean_imputation_and_scaling(imputer=imputer,std_scaler=scaler)
        preprocessed_data = feature_engineering.create_dummies(selected_features)

        # 2. Make predictions
        if len(preprocessed_data) == 1:
            predictions = predict_single_row(xgb_model, selected_features, preprocessed_data)
        else:
            predictions = predict_batch(xgb_model, selected_features, preprocessed_data)

        return jsonify(predictions)

    except Exception as e:
        print('Exception during prediction: ' + str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True,port=1313)
