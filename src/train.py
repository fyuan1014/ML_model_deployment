import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
import sys
import warnings
warnings.filterwarnings("ignore")

class FeatureEngineering:
    def __init__(self, raw_data):
        self.data_origin = raw_data.copy(deep=True)
        self.catVarList = ['ChildrenInHH', 'HandsetRefurbished', 'HandsetWebCapable', 'TruckOwner','UniqueSubs','ActiveSubs']
        self.numVarList = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls', 'PercChangeRevenues', 'DroppedCalls']
        self.data_origin['UniqueSubs'] = self.data_origin['UniqueSubs'].astype(str)
        self.data_origin['ActiveSubs'] = self.data_origin['ActiveSubs'].astype(str)
        self.data_origin['ChurnEdited'] =  np.where(self.data_origin['Churn']=='Yes',1,0)
        self.target = ['ChurnEdited']
        self.data = self.data_origin[self.catVarList+self.numVarList+self.target]

    def mean_imputation_and_scaling(self):
        columns_to_drop = self.catVarList + self.target
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_data = pd.DataFrame(imputer.fit_transform(self.data[self.numVarList]), columns=self.numVarList)

        std_scaler = StandardScaler()
        scaled_data = pd.DataFrame(std_scaler.fit_transform(imputed_data), columns=imputed_data.columns)
        
        self.data = pd.concat([scaled_data, self.data[columns_to_drop]], axis=1, sort=False)
        
        return self.data, imputer, std_scaler

    def create_dummies(self):
        self.data = pd.get_dummies(self.data, columns=self.catVarList, prefix=self.catVarList)
        return self.data

def train():
    try:
        raw=pd.read_csv('../data/cell2celltrain.csv')
        train = raw.copy(deep=True)

        # 1. Feature engineering
        feature_engineering_train = FeatureEngineering(train)
        data_imputer_scaler = feature_engineering_train.mean_imputation_and_scaling()
        train_imputed_std = feature_engineering_train.create_dummies()
        imputer = data_imputer_scaler[1]
        scaler = data_imputer_scaler[2]
        
        # 2. Fitting the base model
        xgbCalls_base = xgb.XGBClassifier()
        xgbCalls_base.fit(train_imputed_std.drop('ChurnEdited',axis=1),train_imputed_std['ChurnEdited'])
        print("Base model successfully trained")

        # 3. Feature selection and fitting the final model with selected features
        xgbCalls_select = SelectFromModel(xgbCalls_base, threshold=0)
        xgbCalls_select.fit(train_imputed_std.drop('ChurnEdited',axis=1),train_imputed_std['ChurnEdited'])
        selected_feature_indices = xgbCalls_select.get_support()
        selected_features = [feature for feature, selected in zip(train_imputed_std.drop('ChurnEdited',axis=1).columns, selected_feature_indices) if selected]

        xgbCalls_final = xgb.XGBClassifier()
        xgbCalls_final.fit(train_imputed_std[selected_features],train_imputed_std['ChurnEdited'])
        print("Final model successfully trained")

        # 3. Saving imputer, scaler, variable list, and trained model
        model_and_features = {
            'imputer':imputer,
            'scaler':scaler,
            'xgb_model': xgbCalls_final,
            'selected_features': selected_features
        }

        combined_filepath = '../model/combined_model_and_features.pkl'
        joblib.dump(model_and_features, combined_filepath)
        print('Model successfully saved')
    
    except Exception as e:
        print('Exception during training: ' + str(e))
        sys.exit(255)

if __name__ == '__main__':
    train()
    sys.exit(0)
