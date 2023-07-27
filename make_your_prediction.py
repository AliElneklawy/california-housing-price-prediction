import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
import joblib
import numpy as np

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name():
    return ['ratio']

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        MinMaxScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
    MinMaxScaler()
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    MinMaxScaler()
)

preprocessing = ColumnTransformer([
    ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
    ('people_per_house', ratio_pipeline(), ['population', 'households']),
    ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
    ('log', log_pipeline, ['total_rooms', 'total_bedrooms', 'population', 'median_income', 'households']), 
    ('cat', cat_pipeline, make_column_selector(dtype_include='object')),
],
remainder=default_num_pipeline)


def get_input():

    user_inputs = {}
    user_inputs['longitude'] = float(input("Enter the longitude of the district: "))
    user_inputs['latitude'] = float(input("Enter the latitude of the district: "))
    user_inputs['housing_median_age'] = float(input("Enter the housing median age of the district: "))
    user_inputs['total_rooms'] = float(input("Enter the total number of rooms in the district: "))
    user_inputs['total_bedrooms'] = float(input("Enter the total number of bedrooms in the district: "))
    user_inputs['population'] = float(input("Enter the population of the district: "))
    user_inputs['households'] = float(input("Enter the number of households in the district: "))
    user_inputs['median_income'] = float(input("Enter the median income of the district: "))
    user_inputs['ocean_proximity'] = input("Enter the ocean proximity category: (NEAR BAY, INLAND, ISLAND, <1H OCEAN, NEAR OCEAN): ")

    user_data = pd.DataFrame([user_inputs])

    return user_data

reloaded_model = joblib.load("california_housing_model.pkl")
input = get_input()

print(f"A house in this district is predicted to be ${reloaded_model.predict(input)[0].round(2)}.")

    
