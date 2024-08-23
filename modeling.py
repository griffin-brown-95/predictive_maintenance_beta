from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import joblib
import pandas as pd

def preprocess_dataframe(data):
    object_cols = data.select_dtypes(include='object').columns.to_list()
    for col in object_cols:
        data[col] = data[col].astype('category')

    category_cols = [col for col in data.columns if data[col].dtype == 'category']

    for col in category_cols:
        dummies = pd.get_dummies(data[col], drop_first=True, dummy_na=True, prefix=col)
        data = pd.concat([data.drop(col, axis=1), dummies], axis=1)

    return data

def feature_engineering(data):
    X = X.copy()  # To avoid modifying the original DataFrame
    
    # Torque to Speed Ratio
    X['tts_ratio'] = X['Torque [Nm]'] / X['Rotational speed [rpm]']
    
    # Temperature and Tool Wear Interaction
    X['temp_wear_interaction'] = X['Process temperature [K]'] * X['Tool wear [min]']
    
    return X

def build_pipeline(config):
    # Create the FunctionTransformer for the preprocessing step
    preprocessor = FunctionTransformer(preprocess_dataframe, validate=False)

    feature_eng = FunctionTransformer(feature_engineering, validate=False)

    # Create the pipeline with preprocessing, imputation, scaling, and logistic regression
    imputer = SimpleImputer(missing_values=pd.NA, strategy=config['model']['imputer_strategy'])
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=500)

    # Build and return the pipeline
    return make_pipeline(preprocessor, feature_eng, imputer, scaler, lr)

def train_model(model, X, y):
    model.fit(X, y)
    return model

def save_model(model, file_path):
    joblib.dump(model, file_path)
