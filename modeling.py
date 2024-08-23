from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
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

def build_pipeline(config):
    # Create the FunctionTransformer for the preprocessing step
    preprocessor = FunctionTransformer(preprocess_dataframe, validate=False)

    # Create the pipeline with preprocessing, scaling, and logistic regression
    imputer = SimpleImputer(missing_values=pd.NA, strategy=config['model']['imputer_strategy'])
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=500)

    # Build and return the pipeline
    return make_pipeline(preprocessor, imputer, scaler, lr)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

def save_model(model, file_path):
    joblib.dump(model, file_path)

def split_data(X, y, config):
    return train_test_split(
        X, y, 
        test_size=config['model']['test_size'], 
        stratify=y if config['model']['stratify'] else None, 
        random_state=config['model']['random_state']
    )
