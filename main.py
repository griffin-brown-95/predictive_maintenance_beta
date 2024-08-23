import yaml
from modeling import build_pipeline, evaluate_model, save_model, split_data
import pandas as pd

def load_data(file_path, drop_columns):
    df = pd.read_csv(file_path)
    df.drop(drop_columns, axis=1, inplace=True)
    return df

def main():
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load data
    df = load_data(config['data']['input_file'], config['data']['drop_columns'])

    # Separate target and features
    y = df[config['data']['target_column']]
    X = df.drop([config['data']['target_column']], axis=1)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    # Build and train model with integrated preprocessing and scaling
    model = build_pipeline(config)
    model.fit(X_train, y_train)

    # Evaluate model
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"Training Accuracy: {train_score}")
    print(f"Testing Accuracy: {test_score}")

    # Save the trained model
    save_model(model, config['output']['model_file'])

if __name__ == "__main__":
    main()
