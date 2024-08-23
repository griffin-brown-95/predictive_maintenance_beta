import yaml
from modeling import build_pipeline, train_model, save_model
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

    # Build pipeline and train model
    model = build_pipeline(config)
    model = train_model(model, X, y)

    # Save trained model
    save_model(model, config['output']['model_file'])

if __name__ == "__main__":
    main()
