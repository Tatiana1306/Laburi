import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def extract_files(zip_paths, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    for file_path in zip_paths:
        if os.path.exists(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"Warning: {file_path} not found.")


def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep=';')
    else:
        print(f"Error: {file_path} not found.")
        return None


def handle_missing_values(df):
    return df.dropna()


def remove_outliers(df, numerical_cols):
    q1 = df[numerical_cols].quantile(0.25)
    q3 = df[numerical_cols].quantile(0.75)
    iqr = q3 - q1
    return df[~((df[numerical_cols] < (q1 - 1.5 * iqr)) | (df[numerical_cols] > (q3 + 1.5 * iqr))).any(axis=1)]


def preprocess_data(df):
    df = handle_missing_values(df)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df = remove_outliers(df, numerical_cols)

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoders


def train_models(x_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, x_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {"accuracy": acc, "report": report}
    return results


def main():
    extract_dir = "./extracted_files"
    zip_paths = ["bank.zip", "bank-additional.zip"]
    extract_files(zip_paths, extract_dir)

    data_path = os.path.join(extract_dir, "bank-full.csv")
    df = load_data(data_path)

    if df is not None:
        print("Initial Data Preview:")
        print(df.head())

        df, _ = preprocess_data(df)

        print("Processed Data Preview:")
        print(df.head())

        if "y" in df.columns:
            x = df.drop(columns=["y"])
            y = df["y"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            models = train_models(x_train, y_train)

            results = evaluate_models(models, x_test, y_test)

            print("Model Performance:")
            for name, res in results.items():
                print(f"{name}: Accuracy = {res['accuracy']:.2f}")
                print(classification_report(y_test, models[name].predict(x_test)))
        else:
            print("Error: Target variable 'y' not found in dataset.")
    else:
        print("Failed to load dataset.")


if __name__ == "__main__":
    main()
