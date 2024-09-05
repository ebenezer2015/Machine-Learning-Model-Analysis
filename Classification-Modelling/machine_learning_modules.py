from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import (  # ,LabelEncoder, MinMaxScaler
    OneHotEncoder, StandardScaler)


def read_csv(csv_file: str):
    return pd.read_csv(csv_file)


def change_column_type(df: pd.DataFrame, column: str, type: str):
    df.loc[:, column] = df[column].astype(type)
    return df


def checking_class_dist(df: pd.DataFrame, class_feature: str, title: str):
    count_classes = df[class_feature].value_counts()
    percentage = (count_classes / count_classes.sum()) * 100
    colors = ['steelblue', 'firebrick']
    ax = count_classes.plot(kind='bar', rot=0, color=colors)
    ax.set_title(title)
    ax.set_xlabel(class_feature)
    ax.set_ylabel("Frequency")
    # Add percentage labels to each bar
    for i, v in enumerate(count_classes):
        ax.text(i, v, f"{v} ({percentage[i]:.1f}%)", ha='center', va='bottom')

    return plt.show()


def treat_skewed_columns(df) -> pd.DataFrame::
    # Get the list of numerical columns
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    # Find skewed columns
    skewed_columns = df[numerical_columns].apply(lambda x: skew(x)).abs().sort_values(ascending=False)
    skewed_columns = skewed_columns[skewed_columns > 0.5]  # Set the threshold for skewness as desired

    # Apply logarithmic transformation to skewed columns
    for col in skewed_columns.index:
        df[col] = np.log1p(df[col])

    return df


# Import other necessary libraries for the remaining techniques
def check_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = df.columns[df.isnull().any()].tolist()
    if len(missing_columns) > 0:
        print("Columns with missing data:")
        for col in missing_columns:
            missing_rows = df[col].isnull().sum()
            print(f"{col}: {missing_rows} missing rows")
    else:
        print("The data has no columns with missing data.")


def plot_numeric_features(df: pd.DataFrame):
    # Determine the number of features and calculate grid dimensions
    num_features = len(df.select_dtypes(include=['number']).columns)
    num_cols = 2
    num_rows = (num_features - 1) // num_cols + 1

    # Create subplots with grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    axes = axes.flatten()  # Flatten the array of axes

    # Plot density plot and histogram for each numerical feature
    for i, column in enumerate(df.select_dtypes(include='number')):
        sns.histplot(data=df, x=column, kde=True, ax=axes[i], bins=int(700/12), color='darkblue')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Density Plot with Histogram of {column}')

    # Remove any unused subplots
    if num_features < len(axes):
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust spacing between subplots
    return plt.show()


def treat_missing_data(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    Treats missing data in a DataFrame based on the specified method.
    If the method is not recognized, missing rows are dropped by default.

    Args:
    df (pd.DataFrame): The DataFrame with missing data.
    method (str): The method to use for filling missing data.
    Options are 'mean', 'median', 'mode', or 'drop'.

    Returns:
        pd.DataFrame: The DataFrame with treated missing data.
    """
    # Create a copy of the DataFrame
    filled_df = df.copy()

    # Default to dropping missing values if method is not recognized
    if method not in ['mean', 'median', 'mode']:
        print("Method not recognized. Dropping columns with missing values.")
        return filled_df.dropna(subset=df.columns)

    # Iterate over each column in the DataFrame
    for col in filled_df.columns:
        if filled_df[col].dtype in [np.float64, np.int64]:
            # Numeric data
            if method == 'mean':
                filled_df[col].fillna(filled_df[col].mean(), inplace=True)
            elif method == 'median':
                filled_df[col].fillna(filled_df[col].median(), inplace=True)
        else:
            # Categorical data
            if method == 'mode':
                filled_df[col].fillna(filled_df[col].mode()[0], inplace=True)

    return filled_df


def remove_unwanted_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Removes specified columns from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_remove (list): A list of column names to be removed.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns removed.
    """
    # Check if all specified columns exist in the DataFrame
    existing_columns = [col for col in columns_to_remove if col in df.columns]

    # Drop the unwanted columns
    df = df.drop(columns=existing_columns, axis=1)

    return df


def shuffle_and_split(data: pd.DataFrame, target_feature: str, test_size: float = 0.4, random_state: int = None,
                      columns_to_remove: List[str] = None):
    """
    Randomly shuffles a DataFrame and splits it into two samples.

    Args:
        df (pd.DataFrame): The input DataFrame to shuffle and split.
        test_size (float, optional): The proportion of the DataFrame to include in the test sample. Defaults to 0.4.
        random_state (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the two resulting samples.
    """
    data = remove_unwanted_columns(data, columns_to_remove)    
    train_sample, validation_sample = train_test_split(data, test_size=test_size, random_state=random_state,
                                                       stratify=data[target_feature])
                                     
    return train_sample, validation_sample


def handle_class_imbalance(X, y, technique: str = 'SMOTE', random_state: int = 42):
    """
    Handles class imbalance using the specified technique.

    Args:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Target series.
        technique (str): The technique for handling class imbalance ('undersampling', 'oversampling', 'SMOTE').
        random_state (int): The seed used by the random number generator.

    Returns:
        X_resampled (np.ndarray): Resampled features.
        y_resampled (np.ndarray): Resampled target.
    """
    if technique == 'SMOTE':
        sampler = SMOTE(random_state=random_state)
    elif technique == 'oversampling':
        sampler = RandomOverSampler(random_state=random_state)
    elif technique == 'undersampling':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("Invalid technique specified. Choose from 'undersampling', 'oversampling', or 'SMOTE'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


def prepare_data(df: pd.DataFrame, target_feature: str, test_size: float = 0.2, random_state: int = None,
                 balance_technique: str = 'SMOTE') -> Tuple:
    """
    Prepare the data for machine learning by handling class imbalance, encoding features, and splitting the dataset.

    Args:
        df (pd.DataFrame): The input dataframe containing features and the target variable.
        target_feature (str): The name of the target feature column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        balance_technique (str): Technique for handling class imbalance ('undersampling', 'oversampling', 'SMOTE').

    Returns:
        X_train_balanced (np.ndarray): Balanced and preprocessed training features.
        X_test (np.ndarray): Preprocessed test features.
        y_train_balanced (np.ndarray): Balanced training target values.
        y_test (np.ndarray): Test target values.
    """
    # Split the data into features and target
    X = df.drop(columns=[target_feature])
    y = df[target_feature]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state,
                                                        test_size=test_size, stratify=y)

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number', 'int64', 'float64']).columns.tolist()

    # Define column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Preprocess the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Handle class imbalance using the specified technique
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_transformed, y_train, 
                                                                technique=balance_technique,
                                                                random_state=random_state)
    return X_train_balanced, X_test_transformed, y_train_balanced, y_test


def train_models(train_sample: pd.DataFrame, target_feature: str, models: list, balance_technique: str, 
                 random_state: int = None, test_size: float = 0.2, cv: int = 10) -> pd.DataFrame:
    """
    Trains multiple machine learning models using cross-validation, returns the model results,
    and saves the best model to the working directory.

    Args:
        trained_df (pd.DataFrame): Pre-split training dataset.
        target_feature (str): Name of the target feature column.
        models (list): List of model instances.
        test_size (float, optional): The proportion of the dataset to include in the test split. 
        Default is 0.2.
        cv (int, optional): Number of cross-validation folds. Default is 10.

    Returns:
        pd.DataFrame: DataFrame with model names and evaluation metrics.
    """

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = prepare_data(train_sample, target_feature, test_size,
                                                    balance_technique, random_state)

    results = []
    best_accuracy = 0.0
    best_model = None
    best_model_name = ""

    for idx, model in enumerate(models):
        model_name = model.__class__.__name__
        # Perform cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        # Fit the model
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Append the results to the list
        results.append([model_name, scores.mean(), scores.std(), accuracy, precision, recall, f1, cm])

        # Check if current model has better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    # Save the best model
    if best_model is not None:
        joblib.dump(best_model, f"best_model_{best_model_name}.joblib")
        print(f"Best model {best_model_name} saved to best_model_{best_model_name}.joblib")

        # Calculate SHAP values for the best model
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_train)

        # Plot SHAP summary plot
        shap.summary_plot(shap_values, X_train, plot_type='bar')
        plt.title(f'SHAP Summary Plot for Best Model: {best_model_name}')
        plt.show()

    # Create a DataFrame with the results
    columns = ['Model', 'Mean CV Accuracy', 'CV Accuracy Std', 'Accuracy', 'Precision', 'Recall',
               'F1-Score', 'Confusion Matrix']
    results_df = pd.DataFrame(results, columns=columns)

    # Add a column to indicate the best model based on accuracy
    results_df['Best Model'] = False
    if best_model is not None:
        best_model_idx = results_df[results_df['Model'] == best_model_name].index[0]
        results_df.loc[best_model_idx, 'Best Model'] = True

    return results_df


def load_model(filename: str):
    """
    Loads a model from a file.

    Args:
        filename (str): File path to load the model from.

    Returns:
        The loaded model instance.
    """
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
