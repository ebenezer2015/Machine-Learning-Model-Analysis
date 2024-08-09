import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import skew

def read_csv(csv_file: str):
    return pd.read_csv(csv_file)

def change_column_type(df: pd.DataFrame, column: str, type: str):
    df.loc[:, column] = df[column].astype(type)
    return df

def checking_class_imbalance(df: pd.DataFrame, class_feature: str, title: str):
    count_classes = df[class_feature].value_counts()
    percentage = (count_classes / count_classes.sum()) * 100
    
    # Define color palette
    colors = ['steelblue', 'firebrick']
    
    ax = count_classes.plot(kind='bar', rot=0, color=colors)
    ax.set_title(title)
    ax.set_xlabel(class_feature)
    ax.set_ylabel("Frequency")
    
    # Add percentage labels to each bar
    for i, v in enumerate(count_classes):
        ax.text(i, v, f"{v} ({percentage[i]:.1f}%)", ha='center', va='bottom')
    
    return plt.show()

def treat_skewed_columns(df):
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
        display(df.info())
        display(df.head())
        display(df.describe())
        missing_columns = df.columns[df.isnull().any()].tolist()
        if len(missing_columns) > 0:
            print("Columns with missing data:")
            for col in missing_columns:
                missing_rows = df[col].isnull().sum()
                print(f"{col}: {missing_rows} missing rows")
        else:
            print("No columns have missing data.")


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
        sns.histplot(data=df, x=column, kde=True, ax=axes[i], bins=int(700/12), color = 'darkblue')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Density Plot with Histogram of {column}')
    
    # Remove any unused subplots
    if num_features < len(axes):
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()  # Adjust spacing between subplots
    return plt.show()

def replace_missing_data(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    # Create a copy of the DataFrame
    filled_df = df.copy()

    # Iterate over each column in the DataFrame
    for col in filled_df.columns:
        # Check if the column contains numeric data
        if filled_df[col].dtype in [np.float64, np.int64]:
            if method == 'mean':
                filled_df[col].fillna(filled_df[col].mean(), inplace=True)
            elif method == 'median':
                filled_df[col].fillna(filled_df[col].median(), inplace=True)
        else:
            filled_df[col].fillna(filled_df[col].mode()[0], inplace=True)

    return filled_df

def drop_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unwanted columns from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.  
    Returns:
        pd.DataFrame: DataFrame with unwanted columns dropped.
    """
    df_dropped = df.dropna(subset=df.columns)
    
    return df_dropped


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Encode categorical features using one-hot encoding
    encoded_df = pd.get_dummies(df, drop_first=True)
    
    return encoded_df

def scale_numerical_features(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    # Scale numerical features using StandardScaler
    scaled_df = df.copy()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Exclude target_feature from scaling
    if target_feature in numerical_cols:
        numerical_cols = numerical_cols.drop(target_feature)

    for col in numerical_cols:
        scaler = StandardScaler()
        scaled_df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return scaled_df

def train_models(trained_df: pd.DataFrame, target_feature: str, models: list, test_size: float = 0.2, cv: int = 10) -> pd.DataFrame:
    """
    Trains multiple machine learning models using cross-validation, returns the model results, feature importance
    using SHAP, and confusion matrix in a DataFrame.

    Args:
        trained_df (pd.DataFrame): Pre-split training dataset.
        target_feature (str): Name of the target feature column.
        models (dict): Dictionary containing model names as keys and corresponding model instances as values.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
        cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with model names, evaluation metrics (accuracy, precision, recall, f1-score),
                      feature importance/explanations, and confusion matrix.
    """
    X = trained_df.drop(target_feature, axis=1)
    y = trained_df[target_feature]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results = []
    best_accuracy = 0.0
    best_model_idx = 0

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
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        #fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Append the results to the list
        results.append([model_name, scores.mean(), scores.std(), accuracy, precision, recall, f1, cm])
        
        # Check if current model has better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_idx = idx

    # Create a DataFrame with the results
    columns = ['Model', 'Mean CV Accuracy', 'CV Accuracy Std', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Confusion Matrix']
    results_df = pd.DataFrame(results, columns=columns)
    
    # Add a column to indicate the best model based on accuracy
    results_df['Best Model'] = False
    results_df.loc[best_model_idx, 'Best Model'] = True

    return results_df

def plot_best_model_explanation(results_df, models, trained_df, target_feature):
    """
    Generates the SHAP summary plot for the best model based on the results DataFrame.

    Args:
        results_df (pd.DataFrame): DataFrame with model results.
        models (list): List of model instances.
        trained_df (pd.DataFrame): Pre-split training dataset.
        target_feature (str): Name of the target feature column.
    """
    best_model_idx = results_df[results_df['Best Model'] == True].index[0]
    best_model_name = results_df.loc[best_model_idx, 'Model']
    X_train = trained_df.drop(target_feature, axis=1)  # Assuming trained_df is the pre-split training dataset

    # Retrieve the best model
    best_model = models[best_model_idx]

    # Initialize the SHAP explainer
    if isinstance(best_model, DecisionTreeClassifier) or isinstance(best_model, RandomForestClassifier):
        background_samples = shap.sample(X_train, 10)  # Use shap.sample to summarize the background data
        explainer = shap.KernelExplainer(best_model.predict_proba, background_samples)
    elif isinstance(best_model, LogisticRegression) or isinstance(best_model, LinearDiscriminantAnalysis):
        explainer = shap.LinearExplainer(best_model, X_train)
    elif isinstance(best_model, SVC):
        explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
    else:
        raise ValueError(f"Unsupported model type: {best_model_name}")

    # Calculate SHAP values for the best model
    shap_values = explainer.shap_values(X_train)

    # Plot the SHAP summary plot for the best model
    shap.summary_plot(shap_values, X_train, plot_type='bar', class_names=best_model.classes_)








