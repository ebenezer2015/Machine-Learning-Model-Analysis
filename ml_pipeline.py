import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap


class MLDataPipeline:
    def __init__(self, df):
        self.data = df.copy()
        
    def check_missing_data(self):
        display(self.data.info())
        display(self.data.head())
        display(self.data.describe())
        missing_columns = self.data.columns[self.data.isnull().any()].tolist()
        if len(missing_columns) > 0:
            print("Columns with missing data:")
            for col in missing_columns:
                missing_rows = self.data[col].isnull().sum()
                print(f"{col}: {missing_rows} missing rows")
        else:
            print("No columns have missing data.")

    def encode_categorical_features(self, features):
        categorical_features = features.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            label_encoder = LabelEncoder()
            features[feature] = label_encoder.fit_transform(features[feature])

        return features

    def scale_numerical_features(self, features):
        numerical_features = features.select_dtypes(include=['float64', 'int64']).columns
        for feature in numerical_features:
            features[feature] = (features[feature] - features[feature].mean()) / features[feature].std()

        return features
    
    def treat_outliers(self):
        # Treat outliers using Interquartile Range (IQR) method
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        for feature in numerical_features:
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.data.loc[self.data[feature] < lower_bound, feature] = lower_bound
            self.data.loc[self.data[feature] > upper_bound, feature] = upper_bound

    def run_multiple_ml_techniques(self, target_feature, techniques, num_folds=10, test_size=0.2):
        
        # Perform data quality checks
        self.data_quality_checks()

        # Define the features and target variable
        X = self.data.drop(target_feature, axis=1)
        y = self.data[target_feature]

        # Apply data manipulation steps to the feature variables only
        X_encoded = self.encode_categorical_features(X)
        #X_treated = self.treat_outliers(X_encoded)
        X_scaled = self.scale_numerical_features(X_encoded)
        

        # Create an empty DataFrame to store the model results
        model_results = pd.DataFrame(columns=['Technique', 'Accuracy', 'Precision', 'Recall', 'Confusion Matrix'])

        # Loop through each technique
        for technique_name, model in techniques:
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

            # Perform cross-validation
            y_train_pred = cross_val_predict(model, X_train, y_train, cv=num_folds)

            # Train the model on the entire training set
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_test_pred = model.predict(X_test)

            # Calculate evaluation metrics (accuracy, precision, recall, etc.)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)

            # Calculate confusion matrix
            cm = confusion_matrix(y_train, y_train_pred)
            
            # Model explanation using SHAP
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train)

            # Append the model results to the DataFrame
            model_results = model_results.append({'Technique': technique_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Confusion Matrix': cm}, ignore_index=True)
            
            # Print model explanation summary
            print("Model Explanation for", technique_name)
            shap.summary_plot(shap_values, X_train, plot_type='bar')

        # Find the best model based on the evaluation metric
        best_model = model_results.sort_values('Accuracy', ascending=False).iloc[0]

        # Print the best model
        print("Best Model:")
        print(best_model)

        return model_results

    def data_quality_checks(self):
        num_missing = self.data.isnull().sum().sum()
        num_outliers = self.check_outliers()
        num_numeric_features, num_categorical_features = self.get_feature_counts()

        print("Data Quality Checks:")
        print("Number of missing values:", num_missing)
        print("Number of outliers:", num_outliers)
        print("Number of numeric features:", num_numeric_features)
        print("Number of categorical features:", num_categorical_features)

    def check_outliers(self):
        # Perform outlier detection or any other data quality checks
        # Return the number of outliers found
        return 0

    def get_feature_counts(self):
        num_numeric_features = len(self.data.select_dtypes(include=['float64', 'int64']).columns)
        num_categorical_features = len(self.data.select_dtypes(include=['object']).columns)

        return num_numeric_features, num_categorical_features
    
    def explain_model(self, technique_name, model, X_test):
        # Initialize the SHAP explainer
        explainer = shap.Explainer(model)

        # Calculate SHAP values
        shap_values = explainer(X_test)

        # Plot the SHAP summary plot
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
        plt.title(f'SHAP Values - {technique_name}')
        plt.show()








