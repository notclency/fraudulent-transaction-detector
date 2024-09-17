import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('fraud_data.csv')


def peak_data():
    print(df.head())
    print(df.tail())
    print(df.describe())
    print(df.info())


def clean_data():
    # Clean 'is_fraud' column
    df['is_fraud'] = df['is_fraud'].astype(str).str.extract(r'(\d)').fillna(0).astype(int)
    print("Unique values in 'is_fraud' column:", df['is_fraud'].unique())

    # Fraud counts by category
    fraud_counts_by_category = df[df['is_fraud'] == 1].groupby('category').size().reset_index(name='fraud_count')
    fig = px.bar(fraud_counts_by_category, x='category', y='fraud_count', title='Fraud count by product category')
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, xaxis_tickangle=-90, xaxis_title='Product Category',
                      yaxis_title='Fraud Count')
    # fig.show()

    # Fraud counts by state
    frauds_by_state = df[df['is_fraud'] == 1].groupby('state').size().reset_index(name='fraud_count')
    merged_df = df.groupby('state').size().reset_index(name='total_transactions').merge(frauds_by_state, on='state',
                                                                                        how='left')
    print(merged_df)
    print(merged_df.isnull().sum())
    sorted_df = merged_df.sort_values(by='fraud_count', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=sorted_df['state'], y=sorted_df['total_transactions'], name='Total Transactions',
                         marker_color='lightskyblue'))
    fig.add_trace(go.Bar(x=sorted_df['state'], y=sorted_df['fraud_count'], name='Fraud Count', marker_color='crimson'))
    fig.update_layout(barmode='group', xaxis_tickangle=-90, xaxis_title='State', yaxis_title='Count',
                      title='Total Transactions and Fraud Count by State')
    # fig.show()

    fig = px.bar(sorted_df, x='state', y='fraud_count', title='Fraud count by state')
    # fig.show()

    category_counts = df.groupby('category')['trans_num'].nunique().reset_index(name='transaction_count')
    fig = px.bar(category_counts, x='category', y='transaction_count', title='Unique transaction count by category')
    fig.update_layout(xaxis={'categoryorder': 'total descending'})
    # fig.show()

    # Data transformation
    df['dob'] = pd.to_datetime(df['dob'], format='%d-%m-%Y')
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%d-%m-%Y %H:%M')
    df['trans_date'] = df['trans_date_trans_time'].dt.date
    df['trans_time'] = df['trans_date_trans_time'].dt.time
    df['age'] = df['trans_date'].apply(lambda x: x.year) - df['dob'].dt.year

    fraud_df = df[df['is_fraud'] == 1]
    fig = px.histogram(fraud_df, x='age', title='Age distribution of victims of fraudulent transactions', nbins=10)
    # fig.show()


def model_training_and_prediction():
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df.drop(
        columns=['trans_date_trans_time', 'trans_date', 'dob', 'trans_num', 'trans_time', 'merchant', 'state', 'city'],
        inplace=True)

    categorical_columns = ['category', 'job']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df['is_fraud'] = df['is_fraud'].astype(int)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def random_forest():
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=7)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    X_train, y_train, X_test, y_test = model_training_and_prediction()
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")

    best_params = grid_search.best_params_
    final_rf = RandomForestClassifier(**best_params, random_state=7)
    final_rf.fit(X_train, y_train)

    y_pred = final_rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    print(f"\nAccuracy: {accuracy}")
    print(f"\nConfusion Matrix: {confusion}")
    print(f"\nClassification Report: {report_df}")


def main():

    peak_data()
    clean_data()
    random_forest()


if __name__ == "__main__":
    main()
