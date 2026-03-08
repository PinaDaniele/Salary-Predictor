import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

DATASET_PATH = ".\\dataset.csv"
EDA_PATH = ".\\eda"

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df.drop(["salary_currency", "salary"], axis=1, inplace=True)

    print(df["company_size"].unique())
    print(df["experience_level"].unique())

    print(df.head())
    print(df.info())
    
    return df
    
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape

    if min(k-1, r-1) == 0:
        return 0
    else:
        return np.sqrt(phi2/min(k-1, r-1))


def analyze_data(df):
    if not os.path.exists(EDA_PATH):
        os.mkdir(EDA_PATH)

    #salary distribution
    plt.figure(figsize=(10, 12))
    sns.histplot(df["salary_in_usd"], kde=True, color="blue")
    plt.title("Salary distribution")
    plt.savefig(os.path.join(EDA_PATH, "salary_distribution.png"))

    #numerical features heatmap
    corr_df = df[["work_year", "salary_in_usd", "experience_level", "company_size"]].copy()
    corr_df["company_size"] = corr_df["company_size"].map({"S":0, "M":1, "L":2})
    corr_df["experience_level"] = corr_df["experience_level"].map({"Entry-level":0, "Mid-level":1, "Senior":2, "Executive":3})
    plt.figure(figsize=(10,10))
    sns.heatmap(corr_df.corr(method="spearman"), annot=True, cmap="coolwarm")
    plt.title("Numerical feature correlation heatmap")
    plt.savefig(os.path.join(EDA_PATH, "num_feature_heatmap.png"))

    #categorical features heatmap (between each other)
    cat_cols = ["job_title", "job_category", "employee_residence", "employment_type", "work_setting", "company_location"]
    rows = []
    for value1 in cat_cols:
        col = []
        for value2 in cat_cols:
            col.append(cramers_v(df[value1], df[value2]))
        rows.append(col)
    matrix = pd.DataFrame(rows, index=cat_cols, columns=cat_cols)
    plt.figure(figsize=(14,14))
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.savefig(os.path.join(EDA_PATH, "cat_feature_heatmap.png"))

#TODO: DROP employee residence, job_category
#log scale salary_in_usd
#encode cat features in some way
def preprocess_data(df):
    salaries = df["salary_in_usd"].to_numpy()
    Y = np.log(salaries)

    df.drop(columns=["employee_residence", "job_category", "salary_in_usd"], inplace=True)
    df["company_size"] = df["company_size"].map({"S":0, "M":1, "L":2})
    df["experience_level"] = df["experience_level"].map({"Entry-level":0, "Mid-level":1, "Senior":2, "Executive":3})

    job_title_threshold = 0.05
    job_counts = df["job_title"].value_counts(normalize=True)
    df["job_title"] = df["job_title"].apply(lambda x: x if job_counts[x] > job_title_threshold else "Other")

    region_threshold = 0.10
    region_counts = df["company_location"].value_counts(normalize=True)
    df["company_location"] = df['company_location'].apply(lambda x: x if region_counts[x] > region_threshold else "Other")

    year_min = df["work_year"].min()
    df["work_year"] = df["work_year"].apply(lambda x: x-year_min)

    num_cols = ["work_year", "experience_level", "company_size"]
    cat_cols = ["job_title", "employment_type", "work_setting", "company_location"]

    X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, Y_train, Y_test, preprocessor



def main():
    df = load_data()
    analyze_data(df)
    X_train, X_test, Y_train, Y_test, preprocessor = preprocess_data(df)
    joblib.dump(preprocessor, "preprocessor")


if __name__ == "__main__":
    main()