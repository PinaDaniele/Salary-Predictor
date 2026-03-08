import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

def main():
    df = load_data()
    analyze_data(df)


if __name__ == "__main__":
    main()