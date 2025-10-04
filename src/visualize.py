import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) == 0:
        print("⚠️ Pas de colonnes numériques pour visualiser.")
        return

    sns.pairplot(df[numeric_cols].sample(min(100, len(df))) )
    plt.show()
