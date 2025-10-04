import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    few_numeric_cols = numeric_cols[:4]
    print(f"Variables affichées dans le pairplot : {few_numeric_cols}")
    sns.pairplot(df[few_numeric_cols].sample(min(100, len(df))))
    plt.tight_layout()
    plt.savefig("pairplot.png")
    plt.close()
    print("✅ Pairplot sauvegardé dans pairplot.png")