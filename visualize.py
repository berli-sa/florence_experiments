import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("cross_similarity_heatmap.csv", index_col=0)

# Convert columns to integers (time in ms)
df.columns = df.columns.astype(int)

# Create the heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Similarity Score'})

plt.xlabel("Time of Caption (ms)")
plt.ylabel("Label (Definition)")
plt.title("Cross-Similarities Between Detailed Caption and Definitions of Labels in Frame Heatmap")
plt.tight_layout()
plt.savefig("cross_similarity_heatmap.png", dpi=300)
