import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd 
import seaborn as sns

# Load the dataset
file_path = os.path.join("Datasets", "housing", "housing.csv")
df = pd.read_csv(file_path)

# Histograms
df.hist(bins=50) #figsize=(15, 15))
plt.show()

# Histogram for median income
df["median_income"].hist(bins=50)
plt.xlabel("Median Income")
plt.ylabel("Frequency")
plt.title("Histogram of Median Income")
plt.show()

# Create income categories
df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3., 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

# Scatter plots
df.plot(kind="scatter", x="longitude", y="latitude")
plt.title("Scatter Plot of Longitude vs Latitude")
plt.show()

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.title("Scatter Plot of Longitude vs Latitude with Alpha=0.1")
plt.show()

# Additional feature creation
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Correlation matrix
corr_matrix = df.corr(method="pearson", numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Bar chart of number of houses per income category
df["income_cat"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Income Category")
plt.ylabel("Number of Houses")
plt.title("Bar Chart of Income Category")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for selected features
selected_features = ["median_house_value", "median_income", "rooms_per_household", "bedrooms_per_room", "population_per_household"]
sns.pairplot(df[selected_features], diag_kind="kde", plot_kws={'alpha':0.2})
plt.show()
