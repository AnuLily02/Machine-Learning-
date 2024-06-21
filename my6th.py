import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

file_path = os.path.join("Datasets", "housing", "housing.csv")
df = pd.read_csv(file_path)

# Handle missing values
imputer = SimpleImputer(strategy="median")
df["total_bedrooms"] = imputer.fit_transform(df[["total_bedrooms"]])

# Print dataset information after imputing
print("\n\n _________Printing dataset information________\n\n")
print(df.info())

# Encode categorical variables
housing_cat = df[['ocean_proximity']]

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("Ordinal Encoding:\n", housing_cat_encoded[:10])

# One-Hot Encoding
cat_encoder = OneHotEncoder()
housing_cat_hot = cat_encoder.fit_transform(housing_cat)
print("One-Hot Encoding:\n", housing_cat_hot.toarray())
print("Categories:\n", cat_encoder.categories_)

# Feature Scaling
new_housing = df.drop(['ocean_proximity'], axis=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_housing)
scaler_df = pd.DataFrame(scaled_data, columns=new_housing.columns)
print("Scaled Data:\n", scaler_df.head())
