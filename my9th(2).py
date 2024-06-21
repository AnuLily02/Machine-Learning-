from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train the decision tree classifier
clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(10,10))  # Set the figure size for better visualization
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, proportion=True)
plt.show()
