from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
clf.fit(X_train,y_train)
tree.plot_tree(clf)
plt.show()

y_pred = clf.predict(X_test)
print("Predicted values for test are")
print(y_pred)

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()