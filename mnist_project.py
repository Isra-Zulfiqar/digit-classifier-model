# mnist_project.py

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST dataset
print("Downloading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# 3. Train SGD Classifier
print("\nTraining SGD Classifier...")
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)

# 4. Evaluate SGD
print("\nSGD Classifier Report:")
y_pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, y_pred_sgd))

# 5. Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate Random Forest
print("\nRandom Forest Classifier Report:")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# 7. Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.matshow(conf_matrix, cmap='viridis')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

from sklearn.metrics import classification_report

# Save classification reports to a file
sgd_report = classification_report(y_test, sgd.predict(X_test))
rf_report = classification_report(y_test, rf.predict(X_test))

with open("classification_reports.txt", "w") as f:
    f.write("SGD Classifier Report:\n")
    f.write(sgd_report)
    f.write("\n\nRandom Forest Classifier Report:\n")
    f.write(rf_report)
