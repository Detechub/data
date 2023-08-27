from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# Simulated dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 4], [3, 1]])
y = np.array([1, 1, 1, -1, -1, -1])

# Create SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# Train the model
model.fit(X, y)

# New data point for classification
new_point = np.array([[2.5, 3]])

# Predict the class of the new point
predicted_class = model.predict(new_point)

# Plotting the data and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(new_point[:, 0], new_point[:, 1], color='green', marker='x', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])

plt.show()

print("Predicted class:", predicted_class)
