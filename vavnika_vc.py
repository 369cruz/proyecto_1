import numpy as np
from sklearn.svm import SVC

# Crea un conjunto de puntos aleatorios
np.random.seed(0)
X = np.random.randn(100, 2)

# Crea una clasificación binaria basada en los puntos
y = (X[:, 0] > 0).astype(int)

# Crea un modelo SVM y entrena con los puntos y la clasificación
clf = SVC(kernel='linear', C=1, random_state=0)
clf.fit(X, y)

# Calcula la dimensión VC del modelo
n_samples, n_features = X.shape
margin = 1 / np.linalg.norm(clf.coef_)
VC_dimension = n_samples * (n_features + 1) * margin ** (-2)

print("La dimensión VC del modelo es:", VC_dimension)
