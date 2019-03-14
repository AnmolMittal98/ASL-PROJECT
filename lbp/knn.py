import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

lbp_train = np.load('../lbp/lbp_trainset.npy')
labels = lbp_train[:, 8]
features = lbp_train[:, [0, 1, 2, 3, 4, 5, 6, 7]]
scaler = StandardScaler()
features = scaler.fit_transform(features)
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, random_state=13, shuffle=True)
knn_classifier = KNeighborsClassifier(n_neighbors=29)
knn_classifier.fit(x_train, y_train)
print(knn_classifier.score(x_val, y_val))
