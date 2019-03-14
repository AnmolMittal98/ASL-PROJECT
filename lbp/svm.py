import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

lbp_train = np.load('../lbp/lbp_trainset.npy')
labels = lbp_train[:, 8]
features = lbp_train[:, [0, 1, 2, 3, 4, 5, 6, 7]]
scaler = StandardScaler()
features = scaler.fit_transform(features)
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.20, random_state=13, shuffle=True)
svm_classifier = svm.SVC(decision_function_shape='ovo', kernel='poly', gamma='scale')
svm_classifier.fit(x_train, y_train)
print(svm_classifier.score(x_val, y_val))
