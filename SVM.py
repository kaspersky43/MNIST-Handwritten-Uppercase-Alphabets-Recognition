from sklearn import svm
from sklearn.model_selection import * 
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from util import *

def transform_pca(train_images,test_images,components):
	pca = PCA(n_components=components)
	train_images = pca.fit_transform(train_images)
	test_images = pca.transform(test_images)
	return train_images, test_images


X, Y = get_data(100000)
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, train_size=0.7,random_state=1)

C = 10
classifier = svm.SVC(kernel='rbf',C=C,gamma='auto',tol=0.0001)
train_images, test_images = transform_pca(train_images, test_images, 0.7)
classifier.fit(train_images, train_labels)
pred_y = classifier.predict(test_images)

#visualization config
vtest = False
n_show_result = 5
display_error = False
visualize(vtest, display_error, n_show_result, pred_y, test_images, test_labels, None)
print("Accuracy: %f" % classifier.score(test_images, test_labels))
