from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from util import *

X, Y = get_data(100000)
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, train_size=0.7,random_state=1)

classifier = MultinomialNB()
classifier.fit(train_images, train_labels)
pred_y = classifier.predict(test_images)

#visualization config
vtest = False
n_show_result = 5
display_error = False
visualize(vtest, display_error, n_show_result, pred_y, test_images, test_labels, None)
print("Accuracy: %f" % classifier.score(test_images, test_labels))

