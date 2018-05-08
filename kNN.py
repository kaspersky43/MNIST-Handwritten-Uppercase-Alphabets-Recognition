from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import string

from util import *

X, Y = get_data(100000)
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, train_size=0.7,random_state=1)

pca = PCA(n_components=0.7)
train_images = pca.fit_transform(train_images)
test_images_2 = pca.transform(test_images)

k_neighbors = range(1,8)

#visuazliation config
vtest = False
n_show_result = 5
display_error = False

acc_output = []
for ne in k_neighbors: 
    neigh = KNeighborsClassifier(n_neighbors=ne)
    neigh.fit(train_images, train_labels)
    knn_output = neigh.predict(test_images_2)
    print("number of neighbors: %d" % ne)

    #visualization
    title_accessory = None
    if(vtest == True):
        title_accessory = ["in neighbors: ",str(ne)]
    visualize(vtest, display_error, n_show_result, knn_output, test_images, test_labels, title_accessory)

    #output
    acc_score = accuracy_score(test_labels, knn_output)
    print("Accuracy with %d neighbors: %f" % (ne, acc_score))
    acc_output.append(acc_score)

ma = acc_output.index(max(acc_output))
print("kNN most accurate when k is: %d with the accuracy: %f" % ((ma+1),max(acc_output)))
