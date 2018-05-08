from sklearn.model_selection import train_test_split
import numpy as np
from util import *

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=5000):
        self.W = np.random.randn(784)
        self.b = 0

        costs = []
        for epoch in range(epochs):
            # misclassified samples to determine the cost
            y_pred = self.predict(X)
            incorrect = np.nonzero(Y != y_pred)[0]
            if len(incorrect) == 0: break # when everything is correct

            i = np.random.choice(incorrect)
            self.W += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(len(Y))
            costs.append(c)

            if(epoch % 200 == 0):
                print('epoch %d:' % epoch)
                print('=== cost: %f'% c)
            
    def predict(self, X):
        return np.sign(X.dot(self.W) + self.b)

    def score(self, X, Y):
        return np.mean(self.predict(X) == Y)

if __name__ == '__main__':

    X, Y = get_data(100000)    
    index = np.logical_or(Y == 0, Y == 1)
    X = X[index]
    Y = Y[index]
    train_images, test_images, train_labels, test_labels = train_test_split(X, Y, train_size=0.7,random_state=1)

    model = Perceptron()
    model.fit(train_images, train_labels, learning_rate=0.01)
    pred_y = model.predict(test_images)

    #visualization config
    vtest = False
    n_show_result = 5 
    display_error = False
    if(display_error == True):
        pred_y[pred_y == -1] = 0
    visualize(vtest, display_error, n_show_result, pred_y, test_images, test_labels, None)
    
    print("Accuracy:", model.score(test_images, test_labels))
