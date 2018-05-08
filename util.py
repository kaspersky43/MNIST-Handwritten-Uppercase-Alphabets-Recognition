import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(limit=None):
    print("Reading the data")
    df = pd.read_csv('./input/handwritten_data.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def plot_image(image, title, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()

digit_to_letter_map = { key: value for key, value in enumerate(string.ascii_uppercase, 0)}

delimiter = " "

def visualize(vtest,display_error, vmax, pred_y, test_images, test_labels, title_add):
    for i in range(len(pred_y)):
        if(vtest == True):
            if(i == vmax): break
            image = test_images[i].reshape((28,28))
            title = [digit_to_letter_map[pred_y[i]]]
            if(title_add != None):
                for ta in title_add:
                    title.append(ta)
            title = delimiter.join(title)
            plot_image(image,title)
        else:
            if(display_error == True):
                if(pred_y[i] != test_labels[i]):
                     print("Error predict: %s actual: %s " % (digit_to_letter_map[pred_y[i]], digit_to_letter_map[test_labels[i]]))        

