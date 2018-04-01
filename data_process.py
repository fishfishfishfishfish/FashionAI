from mxnet import image
import matplotlib.pyplot as plt
import csv


def get_data():
    basic_path = "E:/Documents/Python_Project/FashionAI/warm_up_train_20180201/web/"
    with open(basic_path + "Annotations/skirt_length_labels.csv", 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            img = image.imdecode(open(basic_path + row[0], 'rb').read())
            plt.imshow(img.asnumpy())
            plt.show()
            break


get_data()