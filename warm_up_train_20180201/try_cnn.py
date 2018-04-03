from mxnet import image
import pandas
from data_process import DataLoader

Augs = [image.ForceResizeAug((512, 512))]
raw_data = pandas.read_csv("web/Annotations/skirt_length_labels.csv")
train_data = raw_data.iloc[0: 2000]
DL = DataLoader("web/", train_data, 10, True, for_show=True)
# print(next(DL.__iter__())[1])
# i = 0
# for _, _ in DL:
#     i += 1
# print(i)
# DLGen = ((Data, Label) for Data, Label in DL)

