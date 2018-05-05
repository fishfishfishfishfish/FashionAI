from mxnet import image
from mxnet import nd
import numpy as np


def get_image_mat(data_path, pic_size):
    img = image.imdecode(open(data_path, 'rb').read())
    if pic_size[0]*pic_size[1] != 0:
        img = image.ForceResizeAug(pic_size)(img)
    return img.astype('float32').asnumpy()


def get_labels(string):
    y = []
    for s in string:
        if s == 'y':
            y.append(1)
        elif s == 'm':
            y.append(0.5)
        else:
            y.append(0)
    return y


class DataLoader(object):
    def __init__(self, data_path, raw_data, batch_size, shuffle, transform=None, for_show=False, pic_size=(448, 448)):
        self.data_path = data_path
        self.raw_data = raw_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.for_show = for_show
        self.pic_size = pic_size

    def __iter__(self):
        n = len(self.raw_data)
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.raw_data = self.raw_data.iloc[idx]

        for i in range(n//self.batch_size):
            x = []
            y = []
            for j in range(i*self.batch_size, (i+1)*self.batch_size):
                x.append(get_image_mat(self.data_path + self.raw_data.iloc[j, 0], self.pic_size))
                y.append(get_labels(self.raw_data.iloc[j, 2]))
                # print(raw_data.iloc[j, 0])
            x = nd.array(np.array(x))
            y = nd.array(y)
            if self.transform is not None:
                x, y = self.transform(x, y)
            if not self.for_show:
                x = nd.transpose(x, (0, 3, 1, 2))

            yield (x, y)

    def __len__(self):
        return len(self.raw_data)//self.batch_size


def apply_aug_list(img, augs):
    for f in augs:
        img = f(img)
    return img


def get_transform(augs):
    def transform(data, label):
        # data: sample x height x width x channel
        # label: sample
        data = data.astype('float32')
        if augs is not None:
            # apply to each sample one-by-one and then stack
            data = nd.stack(*[
                apply_aug_list(d, augs) for d in data])
        return data, label.astype('float32')
    return transform
