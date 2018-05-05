import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import numpy as np
import time
import logging


def accuracy(output, label):
    return nd.mean(output.argmax(axis=1) == label).asscalar()


def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0
    for data, label in data_iterator:
        label = label.as_in_context(ctx)
        data = data.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)


data_name = 'skirt_len'
timeStr = time.strftime("%Y-%m-%d(%H-%M-%S)", time.localtime())
logging.basicConfig(level=logging.DEBUG,
                    filename=timeStr+data_name+'4-layer lr = 0.001 wd=0.0001 mom=0.9 with aug.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

Context = mx.cpu()
classNum = 6
decay_epoch = 9
trainPath = 'train_' + data_name + '_data'
trainAugPath = 'train_' + data_name + '_data_aug'
testPath = 'test_' + data_name + '_data'
batch_size = 300
dropProb1 = 0.2
dropProb2 = 0.5

trainSet = nd.load(trainPath)
trainAugSet = nd.load(trainAugPath)
trainSetData = nd.concat(trainSet['data'], trainAugSet['data'], dim=0)
trainSetLabel = nd.concat(trainSet['label'], trainAugSet['label'], dim=0)
testSet = nd.load(testPath)
trainDataset = gluon.data.ArrayDataset(trainSetData, trainSetLabel)
testDataset = gluon.data.ArrayDataset(testSet['data'], testSet['label'])

trainIter = gluon.data.DataLoader(trainDataset, batch_size, shuffle=True)
testIter = gluon.data.DataLoader(testDataset, batch_size, shuffle=False)

NET = gluon.nn.Sequential()
with NET.name_scope():
    NET.add(gluon.nn.Dense(1024, activation='relu'))
    # NET.add(gluon.nn.Dropout(dropProb1))
    NET.add(gluon.nn.Dense(512, activation='relu'))
    NET.add(gluon.nn.Dense(128, activation='relu'))
    # NET.add(gluon.nn.Dropout(dropProb2))
    NET.add(gluon.nn.Dense(classNum))

NET.initialize(ctx=Context)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(NET.collect_params(), 'sgd', {'learning_rate': .01, 'wd': 0.00005, 'momentum': 0.9})


# =============================================================================
# 训练
# =============================================================================
for epoch in range(1000):
    if epoch > decay_epoch and epoch % 10 == 0 and trainer.learning_rate > 0.00000001:
        trainer.set_learning_rate(trainer.learning_rate * 0.1)
        logging.info('set learning rate to ' + str(trainer.learning_rate))
    train_loss = 0.
    train_acc = 0.
    # 计算运行时间
    start = time.time()
    for Data, Label in trainIter:
        Label = Label.as_in_context(Context)
        Data = Data.as_in_context(Context)
        with mx.autograd.record():
            Output = NET(Data)
            Losses = softmax_cross_entropy(Output, Label)
        Losses.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(Losses).asscalar()
        train_acc += accuracy(Output, Label)
    test_acc = evaluate_accuracy(testIter, NET, Context)
#    net.save_params('_ep_' + str(epoch+1)+ '_trainAcc_' + str(train_acc/len(train_iter))
#                        + '_testAcc_' + str(test_acc))
    logging.info("Epoch %d. Loss: %f, Train acc %f, Test acc %f, Time %.1f sec" % (
        epoch, train_loss/len(trainIter),
        train_acc/len(trainIter), test_acc, time.time()-start))
