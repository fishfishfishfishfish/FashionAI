# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:19:42 2018

@author: CJT
"""
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import numpy as np
from time import time

# =============================================================================
# 数据
# =============================================================================
DATA = nd.array(np.loadtxt('output/collar_design.txt'))
LABEL = nd.array(np.loadtxt('output/collar_design_label.txt'))
collar_design_classes = 5

batch_size = 300
train_dataset = gluon.data.ArrayDataset(DATA[:-100], LABEL[:-100])
test_dataset = gluon.data.ArrayDataset(DATA[-100:], LABEL[-100:])

train_iter = gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)
#print(len(train_iter))
test_iter = gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)
#print(len(test_dataset))

# 读取第一个随机数据块
#for data, label in data_iter:
#    print(data, label)
#    break

# =============================================================================
# 定义模型
# =============================================================================
ctx = mx.gpu()

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1024, activation='relu'))
    net.add(gluon.nn.Dense(512, activation='relu'))
    net.add(gluon.nn.Dense(128, activation='relu'))
    net.add(gluon.nn.Dense(collar_design_classes))

net.initialize(ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(),'sgd',{
                        'learning_rate':.001
#                        ,'wd': 0.00005
                        })

# =============================================================================
# 训练
# =============================================================================
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0
    for data, label in data_iterator:
        label = label.as_in_context(ctx)
        data = data.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iterator)


for epoch in range(100):
    train_loss = 0.
    train_acc = 0.
    # 计算运行时间
    start = time()
    for data, label in train_iter:
        label = label.as_in_context(ctx)
        data = data.as_in_context(ctx)
        with mx.autograd.record():
            output = net(data)
            losses = softmax_cross_entropy(output, label)
        losses.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(losses).asscalar()
        train_acc += accuracy(output, label)
    test_acc = evaluate_accuracy(test_iter, net, ctx)
#    net.save_params('_ep_' + str(epoch+1)+ '_trainAcc_' + str(train_acc/len(train_iter)) 
#                        + '_testAcc_' + str(test_acc))
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f, Time %.1f sec" % (
        epoch, train_loss/len(train_iter),
        train_acc/len(train_iter), test_acc, time()-start))


















