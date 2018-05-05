from data_deal import DataLoader
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
import pandas
import numpy as np

# =============================================================================
# 数据
# =============================================================================
root_path = "D:/GT2/As student/software hardware co-design/fashionAI/data/base/"
header = [0, 1, 2]
all_raw_data = pandas.read_csv(root_path + "Annotations/label.csv", names=header)

collar_design_raw_data = all_raw_data.loc[all_raw_data[1] == 'collar_design_labels']
collar_design_classes = 5
print('collar_design_labels: ', len(collar_design_raw_data))

neckline_design_raw_data = all_raw_data.loc[all_raw_data[1] == 'neckline_design_labels']
neckline_design_classes = 10
print('neckline_design_labels: ', len(neckline_design_raw_data))

skirt_length_raw_data = all_raw_data.loc[all_raw_data[1] == 'skirt_length_labels']
skirt_length_classes = 6
print('skirt_length_labels: ', len(skirt_length_raw_data[1]))

sleeve_length_raw_data = all_raw_data.loc[all_raw_data[1] == 'sleeve_length_labels']
sleeve_length_classes = 9
print('sleeve_length_labels: ', len(sleeve_length_raw_data[1]))

neck_design_raw_data = all_raw_data.loc[all_raw_data[1] == 'neck_design_labels']
neck_design_classes = 5
print('neck_design_labels: ', len(neck_design_raw_data[1]))

coat_length_raw_data = all_raw_data.loc[all_raw_data[1] == 'coat_length_labels']
coat_length_classes = 8
print('coat_length_labels: ', len(coat_length_raw_data[1]))

lapel_design_raw_data = all_raw_data.loc[all_raw_data[1] == 'lapel_design_labels']
lapel_design_classes = 5
print('lapel_design_labels: ', len(lapel_design_raw_data[1]))

pant_length_raw_data = all_raw_data.loc[all_raw_data[1] == 'pant_length_labels']
pant_length_classes = 6
print('pant_length_labels: ', len(pant_length_raw_data[1]))

# define a simple data batch
# from collections import namedtuple
# http://huthnpku.is-programmer.com/posts/40709.html
# Batch = namedtuple('Batch', ['data'])

# =============================================================================
# 加载模型
# =============================================================================
# se-reNeXt
seNetSym, arg_params, aux_params = mx.model.load_checkpoint('model/se-resnext-imagenet-50-0', 125)
all_layers = seNetSym.get_internals()
# print(all_layers.list_outputs()[-10:])
seNetSym = all_layers['fc1_output']
seNet = mx.mod.Module(symbol=seNetSym, context=mx.cpu(), label_names=None)
seNet.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
seNet.set_params(arg_params, aux_params)
# dpn
dpnSym, arg_params, aux_params = mx.model.load_checkpoint('model/dpn98', 0)
all_layers = dpnSym.get_internals()
# print(all_layers.list_outputs()[-10:])
dpnSym = all_layers['fc6_output']
dpn = mx.mod.Module(symbol=dpnSym, context=mx.gpu(), label_names=None)
dpn.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
dpn.set_params(arg_params, aux_params)


# =============================================================================
# 通过网络，得到两个网络（1*1000）的输出，并合并为1*2000的数组
# =============================================================================
def get_output(name, RAW_DATA):
    batch_size = 1
    train_data = DataLoader(root_path, RAW_DATA,
                            batch_size, shuffle=True, for_show=False, 
                            pic_size=(224, 224))

    f = open("output/" + name + ".txt", "w")
    f_label = open("output/" + name + "_label.txt", "w")
    
    cnt = 1
    for data, label in train_data:
        print(name, cnt)
        cnt = cnt+1
        seNet.forward(Batch([data]))
        seNet_out = nd.array(seNet.get_outputs()[0].asnumpy())
        
        dpn.forward(Batch([data]))
        dpn_out = nd.array(dpn.get_outputs()[0].asnumpy())
        
        net_in = nd.concat(*[seNet_out, dpn_out])
    #    net_in = nd.stack(*[seNet_out, dpn_out])
    #    print(net_in)
        for i in nd.arange(len(net_in[0])):
            f.write("%f " % (net_in[0][i].asscalar()))
        f.write("\n")
        f_label.write("%d\n" % (label.argmax(axis=1).asscalar()))
    f.close()
    f_label.close()


# =============================================================================
# 得到输出
# =============================================================================
get_output("collar_design", collar_design_raw_data)

# get_output('neckline_design', neckline_design_raw_data)
#
# get_output('skirt_length', skirt_length_raw_data)
#
# get_output('sleeve_length', sleeve_length_raw_data)
#
# get_output('neck_design', neck_design_raw_data)
#
# get_output('coat_length', coat_length_raw_data)
#
# get_output('lapel_design', lapel_design_raw_data)
#
# get_output('pant_length', pant_length_raw_data)

# =============================================================================
# 读取数据
# =============================================================================
a = nd.array(np.loadtxt('output/collar_design.txt'))
print(a)
