import mxnet as mx
from mxnet import ndarray as nd
import matplotlib.pyplot as plt


# show image
def show_img(data_iter):
    while True:
        try:
            x = data_iter.next().data[0]
            x = nd.transpose(x, (0, 2, 3, 1))
            x = x[0].clip(0, 255)/255
            plt.imshow(x.asnumpy())
            plt.show()
        except StopIteration:
            break


# forward
def out_concat_data_label(data_iter, senet, dpn, out_path,
                          dpn_r_mean=124, dpn_g_mean=117, dpn_b_mean=104, dpn_scale=0.0167,
                          se_r_mean=0, se_g_mean=0, se_b_mean=0, se_scale=1, ):
    data_list = []
    label_list = []
    i = 0
    print('start forward!!')
    while True:
        if i % 100 == 0:
            print('reach ', i, ' image')
        i = i + 1
        try:
            se_batch = data_iter.next()
            dpn_batch = se_batch
            # normalize
            dpn_batch.data[0][0:, 0, :, :] -= dpn_r_mean
            dpn_batch.data[0][0:, 1, :, :] -= dpn_g_mean
            dpn_batch.data[0][0:, 2, :, :] -= dpn_b_mean
            dpn_batch.data[0][:, :, :, :] *= dpn_scale
            se_batch.data[0][0:, 0, :, :] -= se_r_mean
            se_batch.data[0][0:, 1, :, :] -= se_g_mean
            se_batch.data[0][0:, 2, :, :] -= se_b_mean
            se_batch.data[0][:, :, :, :] *= se_scale
            # forward
            senet.forward(se_batch)
            senet_out = senet.get_outputs()[0]
            dpn.forward(dpn_batch)
            dpn_out = dpn.get_outputs()[0]
            feature = nd.concat(*[senet_out, dpn_out])
            data_list.append(feature.asnumpy())
            label_temp = se_batch.label[0].asscalar()
            if label_temp == dpn_batch.label[0].asscalar():
                label_list.append(label_temp)
            else:
                print('label conflict!')
                break
        except StopIteration:
            print('stop')
            break
    # record
    out_dict = {'data': nd.array(data_list), 'label': nd.array(label_list)}
    nd.save(out_path, out_dict)


data_name = 'train_skirt_len'
seNet_model_path = '..\\models\\se-resnext50\\se-resnext-imagenet-50-0'
dpn_model_path = '..\\models\\dpn98\\dpn98'
rec_path = '..\\imageRecord\\' + data_name + '.rec'
out_name = data_name + '_data_aug'
imageShape = 224
Context = mx.cpu()
# se-reNeXt
seNetSym, arg_params, aux_params = mx.model.load_checkpoint(seNet_model_path, 125)
seNetSym = seNetSym.get_internals()['dp1_output']
seNet = mx.mod.Module(symbol=seNetSym, context=Context, label_names=None)
seNet.bind(for_training=False, data_shapes=[('data', (1, 3, imageShape, imageShape))])
seNet.set_params(arg_params, aux_params, allow_missing=False)
# dpn
dpnSym, arg_params, aux_params = mx.model.load_checkpoint(dpn_model_path, 0)
dpnSym = dpnSym.get_internals()['flatten_output']
dpnNet = mx.mod.Module(symbol=dpnSym, context=Context, label_names=None)
dpnNet.bind(for_training=False, data_shapes=[('data', (1, 3, imageShape, imageShape))])
dpnNet.set_params(arg_params, aux_params, allow_missing=False)
# get data
dataIter = mx.io.ImageRecordIter(path_imgrec=rec_path,
                                 data_shape=(3, imageShape, imageShape),
                                 batch_size=1,
                                 mirror=True,
                                 max_rotate_angle=360,
                                 max_random_contrast=0.5,
                                 max_random_illumination=2)

out_concat_data_label(dataIter, seNet, dpnNet, out_name)
