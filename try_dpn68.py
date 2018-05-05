import mxnet as mx

kvstore = mx.kv.create('local')
dataiter = mx.io.ImageRecordIter(path_imgrec='skirt_len.rec', data_shape=(3, 200, 200), batch_size=2)
sys, arg_param, aux_param = mx.model.load_checkpoint('dpn68/dpn68', 0)

feature_layer = sys.get_internals()['softmax_output']
module = mx.mod.Module(symbol=feature_layer, context=mx.cpu())
module.bind(for_training=False, data_shapes=[('data', (2, 3, 200, 200))])
module.set_params(arg_param, aux_param, allow_missing=False)

module.forward(dataiter.next())
features = module.get_outputs()[0].asnumpy()
print(features[0])
