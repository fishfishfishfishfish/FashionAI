import mxnet as mx
import logging

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name, last_fc_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    flatten = symbol.get_internals()[layer_name]
    fc_new = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc_new')
    net = mx.symbol.SoftmaxOutput(data=fc_new, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if last_fc_name not in k})
    return net, new_args


logging.basicConfig(level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

batch_size = 15
data_shape = 512
sys, arg_param, aux_param = mx.model.load_checkpoint('dpn68/dpn68', 0)
sys, arg_param = get_fine_tune_model(sys, arg_param, 6, 'flatten_output', 'fc6')
print(sys.infer_shape(data=(batch_size, 3, data_shape, data_shape))[1])
dataIterTrain = mx.io.ImageRecordIter(path_imgrec='train_skirt_len.rec',
                                      data_shape=(3, data_shape, data_shape),
                                      batch_size=batch_size,
                                      scale=1.0/255,
                                      data_name="data",
                                      label_name="softmax_label",
                                      preprocess_threads=4)
dataIterTest = mx.io.ImageRecordIter(path_imgrec='test_skirt_len.rec',
                                     data_shape=(3, data_shape, data_shape),
                                     batch_size=batch_size,
                                     scale=1.0/255,
                                     data_name="data",
                                     label_name="softmax_label",
                                     preprocess_threads=4)
dpn_model = mx.mod.Module(symbol=sys, context=mx.gpu())
kvstore = mx.kv.create('device')
lr_decay = mx.lr_scheduler.FactorScheduler(step=80, factor=0.9)
optimizer = mx.optimizer.SGD(rescale_grad=1.0/batch_size,
                             learning_rate=0.001,
                             wd=0.00005,
                             momentum=0.9,
                             lr_scheduler=lr_decay)
RMSEmetric = mx.metric.RMSE()
ACCmetric = mx.metric.Accuracy()
metric = mx.metric.CompositeEvalMetric(metrics=[RMSEmetric, ACCmetric])
speed_callback = mx.callback.Speedometer(batch_size, 200)
save_checkpoint = mx.callback.do_checkpoint("tuned-dpn68/tuned-dpn68", 1)

logging.info("start train")
dpn_model.fit(dataIterTrain,
              eval_data=dataIterTest,
              optimizer=optimizer,
              eval_metric=metric,
              batch_end_callback=speed_callback,
              epoch_end_callback=save_checkpoint,
              eval_end_callback=speed_callback,
              num_epoch=30,
              kvstore=kvstore)

