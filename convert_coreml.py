import numpy as np
import tensorflow as tf
import coremltools as ct
from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string(
    'weights',
    './checkpoints/yolov4.tf',
    'path to weights file'
)
flags.DEFINE_string(
    'output',
    './checkpoints/yolov4.mlmodel',
    'path to output'
)
flags.DEFINE_integer(
    'input_size',
    608,
    'input size'
)


def gen_mlmodel():
    model = tf.keras.models.load_model(FLAGS.weights, compile=False)
    logging.info("Loaded SavedModel from: {}".format(FLAGS.weights))
    mlmodel = ct.convert(model)
    logging.info("CoreML model saved to: {}".format(FLAGS.output))
    return model, mlmodel


def test_models(model, mlmodel):
    x = np.random.randn(1, FLAGS.input_size, FLAGS.input_size, 3)
    tf_out = model.predict([x])
    print('TF prediction pass OK')

    coreml_out_dict = mlmodel.predict({"input_1": x})
    coreml_out = list(coreml_out_dict.values())[0]
    print('CoreML prediction pass OK')

    np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-2, atol=1e-1)
    print('Output matching OK')


def main(_argv):
    model, mlmodel = gen_mlmodel()
    test_models(model, mlmodel)
    mlmodel.save(FLAGS.output)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
