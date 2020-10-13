import tensorflow as tf
import numpy as np


def build_worker_name(job_name, task_id):
    return "{}_{}".format(job_name, task_id)


def load_checkpoint(sess, checkpoint_dir):
    cp = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()
    if cp and cp.model_checkpoint_path:  # pylint: disable=no-member
        saver.restore(sess, cp.model_checkpoint_path)  # pylint: disable=no-member
        print("Checkpoint Loaded", cp.model_checkpoint_path)  # pylint: disable=no-member
    return saver

def build_summaries(net, key="summaries"):
    summaries = []
    if hasattr(net, key):
        def log(key, val):
            if isinstance(val, dict):
                for k, v in val.items():
                    log("{}/{}".format(key, k), v)
            else:
                if hasattr(val, "get_shape") and len(val.shape) > 0:
                    summaries.append(tf.summary.histogram("{}".format(key), val))
                else:
                    summaries.append(tf.summary.scalar("{}".format(key), val))
        for key, val in getattr(net, key).items():
            log(key, val)
    return summaries

class Summary(tf.summary.FileWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_histogram(self, tag, values, step, bins=1000):
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(np.shape(values)))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(np.square(values)))
        bin_edges = bin_edges[1:]
        for edge in bin_edges: hist.bucket_limit.append(edge)
        for c in counts: hist.bucket.append(c)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.add_summary(summary, step)

    def add_scalar(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.add_summary(summary, step)


def normalize(data, offset, scale):
    return np.multiply(np.add(data, offset), scale)


def unnormalize(data, offset, scale):
    return np.subtract(np.divide(data, scale), offset)
