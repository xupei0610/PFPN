import os
from abc import ABCMeta

import tensorflow as tf
import numpy as np

from .distributed_model import AbstractDistributedWorker, DistributedModel
from .utils import Summary, build_worker_name

__all__ = [
    "LearnerModel"
]

class AbstractLearner(AbstractDistributedWorker):

    def __init__(self, _is_learner=None, queue_capacity=1, queue_batch_size=None, **kwargs):
        kwargs["_is_chief"] = False
        super().__init__(**kwargs)
        self.queue_capacity = queue_capacity
        if queue_batch_size is None:
            if self.episodic_training:
                self.queue_batch_size = self.batch_size
            else:
                self.queue_batch_size = 1
        else:
            self.queue_batch_size = queue_batch_size
        self.queue, self.dequeue = None, None
        self.is_learner = _is_learner
        assert(not (self.is_learner and self.is_evaluator))
    
    def init(self):
        if not self.is_learner: self.batch_size = 1 # resolve for the episodic training
        super().init()
        assert(self.target_net == self.global_net)
        if self.global_net.trainable:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.GLOBAL_NET_NAME)
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.GLOBAL_NET_NAME)
            with tf.variable_scope("optimizer"):
                def build_optimizer():
                    optimizer, grads_and_vars = self.build_optimizer(self.global_net, global_vars, global_vars)
                    self.opt_op = lambda : optimizer.apply_gradients(self.clip_grads(grads_and_vars), global_step=self.global_step)
                    if self.global_net.train_ops:
                        with tf.control_dependencies([self.opt_op()]):
                            self.opt_op = tf.group([_() for _ in self.global_net.train_ops])
                    else:
                        self.opt_op = self.opt_op()

                if update_ops and (self.off_policy or self.opt_epochs == 1):
                    with tf.control_dependencies(update_ops):
                        build_optimizer()
                else:
                    if update_ops:
                        worker = self
                        class UpdateHook(tf.train.SessionRunHook):
                            def before_run(self, run_context):
                                if worker.current_opt_epoch == 0:
                                    return tf.train.SessionRunArgs(update_ops)
                        self.hooks.append(UpdateHook())
                    build_optimizer()
                
            summary = None
            if self.logger:
                summary = self.build_summaries(self.target_net)
            if summary:
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = tf.no_op()

            dtypes, shapes, names = [], [], []
            for name, tensor in self.exp_tensors.items():
                dtypes.append(tensor.dtype)
                shape = list(tensor.shape)
                # shape[0] = None
                shapes.append(shape)
                names.append(name)
            if self.episodic_training:
                dtypes.append(tf.int32)
                shapes.append(())
                names.append(-1)
            self.queue = tf.PaddingFIFOQueue(self.queue_capacity, dtypes=dtypes, shapes=shapes, names=names, shared_name="pipe_queue")
            if self.queue_batch_size == 1:
                dequeue = self.queue.dequeue()
            else:
                dequeue = self.queue.dequeue_many(self.queue_batch_size)
            self.dequeue = lambda step_context: step_context.session.run(dequeue)

            enqueue_in = {
                n: tf.placeholder(dtype=t, shape=s) \
                for t, s, n in zip(self.queue.dtypes, self.queue.shapes, self.queue.names)
            }
            if self.episodic_training:
                self.enqueue_tensors = {k:v for k, v in enqueue_in.items()}
                del self.enqueue_tensors[-1]
                self.sequence_length_tensor = enqueue_in[-1]
            else:
                self.enqueue_tensors = enqueue_in
                self.sequence_length_tensor = None
            self.enqueue_op = self.queue.enqueue(enqueue_in)
        self.on_policy |= not self.is_learner
    
    def build_local_net(self, name):
        with tf.variable_scope(name):
            with tf.variable_scope("episode"):
                local_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name="episode")
                inc_local_episode = tf.assign_add(local_episode, 1, name="inc_episode")
        if name == self.name:
            self.inc_episode = lambda step_context: step_context.session.run([inc_local_episode, self.inc_global_episode])

    def work(self, sess):
        if self.is_learner and self.on_policy: self.setup_buffers = lambda : None
        self.request_stop = False
        self.before_work(sess)
        if self.is_learner:
            while not (sess.should_stop() or self.request_stop):
                self.exp = self.extract_exp(sess.run_step_fn(self.dequeue))
                if self.exp:
                    if self.sequence_length_tensor is not None:
                        self.episodic_train(sess)
                    else:
                        self.flat_train(sess)
                    self.after_train(sess)
        else:
            while not (sess.should_stop() or self.request_stop):
                state = self.before_episode(sess)
                terminal = False
                while not terminal:
                    action, *_ = self.sample_action(sess, state)
                    state, _, terminal, info = self.interact_with_env(sess, state, action)
                    if self.target_net.trainable and self.need_train(sess, terminal, state, info):
                        self.enqueue(sess)
                        self.after_train(sess)
                self.after_episode(sess, info)
        self.after_work(sess)

    def enqueue(self, sess):
        data = [self.exp[k] for k in self.exp_tensors.keys()]
        feed_dict = {
            self.enqueue_tensors[t]: #self.zero_pad(buf) if hasattr(buf, "__len__") else buf \
                buf \
                for t, buf in zip(self.enqueue_tensors, data)
        }
        if self.sequence_length_tensor is not None:
            for buf in data:
                if hasattr(buf, "__len__"):
                    # self.sequence_length.append(len(buf))
                    feed_dict[self.sequence_length_tensor] = len(buf) #self.sequence_length[-1]
                    break
        sess.run(self.enqueue_op, feed_dict=feed_dict)

    def extract_exp(self, dequeued_exp):
        if self.queue_batch_size != 1:
            for _, e in dequeued_exp.items():
                if len(e.shape) > 1:
                    # B, N, ... to BxN, ...
                    self.sample_counter = e.shape[0]*e.shape[1]
                    e.shape = (self.sample_counter, ) + e.shape[2:]
        if self.sequence_length_tensor is not None:
            self.sequence_length = dequeued_exp[-1]
        if self.on_policy:
            return dequeued_exp
        for k, e in dequeued_exp.items():
            for _ in e:
                self.exp[k].append(_)
            self.sample_counter = len(e)
        return self.exp

    
    def sync(self, sess):
        pass

    def clear_buffers(self):
        if not self.is_learner:
            return super().clear_buffers()

    def before_work(self, sess):
        if self.is_learner:
            episodic_training = self.episodic_training
            self.episodic_training = False  # disable sequence data initialization
            super().before_work(sess)
            self.episodic_training = episodic_training
        else:
            super().before_work(sess)

    @property
    def exp_tensors(self):
        return {
            "state": self.target_net.state,
            "action": self.target_net.action,
            "reward": self.target_net.reward
        }

    @property
    def exp_buffers(self):
        return ["state", "action", "reward"]

class LearnerModel(DistributedModel):
    
    def target_device(self, job_name, task_id, device):
        assert(job_name in ["learner", "actor"])
        if job_name == "learner":
            return device[0]
        if len(device) == 1:
            return device[0]
        return device[task_id+1]

    def dispatch(self, distributions, job_name, task_id, device):
        print("[SYSTEM] {} {} with PID {} started.".format(job_name, task_id, os.getpid()))
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = self.build_tf_sess_config()
        cluster = tf.train.ClusterSpec(distributions)
        server = tf.train.Server(cluster, job_name=job_name, task_index=task_id,
                                 config=config)
        if job_name == "ps":
            with tf.device("/job:ps/task:{}".format(task_id)):
                queue = tf.FIFOQueue(1, tf.int32, shared_name="done_queue{}".format(task_id))
            with tf.Session(server.target) as sess:
                for _ in range(cluster.num_tasks("worker")):
                    sess.run(queue.dequeue())
            print("PS {} {} work is done.".format(task_id, os.getpid()))
        else:            
            # Important!
            # By default, numpy shares the same random state among different sub-processing
            if self.seed is None: 
                random_seed = int.from_bytes(os.urandom(4), byteorder="big")
            else:
                random_seed = self.seed+task_id+1

            target_device = self.target_device(job_name, task_id, device)

            done_queues = []
            if "ps" in distributions:
                for i in range(cluster.num_tasks("ps")):
                    with tf.device("/job:ps/task:{}".format(i)):
                        done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name="done_queue{}".format(task_id)).enqueue(1))

            is_learner = job_name == "learner"
            is_chief = is_learner and task_id == 0
            is_chief_actor = not is_learner and task_id == 0

            n_learners = cluster.num_tasks("learner")
            if n_learners > 1:
                if is_learner:
                    target_learner = task_id
                else:
                    target_learner = task_id // n_learners
                d = tf.train.replica_device_setter(
                    "/job:learner/task:{}/{}".format(target_learner, self.target_device("learner", target_learner, device))
                )
            else:
                d = "/job:learner/task:0/{}".format(self.target_device("learner", task_id, device))
            with tf.device(d):
                worker = self.worker_wrapper(_job_name=job_name, _task_id=task_id,
                    _is_evaluator=False, _is_learner=is_learner, _is_chief=is_chief,
                    _log_dir=None if not self.debug or (self.debug != "all" and not is_chief and not is_chief_actor) else self.log_dir,
                    _max_samples=self.max_samples,
                    _env_wrapper=self.env_wrapper, _network_wrapper=self.network_wrapper, _seed=random_seed, render=False)
                worker.init()
                    
            if is_chief:
                for i in range(cluster.num_tasks("actor")):
                    with tf.device("/job:actor/task:{}/{}".format(i, self.target_device("actor", i, device))):
                        worker_name = build_worker_name("actor", i)
                        worker.build_local_net(worker_name)
            if not is_learner:
                with tf.device("/job:actor/task:{}/{}".format(task_id, target_device)):
                    worker.build_local_net(worker.name)
                    
            chief_hooks = []
            hooks = worker.hooks + [tf.train.StopAtStepHook(self.max_iterations)]
            
            saver = tf.train.Saver(max_to_keep=1)
            class EndNotificationHook(tf.train.SessionRunHook):
                def end(self, sess):
                    print("end notifier from ", worker.name)
                    worker.stop()
                    sess.run(done_queues)
            hooks.append(EndNotificationHook())
            scaffold = tf.train.Scaffold(saver=saver)
            with tf.train.MonitoredTrainingSession(server.target, is_chief, self.checkpoint_dir,
                                                   config=config, hooks=hooks, chief_only_hooks=chief_hooks,
                                                   scaffold=scaffold, 
                                                   save_checkpoint_steps=self.save_checkpoint_interval,
                                                   log_step_count_steps=5000, max_wait_secs=6000) as sess:
                try:
                    worker.work(sess)
                except KeyboardInterrupt:
                    worker.stop()
            print("{} {} work is done.".format(worker.name, os.getpid()))

