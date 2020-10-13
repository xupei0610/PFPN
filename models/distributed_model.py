import os, sys
import time, math
import multiprocessing
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import random

from .utils import build_worker_name, build_summaries, Summary


class AbstractDistributedWorker(ABC):

    GLOBAL_NET_NAME = "global_net"

    def __init__(self, _job_name, _task_id, _is_chief, _is_evaluator,
                _env_wrapper, _network_wrapper, _on_policy, 
                batch_size, unroll_length, 
                opt_epochs=1, episodic_training=False, render=False,
                _seed=None, _log_dir=None, _max_samples=None#, _max_iterations=None
                ):
        self.name = build_worker_name(_job_name, _task_id)
        self.job_name = _job_name
        self.task_id = _task_id
        self.is_chief = _is_chief
        self.is_evaluator = _is_evaluator
        self.seed = _seed
        self.log_dir = _log_dir

        # self.max_iterations = _max_iterations

        self.env_wrapper = _env_wrapper
        self.network_wrapper = _network_wrapper

        self.on_policy = _on_policy

        self.batch_size = batch_size
        self.unroll_length = unroll_length
        self.opt_epochs = opt_epochs
        self.episodic_training = episodic_training

        self.render = render

        self.env, self.global_net= None, None
        self.logger = None
        self.opt_op = None
        self.summary_op = None
        self.global_step = None
        self.init_ops = None
        self.pull_weights = None
        self.update_ops = None
    
        self.hooks = []

        self.max_samples = _max_samples

    @property
    def off_policy(self):
        return not self.on_policy

    @off_policy.setter
    def off_policy(self, v):
        self.on_policy = not v

    def set_env_seed(self, seed):
        if callable(getattr(self.env, "seed", None)): self.env.seed(seed)
        try:
            self.env.action_space.np_random.seed(seed)
        except:
            pass

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        tf.set_random_seed(seed)
        self.set_env_seed(seed)
        np.random.seed(seed)

    def init(self):
        self.env = self.env_wrapper(self.name, self.render)
        self.set_seed(self.seed)
        if self.log_dir:
            self.logger = Summary(os.path.join(self.log_dir, self.name))
        else:
            self.logger = None
        if callable(getattr(self.env, "init", None)):
            self.env.init()
        with tf.variable_scope("step"):
            self.global_step = tf.train.get_or_create_global_step()
        self.global_net, self.inc_global_episode, inc_total_samples = \
            self.build_global_net(not self.is_evaluator, self.network_wrapper, self.env)
        if inc_total_samples is None:
            self.inc_total_samples = None
        else:
            self.inc_total_samples = lambda step_context: step_context.session.run(inc_total_samples)
        self.local_net = self.target_net = self.global_net
        if self.global_net.trainable and hasattr(self.global_net, "init_ops") and self.global_net.init_ops:
            self.init_ops = lambda step_context: step_context.session.run(self.global_net.init_ops)
        
    @staticmethod
    def build_global_net(trainable, network_wrapper, env):
        with tf.variable_scope(AbstractDistributedWorker.GLOBAL_NET_NAME):
            net = network_wrapper(env, trainable)
            net.init()
            with tf.variable_scope("episode"):
                global_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name="episode")
                if trainable:
                    inc_global_episode = tf.assign_add(global_episode, 1, name="inc_episode")
                else:
                    inc_global_episode = None
            with tf.variable_scope("samples"):
                if trainable:
                    total_samples = tf.Variable(0, dtype=tf.int64, trainable=False, name="samples")
                    inc_total_samples = tf.assign_add(total_samples, 1, name="inc_samples")
                else:
                    inc_total_samples = None
        return net, inc_global_episode, inc_total_samples
    
    def build_local_net(self, name):
        pass    

    def build_sync_vars(self, local_net_name):
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.GLOBAL_NET_NAME)
        local_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, local_net_name)
        sync_local_vars, sync_global_vars = [], []
        for v in global_vars:
            try:
                for lv in local_vars:
                    if local_net_name + "/" + v.name.split("/", 1)[1] in lv.name and  lv != v:
                        sync_local_vars.append(lv)
                        sync_global_vars.append(v)
            except:
                print("[WARN] Failed to find the local copy of the global variable: {}".format(v))
        return sync_local_vars, sync_global_vars
    
    @abstractmethod
    def build_optimizer(self, local_net, local_vars, target_vars):
        raise NotImplementedError
    
    def build_summaries(self, target_net):
        return build_summaries(target_net)
    
    def stop(self):
        if self.logger:
            self.logger.flush()
        try:
            self.env.close()
        except:
            pass
        if not self.is_evaluator:
            print("[SYSTEM] Stopping {}".format(self.name))
    
    def work(self, sess):
        self.batch_size = None if self.batch_size is None else int(self.batch_size)
        self.unroll_length = None if self.unroll_length is None else int(self.unroll_length)
        self.opt_epochs = None if self.opt_epochs is None else int(self.opt_epochs)
        self.request_stop = False
        self.before_work(sess)
        while not (sess.should_stop() or self.request_stop):
            state = self.before_episode(sess)
            terminal = False
            while not terminal:
                action, *_ = self.sample_action(sess, state)
                state, _, terminal, info = self.interact_with_env(sess, state, action)
                if self.target_net.trainable:
                    need_train = self.need_train(sess, terminal, state, info)
                    if self.sequence_length is not None:
                        if terminal or self.episodic_terminal:
                            n = len(self.exp["state"]) - sum(self.sequence_length)
                            if self.unroll_length is not None:
                                remaining = self.unroll_length - n
                                for _, buf in self.exp.items():
                                    for __ in range(remaining):
                                        buf.append(np.zeros_like(buf[0]))
                            self.sequence_length.append(n)
                            self.sequence_terminal.append(terminal)
                    if need_train:
                        if self.sequence_length is not None:
                            if self.unroll_length is None:
                                max_n = max(self.sequence_length)
                                for i, n in enumerate(self.sequence_length):
                                    if n < max_n:
                                        remaining = max_n - n
                                        for _, buf in self.exp.items():
                                            for __ in range(remaining):
                                                buf.insert(i*max_n+n, np.zeros_like(buf[0]))
                            self.episodic_train(sess)
                        else:
                            self.flat_train(sess)
                        self.after_train(sess)
            self.after_episode(sess, info)
        self.after_work(sess)
    
    def sync(self, sess):
        if self.pull_weights:
            sess.run_step_fn(self.pull_weights)
            
    def setup_buffers(self):
        if self.exp_buffers is not None:
            if self.sequence_length is not None:
                buf_size = self.batch_size*self.unroll_length if self.unroll_length is not None else 4096
            else:
                buf_size = self.unroll_length if self.unroll_length is not None else 4096
            self.exp = {k: [None]*buf_size for k in self.exp_buffers}
            for _, v in self.exp.items(): v.clear()
            

    def clear_buffers(self):
        if self.on_policy:
            if self.exp_buffers is not None:
                for _, v in self.exp.items(): v.clear()
            if hasattr(self, "last_training_state"):
                self.sequence_length.clear()
                self.sequence_terminal.clear()
        self.sample_counter = 0
    
    def before_work(self, sess):
        if self.episodic_training:
            self.sequence_length = []
            self.sequence_terminal = []
            self.last_training_state = None
        else:
            self.sequence_length = None
        self.evaluator_counter = 0
        self.sample_counter = 0
        self.setup_buffers()
        if self.init_ops: sess.run_step_fn(self.init_ops)
        self.sync(sess)
    
    def after_work(self, sess):
        self.stop()

    def before_episode(self, sess):
        self.local_net.reset_running_state()
        if not self.evaluator_counter:
            self.total_reward, self.episode_step = 0.0, 0
        self.episode_buffer_stamp = 0
        return self.env.reset()

    def after_episode(self, sess, info):
        if self.is_evaluator:
            self.evaluator_counter += 1
            if self.evaluator_counter >= self.is_evaluator:
                self.episode_step /= self.evaluator_counter
                self.total_reward /= self.evaluator_counter
                summaries = [
                    tf.Summary.Value(tag="performance_test/reward", simple_value=self.total_reward),
                    tf.Summary.Value(tag="performance_test/reward_avg", simple_value=self.total_reward/self.episode_step),
                    tf.Summary.Value(tag="performance_test/frames", simple_value=self.episode_step)
                ]
                if hasattr(self, "total_samples"):
                    summaries.append(tf.Summary.Value(tag="performance_test/samples", simple_value=self.total_samples))
                    print("[PERFORM] Life Time: {}; Total Reward: {:.4f}; Avg Reward: {:.4f}; Step: {}; Samples: {}; {}".format(
                        self.episode_step, self.total_reward, self.total_reward/self.episode_step, self.global_step, self.total_samples,
                        time.strftime("%m-%d %H:%M:%S", time.localtime())
                    ))
                else:
                    print("[PERFORM] Life Time: {}; Total Reward: {:.4f}; Avg Reward: {:.4f}; Step: {}; {}".format(
                        self.episode_step, self.total_reward, self.total_reward/self.episode_step, self.global_step,
                        time.strftime("%m-%d %H:%M:%S", time.localtime())
                    ))
                self.summary = tf.Summary(value=summaries)
                # self.request_stop = True
                self.evaluator_counter = 0

    def sample_action(self, sess, state):
        return self.local_net.run(sess, state, self.update_ops)

    def interact_with_env(self, sess, state, action):
        self.sample_counter += 1
        self.episode_step += 1
        
        if hasattr(self.env.action_space, "low"):
            a = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        else:
            a = action

        state_, reward, terminal, info = self.env.step(a)
        
        self.exp["state"].append(state)
        self.exp["action"].append(action)
        self.exp["reward"].append(reward)
        if "episode_reward" in info:
            if info["episode_reward"] != 0:
                for i in range(self.episode_buffer_stamp, len(self.exp["reward"])):
                    self.exp["reward"][i] += info["episode_reward"]
                self.total_reward += (len(self.exp["reward"])-self.episode_buffer_stamp)*info["episode_reward"]
            self.episode_buffer_stamp = len(self.exp["reward"])
        
        self.total_reward += reward
        
        if self.inc_total_samples is not None:
            self.total_samples = sess.run_step_fn(self.inc_total_samples)

        return state_, reward, terminal, info

    def need_train(self, sess, terminal, last_state, info):
        if self.episodic_training:
            if terminal or self.episodic_terminal:
                return len(self.sequence_length)+1 >= self.batch_size
            return False
        if self.unroll_length is None:
            return terminal
        return self.sample_counter >= self.unroll_length
    
    @property
    def episodic_terminal(self):
        if hasattr(self, "sequence_length") and self.sequence_length is not None and self.unroll_length is not None:
            n = len(self.sequence_length)
            if n > 0: n *= max(self.sequence_length)
            return len(self.exp["state"]) - n >= self.unroll_length
        return False

    def after_train(self, sess):
        self.sync(sess)
        self.clear_buffers()
        if self.inc_total_samples is not None and self.max_samples is not None:
            if hasattr(self, "total_samples") and self.total_samples > self.max_samples:
                #self.request_stop = True
                for h in sess._hooks:
                    if isinstance(h, tf.train.StopAtStepHook):
                        h._last_step = self.global_step
                        break

    def flat_train(self, sess):
        n = len(self.exp["state"])
        exp_bak = self.exp
        if self.on_policy: # disposable experience replay buffer
            exp = {
                k: np.asarray(v, dtype=np.float32) for k, v in exp_bak.items()
            }
            ids = np.arange(n)
            self.current_opt_epoch = 0
            while self.current_opt_epoch < self.opt_epochs:
                np.random.shuffle(ids)
                if self.batch_size:
                    for s in range(0, n, self.batch_size):
                        e = s + self.batch_size
                        cand = ids[s:e]
                        self.exp = {
                            k: buf[cand] for k, buf in exp.items()
                        }
                        self.train(sess)
                else:
                    self.exp = {
                        k: buf[ids] for k, buf in exp.items()
                    }
                    self.train(sess)
                self.current_opt_epoch += 1
        elif n >= self.batch_size:
            for _ in range(self.opt_epochs if self.opt_epochs else self.sample_counter):
                ids = np.random.choice(n, self.batch_size)
                self.exp = {
                    k: list(map(v.data.__getitem__, ids)) for k, v in exp_bak.items()
                }
                self.train(sess)
        self.exp = exp_bak

    def episodic_train(self, sess):
        assert(self.on_policy)
        for _ in range(self.opt_epochs):
            #self.batch_size x (self.unroll_length or self.sequence_length[0:])
            if not hasattr(self, "last_training_state") or self.last_training_state is None or self.batch_size > 1:
                # the training state should be reset to zero state
                # for multiple-batch cases, since some previous training
                # batches may lead the terminal state
                self.target_net.reset_training_state()
            else:
                self.target_net.reset_training_state(self.last_training_state)
            self.train(sess)
        if hasattr(self, "last_training_state") and self.batch_size <= 1:
            if len(self.sequence_terminal) < 1 or self.sequence_terminal[-1]:
                self.last_training_state = None
            else:
                self.last_training_state = self.target_net.training_state

    def train(self, sess):
        if sess.should_stop() or self.request_stop:
            return None, None
        ops = self.train_ops
        loss, result = self.target_net.train(
            sess, self.opt_op, ops, *self.train_args
        )
        summary = result[ops.index(self.summary_op)]
        global_step = result[ops.index(self.global_step)]
        if self.logger and summary:
            self.logger.add_summary(summary, global_step)
        # if self.max_iterations and global_step >= self.max_iterations:
        #     self.request_stop = True
        return loss, result
    
    @property
    def exp_buffers(self):
        return ["state", "action", "reward"]

    @property
    def train_ops(self):
        return [self.summary_op, self.global_step]
    
    @property
    def train_args(self):
        args = [
            self.exp["state"], self.exp["action"],
            self.exp["reward"]
        ]
        if self.sequence_length is not None:
            args.append(self.sequence_length)
        return args


class DistributedModel(object):

    def __init__(self, worker_wrapper, env_wrapper, network_wrapper,
                 checkpoint_dir, save_checkpoint_interval, max_iterations, device,
                 log_dir=None, debug="chief", max_samples=None,
                 use_evaluator=True, seed=None):
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self.worker_wrapper = worker_wrapper
        self.env_wrapper = env_wrapper
        self.network_wrapper = network_wrapper

        self.log_dir = log_dir
        self.debug = debug
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoint_interval = save_checkpoint_interval
        self.max_iterations = max_iterations
        self.max_samples = max_samples
        self.device = device.split(";")

        self.use_evaluator = use_evaluator
        self.seed = seed

    def start(self, distributions=None):
        if distributions is None:
            self.test()
        else:
            self.train(distributions)
    
    def test(self):
        worker = self.worker_wrapper(_job_name="evaluator", _task_id=0, _is_chief=False, _is_evaluator=1,
                                     _env_wrapper=self.env_wrapper, _network_wrapper=self.network_wrapper,
                                     _seed=self.seed,
                                     render=True)
        worker.init()
        while True:
            with tf.train.SingularMonitoredSession(checkpoint_dir=self.checkpoint_dir) as sess:
                worker.work(sess)
    
    def evaluate(self, n=10, evaluation_cond=lambda : True):
        from tensorflow.python import pywrap_tensorflow

        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        worker = self.worker_wrapper(_job_name="evaluator", _task_id=0, _is_chief=False, _is_evaluator=n,
                                     _env_wrapper=self.env_wrapper, _network_wrapper=self.network_wrapper,
                                     _seed=self.seed, _log_dir=self.log_dir, _max_samples=self.max_samples,
                                     render=False)
        worker.init()
        rng = np.random.RandomState(self.seed)
        class Evaluator(tf.train.SessionRunHook):
            def __init__(self, checkpoint_dir):
                self.lastest_checkpoint = None
                self.saver = tf.train.Saver()
                self.checkpoint_dir = checkpoint_dir

            def after_create_session(self, session, coord):
                self.sess = session

            def before_run(self, run_context):
                if self.lastest_checkpoint is None or (worker.evaluator_counter == 0 and worker.episode_step == 0):
                    if self.lastest_checkpoint is not None and worker.logger:
                        worker.logger.add_summary(worker.summary, worker.global_step)
                        worker.logger.flush()

                    checkpoint = None
                    while checkpoint is None and evaluation_cond():
                        time.sleep(30)
                        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                    while self.lastest_checkpoint == checkpoint and evaluation_cond():
                        time.sleep(30)
                        checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

                    if checkpoint is None or self.lastest_checkpoint == checkpoint:
                        worker.request_stop = True
                        # worker.after_episode = lambda *args, **kwargs: run_context.request_stop()
                        # run_context.request_stop()
                    else:
                        self.saver.restore(self.sess, checkpoint)
                        self.lastest_checkpoint = checkpoint
                        worker.global_step = int(self.lastest_checkpoint.split('-')[-1])
                        reader = pywrap_tensorflow.NewCheckpointReader(self.lastest_checkpoint)
                        worker.total_samples = reader.get_tensor(worker.GLOBAL_NET_NAME+"/samples/samples")
                        worker.set_env_seed(rng.randint(1))

        h = Evaluator(self.checkpoint_dir)
        config = self.build_tf_sess_config()
        with tf.train.SingularMonitoredSession(hooks=[h], config=config) as sess:
            sess.run_step_fn = None
            worker.work(sess)

    @staticmethod
    def build_tf_sess_config(**kwargs):
        return tf.ConfigProto(
                #inter_op_parallelism_threads=1,
                #intra_op_parallelism_threads=1,
                # log_device_placement=True,
                **kwargs,
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
    
    def train(self, distributions):
        n_workers = sum([len(v) for _, v in distributions.items()])
        if len(self.device) == 1:
            self.device = self.device*n_workers
        workers = []
        for job, d in distributions.items():
            for task_id in range(len(d)):
                workers.append(multiprocessing.Process(
                    target=self.dispatch,
                    args=(distributions, job, task_id, self.device)
                ))
        for w in workers: w.start()
        print("==== Process ID List ===")
        for w in workers:
            print(w.pid)
        print("========================")

        try:
            if self.use_evaluator:
                self.evaluate(evaluation_cond=lambda : all(w.is_alive() for w in workers))
                print("evaluator work is done")
                for w in workers:
                    print(w.pid, w.is_alive())
            else:
                for w in workers: w.join()
        except KeyboardInterrupt:
            pass
        finally:
            print("closing worker threads...")
            for w in workers:
                try:
                    print("finishing ", w.pid)
                    w.terminate()
                    w.join()
                except:
                    pass
        print("training is done")
    

    def dispatch(self, distributions, job_name, task_id, device):
        print("[SYSTEM] {} {} with PID {} started.".format(job_name, task_id, os.getpid()))
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = self.build_tf_sess_config()
        cluster = tf.train.ClusterSpec(distributions)
        server = tf.train.Server(cluster, job_name=job_name, task_index=task_id, config=config)
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
                seed = int.from_bytes(os.urandom(4), byteorder="big")
            else:
                seed = self.seed+task_id+1

            target_device = self.target_device(job_name, task_id, device)
            # if ":" in target_device and target_device.split(":")[-2][-3:].lower() == "gpu":
            #     os.environ["CUDA_VISIBLE_DEVICES"] = target_device.split(":")[-1]
            #     target_device = "device:gpu:0"
            # else:
            #     os.environ["CUDA_VISIBLE_DEVICES"] = ""

            done_queues = []
            if "ps" in distributions:
                for i in range(cluster.num_tasks("ps")):
                    with tf.device("/job:ps/task:{}".format(i)):
                        done_queues.append(tf.FIFOQueue(1, tf.int32, shared_name="done_queue{}".format(i)).enqueue(1))

            is_chief = (task_id == 0)
            n_workers = cluster.num_tasks(job_name)
            worker = self.worker_wrapper(_job_name=job_name, _task_id=task_id, _is_chief=is_chief, _is_evaluator=False,
                                         _env_wrapper=self.env_wrapper, _network_wrapper=self.network_wrapper, _seed=seed,
                                         _log_dir=None if not self.debug or (self.debug != "all" and not is_chief) else self.log_dir,
                                         _max_samples=self.max_samples,
                                         **self.worker_wrapper_kwargs(job_name, task_id, is_chief, cluster),
                                         render=False)
            with tf.device(tf.train.replica_device_setter(cluster=cluster,
                worker_device="/job:worker/task:{}/{}".format(task_id, target_device)
            )):
                worker.init()
            
            if is_chief:
                for i in range(n_workers):
                    with tf.device("/job:worker/task:{}/{}".format(i, target_device if i == task_id else self.target_device(job_name, i, device))):
                        worker_name = build_worker_name(job_name, i)
                        worker.build_local_net(worker_name)
            else:
                with tf.device("/job:worker/task:{}/{}".format(task_id, target_device)):
                    worker.build_local_net(worker.name)

            chief_hooks = []
            hooks = worker.hooks + [tf.train.StopAtStepHook(last_step=self.max_iterations)]
            
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
                                                   save_checkpoint_steps=self.save_checkpoint_interval,
                                                   log_step_count_steps=5000, scaffold=scaffold, max_wait_secs=60) as sess:
                try:
                    worker.work(sess)
                except KeyboardInterrupt:
                    worker.stop()
                    sess.run(done_queues)
            print("{} {} work is done.".format(worker.name, os.getpid()))

    def worker_wrapper_kwargs(self, job_name, task_id, is_chief, cluster):
        return {} 
    
    def target_device(self, job_name, task_id, device):
        if job_name == "ps": return None
        tar = device[0] if len(device) == 1 else device[task_id]
        tar = tar.upper()
        if tar.startswith("GPU"):
            return "device:GPU:{}".format(tar[(4 if tar[3]==":" else 3):])
        else:
            assert(tar.startswith("GPU") or tar == "CPU" or tar == "CPU:0")
            return "cpu:0"
