import tensorflow as tf

from .distributed_model import AbstractDistributedWorker, DistributedModel

__all__ = [
    "SyncModel"
]

class SyncWorker(AbstractDistributedWorker):

    def __init__(self, n_workers=1, **kwargs):
        super().__init__(**kwargs)
        self.n_workers = n_workers
        self.n_aggregate = self.n_workers #max(1, int(0.9*self.n_workers))
    
    def build_local_net(self, name):
        pull_weights = None
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.GLOBAL_NET_NAME)
        push_weights, sync_pushed_weights = [], []
        with tf.variable_scope(name):
            local_net = None
            if self.n_workers > 1 and self.global_net.local_update_variables:
                local_net = self.network_wrapper(self.env, True)
                local_net.init()
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
                with tf.variable_scope("sync"):
                    with tf.name_scope("pull_weights"):
                        local_vars, global_vars = self.build_sync_vars(name)
                        if local_vars:
                            pull_weights = [
                                tf.assign(lv, gv) for lv, gv in zip(local_vars, global_vars)
                            ]
                    if self.logger:
                        print("Sync vars:")
                        for v, v_ in zip(local_vars, global_vars):
                            print(v_.name, "->", v.name, v.shape)
                    with tf.name_scope("push_weights"):
                        for v in local_net.local_update_variables:
                            try:
                                for gv in global_vars:
                                    if self.GLOBAL_NET_NAME + "/" + v.name.split("/", 1)[1] in gv.name and v != gv:
                                        push_weights.append((v.read_value(), gv))
                                        sync_pushed_weights.append(tf.assign(v, gv.read_value()))
                                        if self.logger: print(v.name, "->", gv.name, v.shape)
                            except:
                                raise ValueError("Failed to find the target variable of local copy {}".format(v.name))
            with tf.variable_scope("episode"):
                local_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name="episode")
                inc_local_episode = tf.assign_add(local_episode, 1, name="inc_episode")

        if name == self.name:
            self.target_net = self.global_net
            self.local_net = local_net if local_net else self.global_net

            if not self.global_net.trainable: return

            self.inc_episode = lambda step_context: \
                step_context.session.run([inc_local_episode, self.inc_global_episode])

            with tf.name_scope("optimizer"):
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.GLOBAL_NET_NAME)
                def build_optimizer():
                    optimizer, grads_and_vars = self.build_optimizer(self.global_net, global_vars, global_vars)
                    grads_and_vars = self.clip_grads(grads_and_vars)
                    # if self.local_net != self.global_net: # async, shared network
                    if self.n_workers > 1:
                        if pull_weights:
                            self.pull_weights = lambda step_context: step_context.session.run(pull_weights)
                        if push_weights or self.global_net.train_ops:
                            grads_and_vars += push_weights
                            class SyncUpdateOpt(tf.train.Optimizer):
                                def __init__(self, optimizer, global_net):
                                    self.optimizer = optimizer
                                    self.global_net = global_net
                                def apply_gradients(self, grads_and_vars, global_step=None, name=None):
                                    grads_and_vars = list(grads_and_vars)
                                    def build_opt_op():
                                        op = lambda : self.optimizer.apply_gradients(grads_and_vars[:-len(push_weights)], global_step, name)
                                        if self.global_net.train_ops:
                                            with tf.control_dependencies([op()]):
                                                op = tf.group([_() for _ in self.global_net.train_ops])
                                        else:
                                            op = op()
                                        return op
                                    if push_weights:
                                        with tf.control_dependencies([tf.assign(v, v_) for v_, v in grads_and_vars[-len(push_weights):]]):
                                            op = build_opt_op()
                                    else:
                                        op = build_opt_op()
                                    return op
                            optimizer = SyncUpdateOpt(optimizer, self.global_net)
                        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                replicas_to_aggregate=self.n_aggregate, total_num_replicas=self.n_workers)
                        sync_replicas_hook = optimizer.make_session_run_hook(self.is_chief, num_tokens=self.n_workers)
                        self.hooks.append(sync_replicas_hook)
                    self.opt_op = lambda : optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
                    if self.n_workers <= 1 and self.global_net.train_ops:
                        with tf.control_dependencies([self.opt_op()]):
                            self.opt_op = tf.group([_() for _ in self.global_net.train_ops])
                    else:
                        self.opt_op = self.opt_op()

                if update_ops and (self.off_policy or self.opt_epochs == 1) \
                    and self.local_net == self.global_net:
                        with tf.control_dependencies(update_ops):
                            build_optimizer()
                else:
                    if update_ops and self.local_net == self.global_net:
                        class UpdateHook(tf.train.SessionRunHook):
                            def __init__(self, worker, update_ops):
                                self.worker = worker
                                self.update_ops = update_ops
                                super().__init__()
                            def before_run(self, run_context):
                                if self.worker.off_policy or self.worker.current_opt_epoch == 0:
                                    return tf.train.SessionRunArgs(self.update_ops)
                        self.hooks.append(UpdateHook(self, update_ops))
                    elif self.local_net != self.global_net:
                        if update_ops:
                            # locally run UPDATE_OPS
                            # sync update updated variables
                            # pull updated variables immediately for the next run of UPDATE_OPS
                            class LocalUpdateHookPre(tf.train.SessionRunHook):
                                def __init__(self, worker, update_ops):
                                    self.worker = worker
                                    self.update_ops = update_ops
                                    super().__init__()
                                def before_run(self, run_context):
                                    if self.worker.off_policy or self.worker.current_opt_epoch == 0:
                                        feed_dict = {}
                                        for k, v in run_context.original_args.feed_dict.items():
                                            try:
                                                holder = tf.get_default_graph().get_tensor_by_name(name + "/" + k.name.split("/", 1)[1])
                                                feed_dict[holder] = v
                                            except:
                                                pass
                                        return tf.train.SessionRunArgs(self.update_ops, feed_dict)
                            self.hooks.append(LocalUpdateHookPre(self, update_ops))
                        if sync_pushed_weights:
                            class LocalUpdateHookPost(tf.train.SessionRunHook):
                                def __init__(self, worker, sync_pushed_weights):
                                    self.worker = worker
                                    self.sync_pushed_weights = sync_pushed_weights
                                    super().__init__()
                                def after_run(self, run_context, run_values):
                                    if self.worker.global_net.train_ops or self.worker.off_policy or self.worker.current_opt_epoch == 0:
                                        run_context.session.run(self.sync_pushed_weights)
                            self.hooks.append(LocalUpdateHookPost(self, sync_pushed_weights))
                    build_optimizer()

            summary = None
            if self.logger:
                summary = self.build_summaries(self.target_net)
            if summary:
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = tf.no_op()


class SyncModel(DistributedModel):
    
    def worker_wrapper_kwargs(self, job_name, task_id, is_chief, cluster):
        return {
            "n_workers": cluster.num_tasks("worker")
        } 
