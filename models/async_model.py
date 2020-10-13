
import tensorflow as tf

from .distributed_model import AbstractDistributedWorker, DistributedModel

__all__ = [
    "AsyncModel"
]


class AsyncWorker(AbstractDistributedWorker):

    def __init__(self, n_workers=1, **kwargs):
        super().__init__(**kwargs)
        self.n_workers = n_workers
        
    def build_local_net(self, name):
        pull_weights = None
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.GLOBAL_NET_NAME)
        push_weights = []
        with tf.variable_scope(name):
            if self.n_workers > 1:
                local_net = self.network_wrapper(self.env, True)
                local_net.init()
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=name)
                with tf.variable_scope("sync"):
                    with tf.name_scope("pull_weights"):
                        local_vars, global_vars = self.build_sync_vars(name)
                        if local_vars:
                            pull_weights =[
                                tf.assign(lv, gv) for lv, gv in zip(local_vars, global_vars)
                            ]
                    if self.logger:
                        print("Sync vars:")
                        for v, v_ in zip(local_vars, global_vars):
                            print(v_.name, "->", v.name, v.shape)
                    with tf.variable_scope("push_weights"):
                        for v in local_net.local_update_variables:
                            try:
                                for gv in global_vars:
                                    if self.GLOBAL_NET_NAME + "/" + v.name.split("/", 1)[1] in gv.name and v != gv:
                                        old_v = tf.Variable(initial_value=tf.zeros_like(gv), trainable=False, dtype=gv.read_value().dtype)
                                        push_weights.append((gv, v, old_v))
                                        pull_weights.append(tf.assign(old_v, gv))
                                        if self.logger: print(v.name, "->", gv.name, v.shape)
                            except:
                                raise ValueError("Failed to find the target variable of local copy {}".format(v.name))
            with tf.variable_scope("step"):
                local_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="step")
                inc_global_step = tf.assign_add(self.global_step, 1, name="inc_global_step")

            with tf.variable_scope("episode"):
                local_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name="episode")
                inc_local_episode = tf.assign_add(local_episode, 1, name="inc_episode")

        if name == self.name:
            self.target_net = self.local_net = local_net if self.n_workers > 1 else self.global_net
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.GLOBAL_NET_NAME)
            if self.local_net == self.global_net:
                local_vars = global_vars
            else:
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            if not self.global_net.trainable: return

            self.inc_episode = lambda step_context: \
                step_context.session.run([inc_local_episode, self.inc_global_episode])
            
            with tf.name_scope("optimizer"):
                def build_optimizer():
                    optimizer, grads_and_vars = self.build_optimizer(self.local_net, local_vars, global_vars)

                    if pull_weights:
                        self.pull_weights = lambda step_context: step_context.session.run(pull_weights)
                    
                    def opt_op():
                        op = [
                            optimizer.apply_gradients(self.clip_grads(grads_and_vars), global_step=local_step),
                            inc_global_step
                        ]
                        if push_weights:
                            with tf.control_dependencies([tf.assign_add(gv, v-old_v) for gv, v, old_v in push_weights] + op):
                                op = tf.group([tf.assign(old_v, v) for gv, v, old_v in push_weights])
                        else:
                            op = tf.group(op)
                        return op
                    
                    if self.global_net.train_ops:
                        with tf.control_dependencies([opt_op()]):
                            self.opt_op = tf.group([_() for _ in self.global_net.train_ops])
                    else:
                        self.opt_op = opt_op()
                        

                if update_ops and (self.off_policy or self.opt_epochs == 1):
                    with tf.control_dependencies(update_ops):
                        build_optimizer()
                else:
                    if update_ops:
                        class UpdateHook(tf.train.SessionRunHook):
                            def __init__(self, worker, update_ops):
                                self.worker = worker
                                self.update_ops = update_ops
                                super().__init__()
                            def before_run(self, run_context):
                                if self.worker.current_opt_epoch == 0:
                                    return tf.train.SessionRunArgs(self.update_ops)
                        self.hooks.append(UpdateHook(self, update_ops))
                    build_optimizer()

            summary = None
            if self.logger:
                summary = self.build_summaries(self.target_net)
            if summary:
                self.summary_op = tf.summary.merge(summary)
            else:
                self.summary_op = tf.no_op()


class AsyncModel(DistributedModel):
    def worker_wrapper_kwargs(self, job_name, task_id, is_chief, cluster):
        return {
            "n_workers": cluster.num_tasks("worker")
        } 
