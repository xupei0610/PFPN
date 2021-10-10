import os
import inspect, functools

import pybullet as pb

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

class BulletEnv(object):

    COV_ENABLE_Y_AXIS_UP = pb.COV_ENABLE_Y_AXIS_UP
    POSITION_CONTROL = pb.POSITION_CONTROL
    TORQUE_CONTROL = pb.TORQUE_CONTROL
    URDF_MAINTAIN_LINK_ORDER = pb.URDF_MAINTAIN_LINK_ORDER

    def __init__(self, time_step=1.0/240, gravity=(0,0,-9.8), recorder="", **kwargs):
        self.info = {"gravity": gravity, "time_step": time_step, "log": {}}
        self.recorder = recorder
        self.video_logger = None
        self._render = False

    def __del__(self):
        self.close()

    def __getattr__(self, name):
        attr = getattr(pb, name)
        if inspect.isbuiltin(attr):
            attr = functools.partial(attr, physicsClientId=self.bullet_cid)
        return attr

    def _init(self):
        self.bullet_cid = pb.connect(pb.GUI if self._render else pb.DIRECT)
        pb.setTimeStep(self.info["time_step"], physicsClientId=self.bullet_cid)
        pb.setGravity(self.info["gravity"][0], self.info["gravity"][1], self.info["gravity"][2], physicsClientId=self.bullet_cid)
        if self.recorder:
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=self.bullet_cid)
            self.video_logger = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, self.recorder,
                                                     physicsClientId=self.bullet_cid)

    def render(self, mode="human"):
        self.close()
        self._render = True
        self._init()

    def do_simulation(self):
        pb.stepSimulation(physicsClientId=self.bullet_cid)
        
    def step(self):
        self.do_simulation()
    
    def close(self):
        if hasattr(self, "video_logger") and self.video_logger is not None:
            pb.stopStateLogging(self.video_logger, physicsClientId=self.bullet_cid)
        try:
            pb.disconnect(physicsClientId=self.bullet_cid)
        except pb.error:
            pass

    def seed(self, s):
        pass

    @property
    def time_step(self):
        return self.info["time_step"]

    @time_step.setter
    def time_step(self, val):
        if self.bullet_cid is not None:
            return pb.setTimeStep(val, physicsClientId=self.bullet_cid)
        self.info["time_step"] = val

    @property
    def gravity(self):
        return self.info["gravity"]

    @gravity.setter
    def gravity(self, val):
        if self.bullet_cid is not None:
            pb.setGravity(val[0], val[1], val[2], physicsClientId=self.bullet_cid)
        self.info["gravity"] = val
    
    @property
    def physics_engine_parameters(self):
        return pb.getPhysicsEngineParameters(physicsClientId=self.bullet_cid)

    def load_urdf(self, *args, **kwargs):
        return pb.loadURDF(*args, **kwargs, physicsClientId=self.bullet_cid)

    def configure_debug_visualizer(self, *args, **kwargs):
        return pb.configureDebugVisualizer(*args, **kwargs, physicsClientId=self.bullet_cid)

    # def set_gravity(self, *args, **kwargs):
    #     return pb.setGravity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def set_joint_motor_control_multi_dof(self, *args, **kwargs):
        return pb.setJointMotorControlMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_state_multi_dof(self, *args, **kwargs):
        return pb.getJointStateMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_joint_state_multi_dof(self, *args, **kwargs):
        return pb.resetJointStateMultiDof(*args, **kwargs, physicsClientId=self.bullet_cid)

    def set_joint_motor_control2(self, *args, **kwargs):
        return pb.setJointMotorControl2(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_joint_state(self, *args, **kwargs):
        return pb.resetJointState(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_num_joints(self, *args, **kwargs):
        return pb.getNumJoints(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_info(self, *args, **kwargs):
        return pb.getJointInfo(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_joint_state(self, *args, **kwargs):
        return pb.getJointState(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_link_state(self, *args, **kwargs):
        return pb.getLinkState(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def get_dynamics_info(self, *args, **kwargs):
        return pb.getDynamicsInfo(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def change_dynamics(self, *args, **kwargs):
        return pb.changeDynamics(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_aabb(self, *args, **kwargs):
        return pb.getAABB(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_base_position_and_orientation(self, *args, **kwargs):
        return pb.getBasePositionAndOrientation(*args, **kwargs, physicsClientId=self.bullet_cid)
    
    def reset_base_position_and_orientation(self, *args, **kwargs):
        # pybullet will reset base velocity to zero while reseting the base position and orientation
        i = kwargs["objectUniqueId"] if "objectUniqueId" in kwargs else args[0]
        v = pb.getBaseVelocity(i, physicsClientId=self.bullet_cid)
        res = pb.resetBasePositionAndOrientation(*args, **kwargs, physicsClientId=self.bullet_cid)
        pb.resetBaseVelocity(i, *v, physicsClientId=self.bullet_cid)
        return res
    
    def get_base_velocity(self, *args, **kwargs):
        return pb.getBaseVelocity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_base_velocity(self, *args, **kwargs):
        return pb.resetBaseVelocity(*args, **kwargs, physicsClientId=self.bullet_cid)

    def get_contact_points(self, *args, **kwargs):
        return pb.getContactPoints(*args, **kwargs, physicsClientId=self.bullet_cid)

    def calculate_mass_matrix(self, *args, **kwargs):
        return pb.calculateMassMatrix(*args, **kwargs, physicsClientId=self.bullet_cid)

    def calculate_inverse_dynamics(self, *args, **kwargs):
        return pb.calculateInverseDynamics(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_collision_shape(self, *args, **kwargs):
        return pb.createCollisionShape(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_visual_shape(self, *args, **kwargs):
        return pb.createVisualShape(*args, **kwargs, physicsClientId=self.bullet_cid)

    def create_multi_body(self, *args, **kwargs):
        return pb.createMultiBody(*args, **kwargs, physicsClientId=self.bullet_cid)

    def reset_debug_visualizer_camera(self, *args, **kwargs):
        return pb.resetDebugVisualizerCamera(*args, **kwargs, physicsClientId=self.bullet_cid)
