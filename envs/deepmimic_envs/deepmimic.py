import os, time
import math
import numpy as np

from . import bullet_env, agent
from .gym_api import spaces
from .bullet_env import BulletEnv
from .utils import so_fb_butter_lpf
from .utils import quat2euler_zyx, quat2axis_angle, axis_angle2quat
from .utils import quatmultiply, quatconj, quatdiff, quatdiff_rel
from .utils import lerp, slerp
from .utils import rotate_vector, quat2mat

MOTION_DIR = os.path.join(bullet_env.DATA_DIR, "motions")

import json
class ReferenceMotionHumanoid(object):

    def __init__(self, agent, action):
        self.agent = agent
        self.motion_file = os.path.join(MOTION_DIR, "humanoid3d_{}.txt".format(action))

    def init(self):
        def linear_vel(p0, p1, delta_t):
            if hasattr(p0, "__len__"):
                return [(v1-v0)/delta_t for v0, v1 in zip(p0, p1)]
            return (p1-p0)/delta_t

        def angular_vel(q0, q1, delta_t):
            axis, angle = quat2axis_angle(quatdiff(q0, q1))
            angle /= delta_t
            return [angle*a for a in axis]

        def angular_vel_rel(q0, q1, delta_t):
            axis, angle = quat2axis_angle(quatdiff_rel(q0, q1))
            angle /= delta_t
            return [angle*a for a in axis]

        with open(self.motion_file) as f:
            data = json.load(f)
            motion = data["Frames"]
            try:
                self.contactable_links = data["ContactableBodies"]
            except:
                self.contactable_links = None
        for m in motion:
            # change the order of quaternion from (w, x, y, z) to (x, y, z, w)
            for i in [4, 8, 12, 16, 21, 25, 30, 35, 39]:
                m[i], m[i+1], m[i+2], m[i+3] = m[i+1], m[i+2], m[i+3], m[i]
            if self.agent.env.up_dir == 2: # y-axis up to z-axis up
                m[2], m[3] = -m[3], m[2]
                for i in [4, 8, 12, 16, 21, 25, 30, 35, 39]:
                    m[i+1], m[i+2] = -m[i+2], m[i+1]
                m[20], m[29], m[34], m[43] = -m[20], -m[29], -m[34], -m[43]
        indices = {
            "chest": slice(8, 12),
            "head": slice(12, 16),
            "right_thign": slice(16, 20),
            "right_shin": slice(20, 21),
            "right_foot": slice(21, 25),
            "right_upper_arm": slice(25, 29),
            "right_forearm": slice(29, 30),
            "left_thign": slice(30, 34),
            "left_shin": slice(34, 35),
            "left_foot": slice(35, 39),
            "left_upper_arm": slice(39, 43),
            "left_forearm": slice(43, 44)
        }
        ind_base_pos = slice(1, 4)
        ind_base_orient = slice(4, 8)

        self.duration = 0 
        self.motion = []
        for n in range(len(motion)-1):
            m = motion[n]
            m_ = motion[n+1]
            dt = m[0]
            pose = {
                "time": self.duration,
                "base_position": m[ind_base_pos],
                "base_orientation": m[ind_base_orient],
                "base_linear_velocity": linear_vel(m[ind_base_pos], m_[ind_base_pos], dt),
                "base_angular_velocity": angular_vel(m[ind_base_orient], m_[ind_base_orient], dt),
            }
            for name, ind in indices.items():
                pose[name] = {
                    "position": m[ind],
                }
                assert(hasattr(m[ind], "__len__") and (len(m[ind]) == 4 or len(m[ind]) == 1))
                if len(m[ind]) == 4:
                    pose[name]["velocity"] = angular_vel_rel(m[ind], m_[ind], dt)
                else:
                    pose[name]["velocity"] = linear_vel(m[ind], m_[ind], dt)
            self.motion.append(pose)
            self.duration += dt
        m = motion[-1]
        pose = {
            "time": self.duration,
            "base_position": m[ind_base_pos],
            "base_orientation": m[ind_base_orient],
            "base_linear_velocity": [v for v in self.motion[-1]["base_linear_velocity"]],
            "base_angular_velocity": [v for v in self.motion[-1]["base_angular_velocity"]],
        }
        for name, ind in indices.items():
            pose[name] = {
                "position": m[ind],
                "velocity": [v for v in self.motion[-1][name]["velocity"]]
            }
        self.motion.append(pose)
        
        fs = 1.0 / self.motion[1]["time"]
        fc = 6.0
        for i in range(3):
            vel = so_fb_butter_lpf([p["base_linear_velocity"][i] for p in self.motion], fs, fc)
            for v, p in zip(vel, self.motion): p["base_linear_velocity"][i] = v
            vel = so_fb_butter_lpf([p["base_angular_velocity"][i] for p in self.motion], fs, fc)
            for v, p in zip(vel, self.motion): p["base_angular_velocity"][i] = v
        for name in indices.keys():
            for i in range(len(self.motion[0][name]["velocity"])):
                vel = so_fb_butter_lpf([p[name]["velocity"][i] for p in self.motion], fs, fc)
                for v, p in zip(vel, self.motion): p[name]["velocity"][i] = v

        self.base_origin_pos_offset = [-p for p in self.motion[0]["base_position"]]
        self.base_origin_pos_offset[self.agent.env.up_dir] = 0
        origin_heading = self.agent.env.orient2heading(self.motion[0]["base_orientation"])
        ref_dir = [0, 0, 0]
        ref_dir[self.agent.env.up_dir] = 1
        self.base_origin_orient_offset = axis_angle2quat(ref_dir, -origin_heading)
        
        self.base_pos_offset = [0, 0, 0]
        self.base_orient_offset = [0, 0, 0, 1]
        
    def set_sim_time(self, time, mirror=False, with_base_offset=True):
        return self.dummy_pose(time, mirror, with_base_offset, compute_forward_kinematics=False)

    def reset(self):
        self.set_base_position_offset([0, 0, 0])
        self.set_base_orientation_offset([0, 0, 0, 1])

    def set_base_position_offset(self, offset):
        self.base_pos_offset = [v for v in offset]
    
    def set_base_orientation_offset(self, offset):
        self.base_orient_offset = [v for v in offset]

    def sync(self, time, tar_position=None, tar_orientation=None):
        ref_pose = self.dummy_pose(time, with_base_offset=False)
        if tar_position is not None:
            offset = [v-v0 for v, v0 in zip(tar_position, ref_pose["base_position"])]
            offset[self.agent.env.up_dir] = 0
            self.set_base_position_offset(offset)
        if tar_orientation is not None:
            heading = self.agent.env.orient2heading(tar_orientation)
            ref_heading = self.agent.env.orient2heading(ref_pose["base_orientation"])
            ref_dir = [0, 0, 0]
            ref_dir[self.agent.env.up_dir] = 1
            offset = axis_angle2quat(ref_dir, heading-ref_heading)
            self.set_base_orientation_offset(offset)  

    def dummy_pose(self, time, mirror=False, with_base_offset=True, compute_forward_kinematics=False):
        time = math.fmod(time, self.duration)
        if time < 0: time += self.duration
        # get two ref frames covering the specified time
        f0 = None
        for i in range(len(self.motion)):
            if self.motion[i]["time"] <= time:
                f0 = i
            if self.motion[i]["time"] > time:
                break
        f1 = f0 + 1
        if f1 > len(self.motion) - 1: f1 = f0 
        f0 = self.motion[f0]
        f1 = self.motion[f1]
        # lerp two frames
        dt = f1["time"] - f0["time"]
        frac = (time - f0["time"]) / dt
        pose = {
            "base_position": lerp(f0["base_position"], f1["base_position"], frac),
            "base_orientation": slerp(f0["base_orientation"], f1["base_orientation"], frac),
            "base_linear_velocity": lerp(f0["base_linear_velocity"], f1["base_linear_velocity"], frac),
            "base_angular_velocity": lerp(f0["base_angular_velocity"], f1["base_angular_velocity"], frac)
        }
        for name, joints in self.agent.joint_groups.items():
            if name not in f0: continue
            if len(f0[name]["position"]) == 4:
                lp = slerp
            else:
                lp = lerp
            p = lp(f0[name]["position"], f1[name]["position"], frac)
            v = lerp(f0[name]["velocity"], f1[name]["velocity"], frac)
            if len(joints) == 1:
                pose[joints[0]] = {"position": p, "velocity": v}
            elif len(joints) == 3 and len(p) == 4:
                # from extrinsic to intrinsic euler frame
                # we assume that the hinge joints are stacked in order of around the axis z, y, x
                yaw, pitch, roll = quat2euler_zyx(p)
                vx, vy, vz = rotate_vector(p, v)
                cx, sx = math.cos(roll), math.sin(roll)
                cy, sy = math.cos(pitch), math.sin(pitch)
                cz, sz = math.cos(yaw), math.sin(yaw)
                vx_ = (cz*vx + sz*vy)/cy
                vy_ = cz*vy - sz*vx
                vz_ = ((cy*vz + sy*vx)*cz + sy*vy*sz)/cy
                pose[joints[0]] = {"position": yaw, "velocity": vz_}
                pose[joints[1]] = {"position": pitch, "velocity": vy_}
                pose[joints[2]] = {"position": roll, "velocity": vx_}
            else:
                assert(len(joints) == 1 or (len(joints) == 3 and len(f0[name]["position"]) == 4))

        # normalize the pose such that
        # the initial pose is heading along x-axis and located at the original of the horizontal plane
        # pose["base_position"] = [p0+off for p0, off in zip(pose["base_position"], self.base_orient_offset)]
        # pose["base_orientation"] = quatmultiply(self.base_origin_orient_offset, pose["base_orientation"])
        # pose["base_linear_velocity"] = rotate_vector(self.base_origin_orient_offset, pose["base_linear_velocity"])
        # pose["base_angular_velocity"] = rotate_vector(self.base_origin_orient_offset, pose["base_angular_velocity"])
        
        # mirror
        if mirror:
            if self.agent.env.up_dir == 1: # y axis is up
                u, v = 0, 1
                w = 2
            else:
                assert(self.agent.env.up_dir == 2) # z axis is up
                u, v = 0, 2
                w = 1
            pose["base_position"][w] *= -1
            pose["base_linear_velocity"][w] *= -1
            pose["base_orientation"][u] *= -1
            pose["base_orientation"][v] *= -1
            pose["base_angular_velocity"][u] *= -1
            pose["base_angular_velocity"][v] *= -1
            for name, joints in self.agent.joint_groups.items():
                if name not in f0: continue
                for jid in joints:
                    info = self.agent.joint_info(jid)
                    joint_type = info[2]
                    if joint_type == self.agent.env.JOINT_SPHERICAL:
                        pose[jid]["position"][u] *= -1
                        pose[jid]["position"][v] *= -1
                        pose[jid]["velocity"][u] *= -1
                        pose[jid]["velocity"][v] *= -1
                    elif joint_type == self.agent.env.JOINT_REVOLUTE:
                        axis = info[13]
                        if axis[u] != 0 or axis[v] != 0:
                            pose[jid]["position"] *= -1
                            pose[jid]["velocity"] *= -1
                    else:
                        assert(joint_type in [self.agent.env.JOINT_REVOLUTE, self.agent.env.JOINT_SPHERICAL])
            for r, l in self.agent.joints_pairs:
                pose[self.agent.joints[l]], pose[self.agent.joints[r]] = pose[self.agent.joints[r]], pose[self.agent.joints[l]]

        # offset the pose
        if with_base_offset:
            pose["base_position"] = [p0+off for p0, off in zip(pose["base_position"], self.base_pos_offset)]
            pose["base_orientation"] = quatmultiply(self.base_orient_offset, pose["base_orientation"])
            pose["base_linear_velocity"] = rotate_vector(self.base_orient_offset, pose["base_linear_velocity"])
            pose["base_angular_velocity"] = rotate_vector(self.base_orient_offset, pose["base_angular_velocity"])
        
        # compute forward kinematics and velocity
        if compute_forward_kinematics:
            updated = [-1]
            while True:
                updated_flag = False
                for jid in range(self.agent.n_joints):
                    if jid in updated: continue
                    info = self.agent.joint_info(jid)
                    pid = info[16]
                    if pid not in updated: continue
                    
                    joint_type = info[2]
                    joint_pos_on_parent, joint_orient_on_parent = info[14], info[15]

                    if pid == -1:
                        p_pos = pose["base_position"]
                        p_orient = pose["base_orientation"]
                        p_linear_vel = pose["base_linear_velocity"]
                        p_angular_vel = pose["base_angular_velocity"]
                    else:
                        p_pos = pose[pid]["world_position"]
                        p_orient = pose[pid]["world_orientation"]
                        p_linear_vel = pose[pid]["world_linear_velocity"]
                        p_angular_vel = pose[pid]["world_angular_velocity"]
                    
                    if jid not in pose:
                        if joint_type == self.agent.env.JOINT_FIXED:
                            pose[jid] = { "position": (), "velocity": () }
                        else:
                            pose[jid] = { "position": (0,), "velocity": (0,) }

                    if joint_type == self.agent.env.JOINT_REVOLUTE:
                        axis = info[13]
                        angle = pose[jid]["position"][0] if hasattr(pose[jid]["position"], "__len__") else pose[jid]["position"]
                        joint_rot = axis_angle2quat(axis, angle)
                        joint_orient_on_parent = quatmultiply(joint_orient_on_parent, joint_rot)
                        joint_vel = np.multiply(axis, pose[jid]["velocity"])
                    elif joint_type == self.agent.env.JOINT_SPHERICAL:
                        joint_rot = pose[jid]["position"]
                        joint_vel = pose[jid]["velocity"]
                        joint_orient_on_parent = quatmultiply(joint_orient_on_parent, joint_rot)
                    elif joint_type != self.agent.env.JOINT_FIXED:
                        raise NotImplementedError
                    else:
                        joint_vel = [0, 0, 0]
                    link_center_on_joint, link_orient_on_joint = self.agent.link_state(jid)[2:4]
                    
                    pose[jid]["world_link_frame_position"] = np.add(
                        p_pos,
                        rotate_vector(p_orient, joint_pos_on_parent)
                    )
                    pose[jid]["world_link_frame_orientation"] = quatmultiply(
                        p_orient, joint_orient_on_parent
                    )

                    pose[jid]["world_position"] = np.add(
                        pose[jid]["world_link_frame_position"],
                        rotate_vector(pose[jid]["world_link_frame_orientation"], link_center_on_joint)
                    )
                    pose[jid]["world_orientation"] = quatmultiply(
                        pose[jid]["world_link_frame_orientation"], link_orient_on_joint
                    )
                    
                    pose[jid]["world_angular_velocity"] = np.add(
                        p_angular_vel,
                        rotate_vector(pose[jid]["world_link_frame_orientation"], joint_vel)
                    )

                    world_link_frame_linear_velocity = np.add(
                        p_linear_vel,
                        np.cross(p_angular_vel, rotate_vector(p_orient, joint_pos_on_parent))
                    )
                    pose[jid]["world_linear_velocity"] = np.add(
                        world_link_frame_linear_velocity,
                        np.cross(pose[jid]["world_angular_velocity"], rotate_vector(pose[jid]["world_link_frame_orientation"], link_center_on_joint))
                    )

                    updated_flag = True
                    updated.append(jid)
                assert(updated_flag)
                if len(updated) == self.agent.n_joints + 1: break

        return pose



class DeepMimic(BulletEnv):
    AGENT = agent.Humanoid

    def unnormalize_action(self, a):
        return a
    
    def __init__(self, **kwargs):
        def arg_parse(name, def_val):
            return kwargs[name] if kwargs is not None and name in kwargs else def_val
        
        self.fps = arg_parse("fps", 30.0)
        self.frame_skip = arg_parse("frame_skip", 20)
        kwargs["time_step"] = arg_parse("time_step", 1.0/(self.fps*self.frame_skip))
        super().__init__(**kwargs)
        self.control_mode = arg_parse("control_mode", "spd")
        assert(self.control_mode in ["position", "torque", "spd"])

        self.agent = self.AGENT(self)
        self.ground = None
        self.ref_motion = ReferenceMotionHumanoid(self.agent, arg_parse("action", "walk"))
        self.agent.use_spd = self.control_mode == "spd"
        self.control_mode = self.TORQUE_CONTROL if self.control_mode == "torque" else self.POSITION_CONTROL

        self.observation_space = spaces.Box()
        self.action_space = spaces.Box()
    
        self.random_init_pose = True
        self.overtime = 20
        self.control_range = 4 # control range for the position controller, 4 means 4 times of the oringal movement range

        self.log_torque = False

        self.elapsed_time = 0
        self.init_time = 0

        self._init()

    def build_ground(self):
        n = [0, 0, 0]
        n[self.up_dir] = 1
        c = self.create_collision_shape(self.GEOM_PLANE, planeNormal=n)
        ground = self.create_multi_body(0, c)
        self.change_dynamics(ground, -1, lateralFriction=0.9)
        return ground

    def _init(self):
        super()._init()

        self.agent.init()
        self.up_dir = next((i for i in range(2) if self.info["gravity"][i] != 0), 2)
        self.ref_motion.init()
        self.ground = self.build_ground()

        self.contactable_links = None if self.ref_motion.contactable_links is None else [
            self.agent.links[l] for l in self.ref_motion.contactable_links
        ] 
        if self.log_torque:
            self.info["log"]["torque"] = {}
            for jid in self.agent.motors:
                info = self.agent.joint_info(jid)
                jtype = info[2]
                jname = info[1].decode("ascii")
                if jtype == self.JOINT_REVOLUTE:
                    self.info["log"]["torque"][jname] = []
                elif jtype == self.JOINT_SPHERICAL:
                    self.info["log"]["torque"][jname+"_x"] = []
                    self.info["log"]["torque"][jname+"_y"] = []
                    self.info["log"]["torque"][jname+"_z"] = []
                else:
                    assert(jtype == self.JOINT_REVOLUTE or jtype == self.JOINT_SPHERICAL)

        self.elapsed_time = 0
        self.observation_space.shape = np.array(self.observe()).shape
        self.action_space.shape, self.action_space.low, self.action_space.high = self.init_action_space()
    
    def init_action_space(self):
        action_mean, action_std = [], []
        lower_bound, upper_bound = [], []

        if self.control_mode == self.POSITION_CONTROL:
            for jid, lim in zip(self.agent.motors, self.agent.movement_lim):
                info = self.agent.joint_info(jid)
                joint_type = info[2]
                if joint_type == self.JOINT_REVOLUTE:
                    action_mean.append(0.5*(lim[1] + lim[0]))
                    action_std.append((lim[1] - lim[0])*0.5*self.control_range)
                    lower_bound.append(-1.0)
                    upper_bound.append(1.0)
                elif joint_type == self.JOINT_SPHERICAL:
                    # a small offset along z-axis there is when normalizing action in DeepMimic's implementation
                    y_offset, z_offset = 0, 0.2 # self.up_dir == 1
                    if self.up_dir == 2:
                        y_offset, z_offset = -z_offset, y_offset
                    action_mean.extend([0, y_offset, z_offset, 0])  # in order of axis, angle
                    action_std.extend([1, 1, 1, (lim[1] - lim[0])*0.5*self.control_range])
                    lower_bound.extend([-1.0, -1.0-y_offset, -1.0-z_offset, -1.0])
                    upper_bound.extend([1.0, 1.0-y_offset, 1.0-z_offset, 1.0])
                else:
                    assert(joint_type in [self.JOINT_REVOLUTE, self.JOINT_SPHERICAL])
        else:
            assert(self.control_mode in [self.POSITION_CONTROL, self.TORQUE_CONTROL])
            for jid, lim in zip(self.agent.motors, self.agent.torque_lim):
                info = self.agent.joint_info(jid)
                joint_type = info[2]
                if joint_type == self.JOINT_REVOLUTE:
                    action_mean.append(0.0)
                    action_std.append(lim)
                    lower_bound.append(-1.0)
                    upper_bound.append(1.0)
                elif joint_type == self.JOINT_SPHERICAL:
                    action_mean.extend([0.0, 0.0, 0.0])
                    action_std.extend([lim, lim, lim])
                    lower_bound.extend([-1.0, -1.0, -1.0])
                    upper_bound.extend([1.0, 1.0, 1.0])
                else:
                    assert(joint_type in [self.JOINT_REVOLUTE, self.JOINT_SPHERICAL])
                    
        self.unnormalize_action = lambda a: np.add(action_mean, np.multiply(a, action_std))

        return [len(action_mean)], lower_bound, upper_bound

    def pre_process_action(self, action):
        action = self.unnormalize_action(action)
        if self.control_mode == self.POSITION_CONTROL:
            i = 0
            for jid in self.agent.motors:
                joint_type = self.agent.joint_info(jid)[2]
                if joint_type == self.JOINT_REVOLUTE:
                    i += 1
                else: # joint_type == self.JOINT_SPHERICAL
                    q = axis_angle2quat((action[i+0], action[i+1], action[i+2]), action[i+3])
                    action[i+0], action[i+1], action[i+2], action[i+3] = q
                    i += 4
        return action

    def reset(self):
        if self.log_torque:
            for v in self.info["log"]["torque"].values():
                v.clear()

        self.simulation_steps = 0
        if self.random_init_pose and not self._render:
            phase = np.random.rand()
        else:
            phase = 0.0
        self.init_time = phase * self.ref_motion.duration
        self.elapsed_time = self.init_time

        self.ref_motion.reset()
        ref_pose = self.ref_motion.set_sim_time(self.elapsed_time)
        self.agent.reset(ref_pose)

        dist = math.inf
        for lid in range(self.agent.n_links):
            aabb_min, aabb_max = self.agent.aabb(lid)
            ground_max_h = 0
            dist = min(dist, aabb_min[self.up_dir] - ground_max_h)
        dist -= 0.001
        if dist < 0:
            ref_pose["base_position"][self.up_dir] -= dist
            self.agent.reset_base_position_and_orientation(ref_pose["base_position"], ref_pose["base_orientation"])
        state = self.observe()
        if self._render:
            self._render_delay = time.time()
            pos, _ = self.agent.base_position_and_orientation
            self.reset_debug_visualizer_camera(2, 225, -15, [pos[0], 1.0, pos[2]])
            

        return state

    def step(self, action):        
        action = self.pre_process_action(action)
        assert(not np.any(np.isnan(action)))

        self.simulation_steps += 1
        phase = self.phase_state()

        for frame in range(self.frame_skip):
            if self.control_mode == self.TORQUE_CONTROL:
                self.agent.target_torque = action
            else:
                self.agent.target_position = action
            self.do_simulation()
            
            if self._render:
                t = time.time()
                time.sleep(max(0, self.time_step - (t-self._render_delay)/1000))
                pos, _ = self.agent.base_position_and_orientation
                self.reset_debug_visualizer_camera(2, 225, -15, [pos[0], 1.0, pos[2]])
                self._render_delay = t

            self.elapsed_time += self.time_step
            phase_ = self.phase_state()
            if phase_ < phase:
                base_pos, base_orient = self.agent.base_position_and_orientation
                self.ref_motion.sync(self.elapsed_time, base_pos, None)
                phase = phase_
            
            if self.log_torque: self._log_torque()

        terminal = self.contactable_links is not None
        terminal &= self.agent.has_contact(self.ground, exclusive_links=self.contactable_links)

        reward = self.reward(terminal)
        self.info["TimeLimit.truncated"] = not self._render and not terminal and self.elapsed_time >= self.overtime+self.init_time
        terminal |= self.info["TimeLimit.truncated"]
        
        state = self.observe()
        return state, reward, terminal, self.info

    def observe(self):
        assert(self.up_dir == 1 or self.up_dir == 2)
        base_pos, base_orient = self.agent.base_position_and_orientation
        base_height = base_pos[self.up_dir]

        up_dir_vec = [0, 0, 0]
        up_dir_vec[self.up_dir] = 1
        origin = [p for p in base_pos]
        origin[self.up_dir] = 0
        heading = self.orient2heading(base_orient)
        delta_orient_world2base = axis_angle2quat(up_dir_vec, -heading)
        rot = quat2mat(delta_orient_world2base)
        def trans_world2base(translate):
            return np.matmul(rot, translate)

        pos_state = [base_height]
        vel_state = []

        for lid in range(self.agent.n_links):
            m = self.agent.dynamics_info(lid)[0]
            if m > 0:
                state = self.agent.link_state(lid, compute_forward_kinematics=True, compute_link_velocity=True)
                pos, orient, linear_vel, angular_vel = state[0], state[1], state[6], state[7]

                pos = trans_world2base(np.subtract(pos, origin))
                pos[self.up_dir] -= base_height
                if lid > 0: # for base link, record its world orientation
                    orient = quatmultiply(delta_orient_world2base, orient)
                linear_vel_ = trans_world2base(linear_vel)
                angular_vel = trans_world2base(angular_vel)

                if orient[3] < 0: orient = [-e for e in orient]
                pos_state.extend(pos)
                pos_state.extend(orient)
                vel_state.extend(linear_vel)
                vel_state.extend(angular_vel)

        return [self.phase_state()] + pos_state + vel_state
    
    def reward(self, terminal):
        if terminal:
            reward = 0
        else:
            scale = {
                "pose": 2.0, "vel": 0.1, "end_eff": 40, "root": 5, "com": 10
            }
            weight = {
                "pose":0.5, "vel": 0.05, "end_eff": 0.15, "root": 0.2, "com": 0.1
            }
            f = 1.0/sum([v for v in weight.values()])
            weight = {
                k: v*f for k, v in weight.items()
            }
            joint_weight = {
                "base": 1.0, "chest": 0.5, "head": 0.3,
                "right_thign": 0.5, "right_shin": 0.3, "right_foot": 0.2,
                "right_upper_arm": 0.3, "right_forearm": 0.2, # "right_hand": 0,
                "left_thign": 0.5, "left_shin": 0.3, "left_foot": 0.2,
                "left_upper_arm": 0.3, "left_forearm": 0.2, # "left_hand": 0
            }
            f = 1.0/sum([v for v in joint_weight.values()])
            joint_weight = {
                k: v*f for k, v in joint_weight.items()
            }

            base_pos, base_orient = self.agent.base_position_and_orientation
            base_linear_vel, base_angular_vel = self.agent.base_velocity

            # self.ref_motion.set_sim_time(self.elapsed_time)
            # ref_base_pos, ref_base_orient = self.ref_motion.dummy_agent.base_position_and_orientation
            # ref_base_linear_vel, ref_base_angular_vel = self.ref_motion.dummy_agent.base_velocity
            ref_pose = self.ref_motion.dummy_pose(self.elapsed_time, compute_forward_kinematics=True)
            # ref_pose = self.tar_pose
            ref_base_pos = ref_pose["base_position"]
            ref_base_orient = ref_pose["base_orientation"]
            ref_base_linear_vel = ref_pose["base_linear_velocity"]
            ref_base_angular_vel = ref_pose["base_angular_velocity"]

            assert(self.up_dir == 1 or self.up_dir == 2)
            up_dir_vec = [0, 0, 0]
            up_dir_vec[self.up_dir] = 1

            heading = self.orient2heading(base_orient)
            rot = quat2mat(axis_angle2quat(up_dir_vec, -heading))
            def trans_world2base(translate):
                return np.matmul(rot, translate)
            
            ref_heading = self.orient2heading(ref_base_orient)
            ref_rot = quat2mat(axis_angle2quat(up_dir_vec, -ref_heading))
            def ref_trans_world2base(translate):
                return np.matmul(ref_rot, translate)

            _, angle = quat2axis_angle(quatdiff(base_orient, ref_base_orient))
            pose_err = joint_weight["base"] * (angle*angle)
            dv = np.linalg.norm(np.subtract(base_angular_vel, ref_base_angular_vel))
            vel_err = joint_weight["base"] * (dv*dv)
            # for jid in self.agent.motors:
            for name, joints in self.agent.joint_groups.items():
                if len(joints) == 1:
                    pos, vel, *_ = self.agent.joint_state(joints[0])
                    # ref_pos, ref_vel, *_ = self.ref_motion.dummy_agent.joint_state(joints[0])
                    ref_pos = ref_pose[joints[0]]["position"]
                    ref_vel = ref_pose[joints[0]]["velocity"]
                elif len(joints) == 3:
                    # convert stacked revolute joints (in order z,y,x) to a spherical joint
                    # i.e. convert the 2nd (y) and 3rd (x) joint to the frame of the 1st (z) joint
                    #    then add all things together
                    pos, ref_pos = [0, 0, 0, 1], [0, 0, 0, 1]
                    vel, ref_vel = [0, 0, 0], [0, 0, 0]
                    for jid in joints:
                        axis = self.agent.joint_info(jid)[13]
                        p, v, *_ = self.agent.joint_state(jid)
                        pos = quatmultiply(pos, axis_angle2quat(axis, p[0]))
                        vel = np.add(vel, rotate_vector(pos, np.multiply(v, axis)))
                        # ref_p, ref_v, *_ = self.ref_motion.dummy_agent.joint_state(jid)
                        # ref_p, ref_v = ref_p[0], ref_v[0]
                        ref_p = ref_pose[jid]["position"]
                        ref_v = ref_pose[jid]["velocity"]
                        ref_pos = quatmultiply(ref_pos, axis_angle2quat(axis, ref_p))
                        ref_vel = np.add(ref_vel, rotate_vector(ref_pos, np.multiply(ref_v, axis)))
                    vel = rotate_vector(quatconj(pos), vel)
                    ref_vel = rotate_vector(quatconj(ref_pos), ref_vel)
                else:
                    assert(len(joints) == 1 or len(joints) == 3)

                if len(pos) == 1:
                    dp = pos[0] - ref_pos[0]
                    dv = vel[0] - ref_vel[0]
                elif len(pos) == 4:
                    _, dp = quat2axis_angle(quatdiff(pos, ref_pos))
                    dv = np.linalg.norm([v0 - v1 for v0, v1 in zip(vel, ref_vel)])
                else:
                    assert(len(pos) == 1 or len(pos) == 4)
                pose_err += joint_weight[name] * dp*dp
                vel_err += joint_weight[name] * dv*dv
                
            end_err = 0.0
            world_link_frame_position = 4
            for lid in self.agent.end_effectors:
                state = self.agent.link_state(lid, compute_forward_kinematics=True)
                joint_pos = state[world_link_frame_position]

                # ref_state = self.ref_motion.dummy_agent.link_state(lid, compute_forward_kinematics=True)
                # ref_joint_pos = ref_state[world_link_frame_position]
                ref_joint_pos = ref_pose[lid]["world_link_frame_position"]

                pos_rel = np.subtract(joint_pos, base_pos)
                pos_rel[self.up_dir] = joint_pos[self.up_dir] # - self.ground_height(joint_pos)
                ref_pos_rel = np.subtract(ref_joint_pos, ref_base_pos)
                ref_pos_rel[self.up_dir] = ref_joint_pos[self.up_dir] # - self.ref_ground_height(ref_joint_pos)
                pos_rel = trans_world2base(pos_rel)
                ref_pos_rel = ref_trans_world2base(ref_pos_rel)

                dp = np.linalg.norm(np.subtract(pos_rel, ref_pos_rel))
                end_err += dp*dp
                
            if len(self.agent.end_effectors) > 0:
                end_err /= len(self.agent.end_effectors)

            # base_pos[self.up_dir] -= self.ground_height(base_pos)
            # ref_pose["base_position"][self.up_dir] -= self.ref_ground_height("base_position")
            root_pos_err = np.linalg.norm(np.subtract(base_pos, ref_base_pos))
            _, root_rot_err = quat2axis_angle(quatdiff(base_orient, ref_base_orient))
            root_vel_err = np.linalg.norm(np.subtract(base_linear_vel, ref_base_linear_vel))
            root_ang_vel_err = np.linalg.norm(np.subtract(base_angular_vel, ref_base_angular_vel))
            root_err = root_pos_err*root_pos_err + \
                    0.1 * root_rot_err*root_rot_err + \
                    0.01 * root_vel_err*root_vel_err + \
                    0.001 * root_ang_vel_err*root_ang_vel_err
                    
            mass, momentum, ref_momentum = 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            world_linear_velocity = 6
            for lid in range(self.agent.n_links):
                m = self.agent.dynamics_info(lid)[0]
                if m > 0:
                    linear_vel = self.agent.link_state(lid, compute_link_velocity=True)[world_linear_velocity]

                    # ref_linear_vel = self.ref_motion.dummy_agent.link_state(lid, compute_link_velocity=True)[world_linear_velocity]
                    ref_linear_vel = ref_pose[lid]["world_linear_velocity"]

                    momentum = np.add(momentum, np.multiply(m, linear_vel))
                    ref_momentum = np.add(ref_momentum, np.multiply(m, ref_linear_vel))
                    mass += m
            com_vel, ref_com_vel = momentum/mass,  ref_momentum/mass
            dv = np.linalg.norm(np.subtract(com_vel, ref_com_vel))
            com_err = 0.1 * dv*dv
            pose_reward = math.exp(-scale["pose"]*pose_err)
            vel_reward  = math.exp(-scale["vel"]*vel_err)
            end_reward  = math.exp(-scale["end_eff"]*end_err)
            root_reward = math.exp(-scale["root"]*root_err)
            com_reward  = math.exp(-scale["com"]*com_err)

            reward = weight["pose"] * pose_reward + \
                weight["vel"] * vel_reward  + \
                weight["end_eff"] * end_reward  + \
                weight["root"] * root_reward + \
                weight["com"] * com_reward

        return reward
    
    def phase_state(self):
        phase = math.fmod(self.elapsed_time / self.ref_motion.duration, 1.0)
        if phase < 0: phase += 1
        return phase
    
    def orient2heading(self, orient):
        # heading_vec = rotate_vector(orient, (1, 0, 0))
        if self.up_dir == 1:
            # heading = math.atan2(-heading_vec[2], heading_vec[0])
            _, _, heading = quat2euler_zyx((orient[0], -orient[2], orient[1], orient[3]))
        elif self.up_dir == 2:
            # heading = math.atan2(heading_vec[1], heading_vec[0])
            _, _, heading = quat2euler_zyx(orient)
        else:
            assert(self.up_dir in [1, 2])
        return heading
    
    def _log_torque(self):
        if self.control_mode == self.POSITION_CONTROL and not self.agent.use_spd:
            for jid in self.agent.motors:
                jname = self.agent.joint_info(jid)[1].decode("ascii")
                torque = self.agent.joint_state(jid)[3]
                if len(torque) == 1:
                    self.info["log"]["torque"][jname].append(torque[0])
                elif len(torque) == 3:
                    self.info["log"]["torque"][jname+"_x"].append(torque[0])
                    self.info["log"]["torque"][jname+"_y"].append(torque[1])
                    self.info["log"]["torque"][jname+"_z"].append(torque[2])
                else:
                    assert(len(torque) == 1 or len(torque) == 3)
        else:
            i = 0
            for jid in self.agent.motors:
                info = self.agent.joint_info(jid)
                jname = info[1].decode("ascii")
                joint_type = info[2]
                if joint_type == self.JOINT_REVOLUTE:
                    self.info["log"]["torque"][jname].append(self.agent.control_param["forces"][i])
                    i += 1
                else: # joint_type == self.JOINT_SPHERICAL
                    self.info["log"]["torque"][jname+"_x"].append(self.agent.control_param["forces"][i])
                    self.info["log"]["torque"][jname+"_y"].append(self.agent.control_param["forces"][i+1])
                    self.info["log"]["torque"][jname+"_z"].append(self.agent.control_param["forces"][i+2])
                    i += 3


def foo(tar_action):
    class bar(DeepMimic):
        metadata = {"render.modes": ["human"]}
        @property
        def unwrapped(self):
            return self
        @property
        def reward_range(self):
            return (0, 1)
            
        def __init__(self, *args, **kwargs):
            if "action" in kwargs:
                assert(kwargs["action"] == tar_action)
            else:
                kwargs["action"] = tar_action
            super().__init__(*args, **kwargs)
    return bar
for f in os.listdir(MOTION_DIR):
    if f.startswith("humanoid3d_") and f.endswith(".txt"):
        tar_action = f[11:-4]
        exec("DeepMimic{} = foo('{}');".format("".join(x.capitalize() or "_" for x in tar_action.split("_")), tar_action))
