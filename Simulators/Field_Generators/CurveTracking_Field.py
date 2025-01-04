from Simulators.ORCA_Env import ORCA_Env
from Simulators.Field_Generators.Base_Field import Base_Field
from Viewer.Viewer import Viewer
from Utils import *
import numpy as np
import shapely.geometry as geom
import math,time,os, random
import rasterio
from rasterio.features import geometry_mask
import copy
from Simulators.Field_Generators.vectorfield_stack.distance_field import distancefield_class


# Vector field:
# reverse_direction # flag to invert the direction the curve will be followed
# vr # reference forward speed for the vector field
# kf # convergence gain of the vector field

# #Collision avoidance considering the closest point:
# flag_follow_obstacle # flag to enable the robot to follow an obstacle when it s blocking the vector field
# epsilon # reference distance between the robot and the path being followed
# switch_dist_0 # distance from which the robot will start to follow the obstacle
# switch_dist # distance from which the robot will start to follow the obstacle

class CurveTracking_Field(Base_Field):
    def __init__(self, reverse_direction = False, vr = 1, kf = 0.005,
                 flag_follow_obstacle = True, epsilon = 20, switch_dist_0 = 60, switch_dist = 40, lidar_N=128):
        self.reset_params(reverse_direction, vr, kf,
                          flag_follow_obstacle, epsilon, switch_dist_0, switch_dist, lidar_N)

    def reset_params(self, reverse_direction = False, vr = 1, kf = 0.005,
                     flag_follow_obstacle = True, epsilon = 20, switch_dist_0 = 60, switch_dist = 40, lidar_N=128):
        # params related to vector field
        self.reverse_direction = reverse_direction
        self.vr = vr
        self.kf = kf
        # params related to obstacle
        self.flag_follow_obstacle = flag_follow_obstacle
        self.epsilon = epsilon
        self.switch_dist_0 = switch_dist_0
        self.switch_dist = switch_dist
        self.lidar_N = lidar_N
        # set vec field obj
        self.vec_field_obj = distancefield_class(self.vr, self.kf, self.reverse_direction,
                                                 self.flag_follow_obstacle, self.epsilon, self.switch_dist_0, self.switch_dist)

    # # Example guidance:
    # guidance = {
    #     "type": "lines",
    #     "params": {
    #         "lines":[[[50, 50], [50, 600]],[[50, 600], [300, 600]], [[300, 600], [300, 250]], [[300, 250], [500, 250]]],
    #         # "width": 150,   # influence width
    #         # "decay_rate": 0.9,   # decay rate of the guidance field along width
    #     }
    # }
    # # Example constrains:
    # constrains = {
    #     "filter_path_n_average": 0,  #Number of points to use in the average filter (it is forced to be an odd number) - if 0 the path is not filtered
    #     "closed_path_flag": False,  #Flag to indicate if the path is closed or not
    #     "pt_step_len": 20,  # pt_step_len #The length between two neighbor points
    #     "smooth_condition": 600,  #The smooth condition for trajectory smoothing
    # }
    def get_field(self, guidance, constrains={"filter_path_n_average": 0, "closed_path_flag": False, "pt_step_len": 20, "smooth_condition": 600}):
        assert guidance["type"] == "lines"

        filter_path_n_average = constrains["filter_path_n_average"]
        closed_path_flag = constrains["closed_path_flag"]
        pt_step = constrains["pt_step_len"]
        smooth_cd = constrains["smooth_condition"]
        line_lengths = np.linalg.norm(np.array(guidance["params"]["lines"])[:, 1]-np.array(guidance["params"]["lines"])[:, 0], axis=1)
        point_n = np.sum((line_lengths/pt_step).astype(int))+1
        if point_n < 10:
            pt_step = np.sum(line_lengths) / (10+2)

        path_points = []
        for line_i in guidance["params"]["lines"]:
            pt1 = copy.deepcopy(line_i[0])
            pt2 = copy.deepcopy(line_i[1])
            length_ = np.linalg.norm(np.array(pt2)-np.array(pt1))
            for step_ in range(int(length_/pt_step)):
                pct = step_/int(length_/pt_step)
                pt_ = pct*np.array(pt2)+(1-pct)*np.array(pt1)
                path_points.append((pt_[0], pt_[1], 0.))
        path_points.append((guidance["params"]["lines"][-1][-1][0], guidance["params"]["lines"][-1][-1][1], 0.))

        # path smoothing
        path_2d = np.array(path_points)[:, 0:2]
        path_2d_smooth = traj_smoothing(path_2d, s_=smooth_cd)
        # import matplotlib.pyplot as plt
        # plt.scatter(path_2d[:, 0], path_2d[:, 1], s=1)
        # plt.scatter(path_2d_smooth[:, 0], path_2d_smooth[:, 1], s=1, color='red')
        # plt.show()
        path_points = np.array(path_points)
        path_points[:, 0:2] = np.array(path_2d_smooth)
        path_points = path_points.tolist()

        if self.flag_follow_obstacle:
            lidar_ = Sensor(agent_r=0, obs_all=ORCA_Env.get_all_obstacles_from_scenario(self.scenario),
                            sensorRange=max(self.switch_dist_0,self.switch_dist)*2, N=self.lidar_N)

        grid_map = self.grid["grid_map"]
        grid_infor = self.grid["grid_infor"]
        grid_size = self.grid["grid_size"]
        field = np.zeros((grid_size[0], grid_size[1], 2))
        for xid in range(grid_size[0]):
            for yid in range(grid_size[1]):
                if grid_map[xid][yid] == 1:
                    continue
                center = copy.deepcopy(grid_infor[xid][yid]["center"])
                pos = [center[0], center[1], 0]
                self.vec_field_obj.set_pos(pos)
                self.vec_field_obj.set_path(path_points, 0, filter_path_n_average, closed_path_flag)

                if(self.flag_follow_obstacle):
                    lidar_infor = lidar_.get_sensor_reading([[pos[0], pos[1]]])[0]
                    if 0.0 in lidar_infor:
                        lidar_infor.remove(0.0)
                    min_dir_id = np.argmin(lidar_infor)
                    if abs(lidar_infor[min_dir_id]-max(self.switch_dist_0,self.switch_dist)*2) <= 1e-3:
                        closest_p_from_obs = [1e5, 1e5, 1e5]
                    else:
                        inter_p = np.array(lidar_.dirs[min_dir_id])*lidar_infor[min_dir_id]+np.array([pos[0], pos[1]])
                        closest_p_from_obs = [inter_p[0], inter_p[1], 0]
                    self.vec_field_obj.set_closest(closest_p_from_obs)

                if self.vec_field_obj.is_ready():
                    [Vx,Vy,Vz,terminated] = self.vec_field_obj.vec_field_path()
                    Vec = np.nan_to_num(np.array([Vx, Vy]) / np.linalg.norm(np.array([Vx, Vy])), 0)
                    field[xid][yid] = copy.deepcopy(Vec)

        return field



if __name__ == '__main__':
    scenario_test = {
        "wind_size": [800, 800],
        "obs_list": [{"type": "rectangle", "params": {"vertexes": [[100, 400], [200, 400], [200, 500], [100, 500]]}, "attributes": {}},
                     {"type": "triangle", "params": {"vertexes": [[150, 150], [250, 150], [150, 250]]}, "attributes": {}},
                     {"type": "circle", "params": {"center": [400, 400], "radius": 50}, "attributes": {}}
                     ],
        "zebra_crossing_list": [],
        "passages_list": [],
        "areas_list": [],
    }
    grid_width = 20.0

    guidance = {
        "type": "lines",
        "params": {
            "lines":[[[50, 50], [50, 600]],[[50, 600], [300, 600]], [[300, 600], [300, 250]], [[300, 250], [500, 250]],
                     [[500, 250], [500, 450]], [[500, 450], [750, 750]]],
            "width": 150,   # influence width
            "decay_rate": 0.9,   # decay rate of the guidance field along width
        }
    }
    constrains = {
        "filter_path_n_average": 0,
        "closed_path_flag": False,
        "pt_step_len": 60,
        "smooth_condition": 600,
    }
    crv_fld = CurveTracking_Field(reverse_direction = False, vr = 1, kf = 0.008,
                                  flag_follow_obstacle = True, epsilon = 20, switch_dist_0 = 60, switch_dist = 40, lidar_N=256)
    crv_fld.reset(scenario_test, grid_width)

    field = crv_fld.get_field(guidance, constrains)
    crv_fld.field_visualization(field, guidance)