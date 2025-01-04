from Simulators.ORCA_Env import ORCA_Env
from Simulators.Field_Generators.Base_Field import Base_Field
from Viewer.Viewer import Viewer
from Utils import *
import numpy as np
from scipy.optimize import minimize
import shapely.geometry as geom
import math,time,os
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
import queue
import random
import copy

class Navigation_Field(Base_Field):
    def __init__(self, guidance_handle_colli=True, obs_inful_range=60, lidar_N=128, obs_repel_scale=1.3):
        self.g_handle_colli = guidance_handle_colli
        if self.g_handle_colli:
            self.obs_range = obs_inful_range
            self.lidar_N = lidar_N
            self.obs_repel_scale = obs_repel_scale

    def get_field(self, guidance, constrains):
        guidance_field, visited_map = self.get_guidance_field(guidance)
        if constrains["with_goal"]:
            guidance_field = np.array(guidance_field) * guidance["params"]["strength"]
            nav_field = self.get_navigation_field(guidance_field, constrains["goal_position"])
            return nav_field
        else:
            return guidance_field

    def get_guidance_field(self, guidance_source):
        scenario_ = self.scenario
        grid_infor = self.grid["grid_infor"]
        grid_width = self.grid["grid_width"]

        guidance_field = np.zeros((len(grid_infor), len(grid_infor[0]), 2))
        if guidance_source["type"] == "lines":
            lines = guidance_source["params"]["lines"]
            ### pre_process the lines
            for l_id in range(len(lines)):
                if l_id == 0:
                    if lines[l_id][0][0] % grid_width == 0:
                        lines[l_id][0][0] += (2*random.random()-1)*1e1
                    if lines[l_id][0][1] % grid_width == 0:
                        lines[l_id][0][1] += (2*random.random()-1)*1e1
                if lines[l_id][1][0] % grid_width == 0:
                    lines[l_id][1][0] += (2*random.random()-1)*1e1
                if lines[l_id][1][1] % grid_width == 0:
                    lines[l_id][1][1] += (2*random.random()-1)*1e1
                if l_id+1 < len(lines):
                    lines[l_id+1][0] = copy.deepcopy(lines[l_id][1])

            field_width = int(guidance_source["params"]["width"] / grid_width)
            decay_rate = guidance_source["params"]["decay_rate"]

            # get field core (faster)
            visited_map = np.zeros((len(grid_infor), len(grid_infor[0])))
            cnt_map = np.zeros((len(grid_infor), len(grid_infor[0])))
            current_field_points = []
            grid_size = [len(grid_infor), len(grid_infor[0])]
            transform_ = rasterio.transform.from_bounds(0, 0, grid_width*grid_size[0], grid_width*grid_size[1], grid_size[0], grid_size[1])
            for line_i in lines:
                ls_ = geom.LineString(copy.deepcopy(line_i))
                geom_mask = rasterio.features.geometry_mask([ls_], out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
                geom_mask = np.flip(geom_mask, axis=0).transpose(1, 0)
                inter_ps = (np.argwhere(geom_mask==False)).tolist()

                for p_i in inter_ps:
                    if (grid_infor[p_i[0]][p_i[1]]["free"]==1):
                        continue
                    grid_box = copy.deepcopy(grid_infor[p_i[0]][p_i[1]]["box"])
                    box_poly = geom.Polygon([[p[0],p[1]] for p in grid_box])
                    intersections_ = np.array(box_poly.intersection(geom.LineString(line_i)).coords)
                    if len(intersections_) < 2:
                        continue
                    dir_linei = (np.array(line_i) - np.array(line_i[0]))/np.linalg.norm(np.array(line_i) - np.array(line_i[0]))
                    dir_inter = (np.array(intersections_) - np.array(intersections_[0]))/np.linalg.norm(np.array(intersections_) - np.array(intersections_[0]))
                    if np.linalg.norm(dir_linei-dir_inter) >= 1e-6:
                        mid_ = copy.deepcopy(intersections_[0])
                        intersections_[0] = copy.deepcopy(intersections_[1])
                        intersections_[1] = copy.deepcopy(mid_)
                    if p_i not in current_field_points:
                        current_field_points.append(copy.deepcopy(p_i))
                    intersections_ = (intersections_ - intersections_[0])/np.linalg.norm(intersections_ - intersections_[0])
                    guidance_field[p_i[0]][p_i[1]] += np.array(intersections_[1])
                    cnt_map[p_i[0]][p_i[1]] += 1
                    visited_map[p_i[0]][p_i[1]] = 1
            for pt in current_field_points:
                guidance_field[pt[0]][pt[1]] /= cnt_map[pt[0]][pt[1]]
                guidance_field[pt[0]][pt[1]] = np.array(guidance_field[pt[0]][pt[1]]) / np.linalg.norm(np.array(guidance_field[pt[0]][pt[1]]))

            # expand
            if self.g_handle_colli:
                lidar = Sensor(agent_r=0, obs_all=ORCA_Env.get_all_obstacles_from_scenario(scenario_), sensorRange=self.obs_range*2, N=self.lidar_N)
            for iter in range(int(field_width/2)):
                next_field_points = []
                for pt in current_field_points:
                    for x_dr, y_dr in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                        nb_x = pt[0] + x_dr
                        nb_y = pt[1] + y_dr
                        if(nb_x<0 or nb_x>=len(grid_infor) or nb_y<0 or nb_y>=len(grid_infor[0])
                           or grid_infor[nb_x][nb_y]["free"]==1) or visited_map[nb_x][nb_y]==1 or [nb_x, nb_y] in next_field_points:
                            continue
                        if [nb_x, nb_y] not in next_field_points:
                            next_field_points.append([nb_x, nb_y])

                        vec_cnt = 0.
                        vec = np.array([0., 0.])
                        for x_dr_, y_dr_ in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                            nbnb_x = nb_x + x_dr_
                            nbnb_y = nb_y + y_dr_
                            if(nbnb_x<0 or nbnb_x>=len(grid_infor) or nbnb_y<0 or nbnb_y>=len(grid_infor[0])
                                    or grid_infor[nbnb_x][nbnb_y]["free"]==1):
                                continue
                            if (visited_map[nbnb_x, nbnb_y]==1):
                                vec_cnt += 1
                                vec += np.array(guidance_field[nbnb_x, nbnb_y])
                        vec = vec / vec_cnt * decay_rate
                        vec = vec / np.linalg.norm(vec) * (decay_rate ** (iter+1))

                        if self.g_handle_colli:
                            lidar_infor = lidar.get_sensor_reading([copy.deepcopy(grid_infor[nb_x][nb_y]["center"])])[0]
                            if 0.0 in lidar_infor:
                                lidar_infor.remove(0.0)
                            lidar_infor = np.array(lidar_infor)
                            min_dir_id = np.argmin(lidar_infor)
                            min_dis = lidar_infor[min_dir_id]
                            if min_dis<self.obs_range:
                                dir_ = np.array(lidar.dirs[min_dir_id])
                                ori_vec = copy.deepcopy(vec)
                                ori_vec /= np.linalg.norm(ori_vec)
                                if np.dot(dir_, ori_vec) > 0:
                                    ori_vec -= self.obs_repel_scale*(dir_/np.linalg.norm(dir_)*(1-min_dis/self.obs_range))
                                    vec = ori_vec / np.linalg.norm(ori_vec) * (decay_rate ** (iter+1))

                        guidance_field[nb_x, nb_y] = np.array(vec)

                for nx_pt in next_field_points:
                    visited_map[nx_pt[0], nx_pt[1]] = 1
                current_field_points = copy.deepcopy(next_field_points)

            return guidance_field, visited_map


    # generate the navigation field given [freespace_grid_map, agents' goal position, guidance_field]
    def get_navigation_field(self, guidance_field, goal_position):
        grid_map = self.grid["grid_map"]
        grid_width = self.grid["grid_width"]

        navigation_field = np.zeros((len(grid_map), len(grid_map[0]), 2))
        cost_map = np.ones((len(grid_map), len(grid_map[0]))) * 1e18
        visited_map = np.zeros((len(grid_map), len(grid_map[0])))
        goal_grid = [int(goal_position[0]/grid_width), int(goal_position[1]/grid_width)]
        cost_map[goal_grid[0], goal_grid[1]] = 0
        update_map = np.zeros((len(grid_map), len(grid_map[0])))

        current_visit_list = [copy.deepcopy(goal_grid)]
        while(1):
            current_cost_list = cost_map[np.array(current_visit_list)[:, 0], np.array(current_visit_list)[:, 1]]
            visit_idx = np.argmin(np.array(current_cost_list))
            visit_node = current_visit_list[visit_idx]
            del current_visit_list[visit_idx]
            visited_map[visit_node[0], visit_node[1]] = 1

            for x_dr, y_dr in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                neighbor_x = visit_node[0] + x_dr
                neighbor_y = visit_node[1] + y_dr
                if(neighbor_x<0 or neighbor_x>=len(grid_map) or neighbor_y<0 or neighbor_y>=len(grid_map[0])
                        or grid_map[neighbor_x][neighbor_y]==1):
                    continue
                if visited_map[neighbor_x, neighbor_y] == 0:
                    if [neighbor_x, neighbor_y] not in current_visit_list:
                        current_visit_list.append([neighbor_x, neighbor_y])
                    ### update cost and vector
                    # get this neighbor node's neighbors
                    nb_nbs = []
                    nb_nbs_dirs = []
                    for x_dr_, y_dr_ in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                        nb_nb_x = neighbor_x + x_dr_
                        nb_nb_y = neighbor_y + y_dr_
                        if (nb_nb_x<0 or nb_nb_x>=len(grid_map) or nb_nb_y<0 or nb_nb_y>=len(grid_map[0])
                                or grid_map[nb_nb_x][nb_nb_y]==1):
                            continue
                        if cost_map[nb_nb_x, nb_nb_y] < 1e18:
                            nb_nbs.append([nb_nb_x, nb_nb_y])
                            nb_nbs_dirs.append([x_dr_, y_dr_])

                    update_map[neighbor_x, neighbor_y] = max(update_map[neighbor_x, neighbor_y], len(nb_nbs))
                    ### compute cost for this neighbor node
                    if len(nb_nbs)==1 or (len(nb_nbs)==2 and np.linalg.norm(np.array(nb_nbs_dirs[0])+np.array(nb_nbs_dirs[1]))<1e-6):
                        for nbnb_id in range(len(nb_nbs)):
                            alpha = 0.
                            cost_A = cost_map[nb_nbs[nbnb_id][0], nb_nbs[nbnb_id][1]]
                            cost_B = len(grid_map)*2
                            dir_A = copy.deepcopy(nb_nbs_dirs[nbnb_id])
                            dir_B = None
                            A_G_cross = np.cross(np.array(dir_A), np.array(guidance_field[neighbor_x][neighbor_y]))
                            if abs(A_G_cross) < 1e-6:
                                dir_B = [dir_A[1], dir_A[0]]
                            else:
                                for dir_b in [[-1, 0], [0, -1], [0, 1], [1, 0]]:
                                    if abs(np.cross(np.array(dir_A), np.array(dir_b)) * A_G_cross) > 1e-6:
                                        dir_B = copy.deepcopy(dir_b)
                                        break
                            res = minimize(self.cost_function_full, np.array([alpha]), (cost_A, cost_B, dir_A, dir_B, np.array(guidance_field[neighbor_x][neighbor_y])),
                                           bounds=[(0, 1)])
                            if res.fun<cost_map[neighbor_x, neighbor_y]:
                                cost_map[neighbor_x, neighbor_y] = res.fun
                                vec_ = np.array(res.x[0]*np.array(nb_nbs_dirs[nbnb_id]))
                                navigation_field[neighbor_x,neighbor_y] = np.array(vec_)/np.linalg.norm(np.array(vec_))
                    else:
                        for nb1_id in range(len(nb_nbs)):
                            for nb2_id in range(nb1_id+1, len(nb_nbs)):
                                if np.linalg.norm(np.array(nb_nbs_dirs[nb1_id])+np.array(nb_nbs_dirs[nb2_id]))<1e-6:
                                    continue
                                alpha = 0.
                                cost_A = cost_map[nb_nbs[nb1_id][0], nb_nbs[nb1_id][1]]
                                cost_B = cost_map[nb_nbs[nb2_id][0], nb_nbs[nb2_id][1]]
                                dir_A = copy.deepcopy(nb_nbs_dirs[nb1_id])
                                dir_B = copy.deepcopy(nb_nbs_dirs[nb2_id])
                                res = minimize(self.cost_function_full, np.array([alpha]), (cost_A, cost_B, dir_A, dir_B, np.array(guidance_field[neighbor_x][neighbor_y])),
                                               bounds=[(0, 1)])
                                if res.fun<cost_map[neighbor_x, neighbor_y]:
                                    cost_map[neighbor_x, neighbor_y] = res.fun
                                    vec_ = np.array(res.x[0]*np.array(nb_nbs_dirs[nb1_id])+(1-res.x[0])*np.array(nb_nbs_dirs[nb2_id]))
                                    navigation_field[neighbor_x, neighbor_y] = np.array(vec_)/np.linalg.norm(np.array(vec_))
            if len(current_visit_list)==0:
                break

        return navigation_field


    @staticmethod
    def cost_function_full(alpha, costA, costB, dir_A, dir_B, G_):
        alpha = alpha[0]
        a = np.array(alpha*np.array(dir_A)+(1-alpha)*np.array(dir_B))
        G = np.array(G_)
        s = a.dot(G) + np.sqrt(np.square(a.dot(G))-G.dot(G) + 1)
        return alpha*costA + (1-alpha)*costB + np.linalg.norm(np.array([alpha, 1-alpha]))/s

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
            "width": 150,
            "decay_rate": 0.99,
            "strength": 1-1e-6
        }
    }
    constrains = {
        "with_goal": False,
        "goal_position": None
    }
    nav_fld = Navigation_Field(guidance_handle_colli=True, obs_inful_range=60, lidar_N=128, obs_repel_scale=1.3)
    nav_fld.reset(scenario_test, grid_width)
    vec_field = nav_fld.get_field(guidance, constrains)
    nav_fld.field_visualization(field=vec_field, guidance=guidance)