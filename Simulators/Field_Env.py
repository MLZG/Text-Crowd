import pyglet.window.key

from Simulators.ORCA_Env import *
from Viewer.Viewer import Viewer
from Utils import *
import numpy as np
import shapely.geometry as geom
import math,time,os,queue,random,copy
from scipy.interpolate import interp2d, griddata
import matplotlib.pyplot as plt


class Field_Env(ORCA_Env):
    def __init__(self, agent_num=AGENT_NUM_DEFAULT, visual=True, draw_scale=1.0):
        super(Field_Env, self).__init__(agent_num=agent_num, visual=visual, draw_scale=draw_scale)


    def reset(self, scenario, agent_setting):
        super(Field_Env, self).reset(scenario, agent_setting)
        self.agent_prefvs = []
        for aid in range(self.agent_num):
            self.agent_prefvs.append(self.agent_current_infor[aid]["pref_speed"])

    # group_fields:
    # [
    # {"agent_ids": [agent0, agent1, ...], "field": field0, "grid": grid0},
    # {"agent_ids": [agent2, agent3, ...], "field": field1, "grid": grid1},
    # ...
    # ]
    def perform_action(self, group_fields):
        agent_actions = np.zeros((self.agent_num, 2))

        start_time = time.time()
        for group_id in range(len(group_fields)):
            aids_gi = group_fields[group_id]["agent_ids"]
            field_gi = group_fields[group_id]["field"]
            grid_gi = group_fields[group_id]["grid"]
            grid_width = grid_gi["grid_width"]
            grid_size = grid_gi["grid_size"]
            grid_infor = grid_gi["grid_infor"]
            for agent_id in aids_gi:
                agent_pos = self.agent_current_infor[agent_id]["pos"]
                assert 0.<agent_pos[0]<self.current_scenario["wind_size"][0]
                assert 0.<agent_pos[1]<self.current_scenario["wind_size"][1]
                agent_prefV = self.agent_current_infor[agent_id]["pref_speed"]
                # get agent action based on bilinear interpolation
                closest_coord = [int(agent_pos[0]/grid_width), int(agent_pos[1]/grid_width)]

                dirs = np.array(agent_pos) - np.array(grid_infor[closest_coord[0]][closest_coord[1]]["center"])
                for d_i in range(len(dirs)):
                    if dirs[d_i] > 0:
                        dirs[d_i] = 1
                    elif dirs[d_i] < 0:
                        dirs[d_i] = -1
                    else:
                        dirs[d_i] = [-1, 1][random.randint(0, 1)]
                coords = [[closest_coord[0], closest_coord[1]], [closest_coord[0], int(closest_coord[1]+dirs[1])],
                          [int(closest_coord[0]+dirs[0]), int(closest_coord[1]+dirs[1])], [int(closest_coord[0]+dirs[0]), closest_coord[1]]]

                no_vec_cnt = 0
                for x_cd, y_cd in coords:
                    if x_cd<0 or x_cd>=grid_size[0] or y_cd<0 or y_cd>=grid_size[1] or grid_infor[x_cd][y_cd]["free"] == 1:
                        no_vec_cnt += 1
                if no_vec_cnt == len(coords):
                    labels = np.zeros((grid_size[0], grid_size[1]), dtype=int)
                    for cdi in coords:
                        labels[cdi[0]][cdi[1]] = 1
                    new_coords = []
                    for coord_i in coords:
                        for dir_x in [-1, 0, 1]:
                            for dir_y in [-1, 0, 1]:
                                new_coord_x = coord_i[0]+dir_x
                                new_coord_y = coord_i[1]+dir_y
                                if labels[new_coord_x][new_coord_y] == 0 and [new_coord_x, new_coord_y] not in new_coords:
                                    new_coords.append([new_coord_x, new_coord_y])
                                    labels[new_coord_x][new_coord_y] = 1
                    coords = copy.deepcopy(new_coords)

                points = []
                point_values = []
                fill_value = [0., 0.]
                for x_cd, y_cd in coords:
                    center_ = [x_cd*grid_width+grid_width/2, y_cd*grid_width+grid_width/2]
                    points.append(center_)
                    if x_cd<0 or x_cd>=grid_size[0] or y_cd<0 or y_cd>=grid_size[1] or grid_infor[x_cd][y_cd]["free"] == 1:
                        point_values.append(copy.deepcopy(fill_value))
                    else:
                        point_values.append(copy.deepcopy(field_gi[x_cd][y_cd]))
                points = np.array(points)
                point_values = np.array(point_values)

                mtd = "linear"
                agt_action = griddata(points=points, values=point_values, xi=np.array([agent_pos]), method=mtd, fill_value=0)[0]
                if abs(np.linalg.norm(agt_action) - 0.) < 1e-6:
                    dis_list = np.linalg.norm(np.array(points) - np.array(agent_pos), axis=1)
                    sorted_ids = np.argsort(dis_list)
                    for id_ in sorted_ids:
                        if abs(np.linalg.norm(point_values[id_]) - 0.) >= 1e-6:
                            agt_action = copy.deepcopy(point_values[id_])
                            break

                if np.linalg.norm(agt_action) < 1e-6:
                    agent_actions[agent_id] = np.array(agt_action)
                else:
                    agent_actions[agent_id] = (np.array(agt_action)/np.linalg.norm(np.array(agt_action))) * agent_prefV
        print(f"time in computing actions: {time.time()-start_time}")

        # perform actions
        start_time = time.time()
        super(Field_Env, self).perform_action(agent_actions.tolist())
        print(f"time in performing actions: {time.time()-start_time}")


    def perform_action_fast(self, group_fields):
        if not hasattr(self, 'agent_prefvs'):
            self.agent_prefvs = []
            for aid in range(self.agent_num):
                self.agent_prefvs.append(self.agent_current_infor[aid]["pref_speed"])

        agent_actions = np.zeros((self.agent_num, 2))
        all_agent_positions = np.array(self.get_current_positions())
        # get action for each agent based on the field of each group
        for group_id in range(len(group_fields)):
            aids_gi = group_fields[group_id]["agent_ids"]
            field_gi = group_fields[group_id]["field"]
            grid_gi = group_fields[group_id]["grid"]
            grid_width = grid_gi["grid_width"]
            grid_size = grid_gi["grid_size"]
            grid_infor = grid_gi["grid_infor"]

            xp = np.arange(0, grid_size[0])*grid_width + grid_width/2
            yp = np.arange(0, grid_size[1])*grid_width + grid_width/2
            zp_0 = field_gi[:, :, 0]
            zp_1 = field_gi[:, :, 1]

            agent_ps_gi = all_agent_positions[aids_gi]
            actions_gi = np.concatenate((interp_grid_fast(agent_ps_gi[:, 0], agent_ps_gi[:, 1], xp, yp, zp_0).reshape(-1, 1),
                                        interp_grid_fast(agent_ps_gi[:, 0], agent_ps_gi[:, 1], xp, yp, zp_1).reshape(-1, 1)),
                                        axis=1)

            # handle the actions that equal to zero
            zero_aids = np.argwhere(np.linalg.norm(actions_gi, axis=1)<1e-6).reshape(1, -1)[0]
            if len(zero_aids) > 0:
                za_positions = copy.deepcopy(agent_ps_gi[zero_aids])
                za_actions = interp_grid_closest_4(za_positions[:, 0], za_positions[:, 1], xp, yp, field_gi)
                actions_gi[zero_aids] = za_actions

            actions_gi = actions_gi / np.linalg.norm(actions_gi, axis=1).reshape(-1, 1) * (np.array(self.agent_prefvs)[aids_gi]).reshape(-1, 1)
            actions_gi = np.nan_to_num(actions_gi, 0.)

            agent_actions[aids_gi] = actions_gi

        # perform actions
        super(Field_Env, self).perform_action(agent_actions.tolist())


    def set_viewer_guidance(self, guidance):
        if guidance is None:
            self.viewer.set_traj([], [])
            return
        if guidance["type"] == "lines":
            lines = guidance["params"]["lines"]
            trajs = []
            traj_colors = []
            for lidx, line_i in enumerate(lines):
                trajs.append(np.array(line_i).reshape(1, -1)[0].tolist())
                traj_colors.append([255, 0, 0])
            self.viewer.set_traj(np.array(trajs), np.array(traj_colors))

    def set_viewer_field(self, grid, field):
        if grid is None or field is None:
            self.viewer.set_arrows([], [])
            return
        arrows = []
        arrow_colors = []
        for i in range(len(field)):
            for j in range(len(field[0])):
                vec = field[i][j]
                arrow = np.array([[0, 0], vec])
                arrow = (arrow - vec/2)*grid["grid_width"]
                arrow += np.array(grid["grid_infor"][i][j]["center"])
                arrows.append(arrow)
                arrow_colors.append([0, 0, 0])
        self.viewer.set_arrows(np.array(arrows), np.array(arrow_colors))

if __name__ == '__main__':
    from Simulators.Field_Generators.CurveTracking_Field import CurveTracking_Field
    from Simulators.Field_Generators.Navigation_Field import Navigation_Field

    agent_n = 200
    group_n = 2
    groups_centers = [[50, 50], [750, 50]]
    groups_goals = [[750, 750], [50, 570]]
    groups_agents = [50, 50]
    groups_colors = [[255, 128, 0], [135, 200, 240]]
    groups_fields = []
    # generate groups settings
    agent_params = {"init_agent_params": []}
    aid = 0
    for gid in range(group_n):
        gf_i = {"agent_ids": [], "field": None, "grid": None}
        for gaid in range(groups_agents[gid]):
            ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
            ai_params["pos"] = [random.uniform(groups_centers[gid][0]-40, groups_centers[gid][0]+40),
                                random.uniform(groups_centers[gid][1]-40, groups_centers[gid][1]+40)]
            ai_params["goal_pos"] = copy.deepcopy(groups_goals[gid])
            ai_params["radius"] = 3
            ai_params["pref_speed"] = 6
            ai_params["color"] = groups_colors[gid]
            agent_params["init_agent_params"].append(copy.deepcopy(ai_params))
            gf_i["agent_ids"].append(aid)
            aid += 1
        groups_fields.append(copy.deepcopy(gf_i))
    for rst_aid in range(aid, agent_n):
        ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
        ai_params["pos"] = [-1000, -1000]
        ai_params["radius"] = 3
        agent_params["init_agent_params"].append(copy.deepcopy(ai_params))


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

    fld_env = Field_Env(agent_num=agent_n, visual=True, draw_scale=1.0)
    fld_env.reset(scenario=copy.deepcopy(scenario_test), agent_setting=copy.deepcopy(agent_params))

    crv_fld = CurveTracking_Field(reverse_direction = False, vr = 1, kf = 0.008,
                                  flag_follow_obstacle = True, epsilon = 0, switch_dist_0 = 60, switch_dist = 40, lidar_N=256)

    # field for group 1
    grid_width = 20.0
    guidance1 = {
        "type": "lines",
        "params": {
            "lines":[[[50, 50], [50, 600]],[[50, 600], [300, 600]], [[300, 600], [300, 300]], [[300, 300], [500, 300]],
                     [[500, 300], [500, 500]], [[500, 500], [750, 750]]],
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
    crv_fld.reset(scenario_test, grid_width)
    field = crv_fld.get_field(guidance1, constrains)
    grid = copy.deepcopy(crv_fld.grid)
    groups_fields[0]["field"] = copy.deepcopy(field)
    groups_fields[0]["grid"] = copy.deepcopy(grid)
    crv_fld.field_visualization(field, guidance1)

    # field for group 2
    grid_width = 20.0
    guidance2 = {
        "type": "lines",
        "params": {
            "lines":[[[750, 50], [600, 600]],[[600, 600], [50, 570]]],
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
    crv_fld.reset(scenario_test, grid_width)
    field = crv_fld.get_field(guidance2, constrains)
    grid = copy.deepcopy(crv_fld.grid)
    groups_fields[1]["field"] = copy.deepcopy(field)
    groups_fields[1]["grid"] = copy.deepcopy(grid)
    crv_fld.field_visualization(field, guidance2)


    keyboard = pyglet.window.key.KeyStateHandler()
    while(1):
        fld_env.viewer.push_handlers(keyboard)
        if keyboard[pyglet.window.key.Q]:
            fld_env.set_viewer_field(groups_fields[0]["grid"], groups_fields[0]["field"])
        elif keyboard[pyglet.window.key.W]:
            fld_env.set_viewer_field(groups_fields[1]["grid"], groups_fields[1]["field"])
        elif  keyboard[pyglet.window.key.P]:
            fld_env.set_viewer_field(None, None)
        elif keyboard[pyglet.window.key.A]:
            fld_env.set_viewer_guidance(guidance1)
        elif keyboard[pyglet.window.key.S]:
            fld_env.set_viewer_guidance(guidance2)
        elif keyboard[pyglet.window.key.L]:
            fld_env.set_viewer_guidance(None)

        fld_env.render()
        for gid in range(len(groups_fields)):
            for aid in groups_fields[gid]["agent_ids"]:
                if np.linalg.norm(np.array(fld_env.agent_current_infor[aid]["pos"])-np.array(fld_env.agent_current_infor[aid]["goal_pos"]))<50. or \
                        fld_env.agent_current_infor[aid]["pos"][0]<0 or fld_env.agent_current_infor[aid]["pos"][0]>scenario_test["wind_size"][0] or \
                        fld_env.agent_current_infor[aid]["pos"][1]<0 or fld_env.agent_current_infor[aid]["pos"][1]>scenario_test["wind_size"][1]:
                    ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
                    ai_params["pos"] = [-1000, -1000]
                    fld_env.set_agent_params(aid, ai_params)
                    groups_fields[gid]["agent_ids"].remove(aid)
        fld_env.perform_action(groups_fields)
        time.sleep(0.04)

