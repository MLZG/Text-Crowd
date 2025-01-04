import operator
import time

from Viewer.Viewer import Viewer
from Utils import *
import math,rvo2
import numpy as np
import copy

SCENARIO_DEFAULT = {
    "wind_size": [800, 800],
    "obs_list": [{"type": "rectangle", "params": {"vertexes": [[100, 400], [200, 400], [200, 500], [100, 500]]}, "attributes": {}},
                 {"type": "triangle", "params": {"vertexes": [[150, 150], [250, 150], [150, 250]]}, "attributes": {}},
                 {"type": "circle", "params": {"center": [400, 250], "radius": 50}, "attributes": {}}
                 ],
    "zebra_crossing_list": [{"type": "zebra_crossing",
                             "params": {"center": [600, 180], "width": 300, "height": 70, "zb_line_width": 20, "rotation": 30},
                             "attributes": {}},
                            {"type": "zebra_crossing",
                             "params": {"center": [650, 500], "width": 300, "height": 100, "zb_line_width": 20, "rotation": 90},
                             "attributes": {}}
                            ],
    "passages_list": [{"type": "passage",
                       "params": {"center": [350, 550], "width": 140, "height": 200, "passage_width": 80, "rotation": -30},
                       "attributes": {}}
                      ],
    "areas_list": [{"type": "entrance",  # no direction
                    "params": {"center": [80, 730], "width": 80, "height": 80, "rotation": 0},
                    "attributes": {}},
                   {"type": "exit",
                    "params": {"center": [730, 80], "width": 80, "height": 80, "rotation": 0},
                    "attributes": {}}
                   ],
    }


AGNET_PARAM_DEFAULT = {
    "pos": [0., 0.],
    "goal_pos": [0., 0.],
    "nb_Dist": 50,  # 50
    "max_nbs": 10,  # 10
    "timeH": 5,    # 15
    "timeH_Obst": 10,   # 10
    "radius": 8,
    "pref_speed": 5,
    "maxSpd": 20,
    "vlcty": None,
    "color": [124,79,13],
    "draw_goal": False,
    "draw_traj": False,
    "draw_sensor": False,
    "traj_history": []
}

AGENT_NUM_DEFAULT = 5

AGENT_SETTING_DEFAULT = {
    "init_agent_params": [copy.deepcopy(AGNET_PARAM_DEFAULT) for i in range(AGENT_NUM_DEFAULT)],
}

class ORCA_Env:
    def __init__(self, agent_num=AGENT_NUM_DEFAULT, visual=True, draw_scale=1.0):
        self.agent_num = agent_num
        self.visual = visual
        self.draw_scale = draw_scale

        self.viewer = None
        self.sensor = None
        self.current_scenario = None
        self.agent_idx_list = []
        self.agent_current_infor = []
        self.time_step = 0

        self.sim = self.sim_prepare()
        for i in range(self.agent_num):
            agent_ = self.add_agent_sim(0, 0)
            self.agent_idx_list.append(agent_)
            self.agent_current_infor.append(copy.deepcopy(AGNET_PARAM_DEFAULT))

        self.reset(scenario=copy.deepcopy(SCENARIO_DEFAULT), agent_setting={"init_agent_params": copy.deepcopy(self.agent_current_infor)})


    def reset(self, scenario, agent_setting):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.sensor = None
        self.time_step = 0

        # preprocess the scenario and agent_settings
        scenario = self.get_scenario_extension(scenario_=copy.deepcopy(scenario))

        # reset env in sim
        self.current_scenario = copy.deepcopy(scenario)
        self.reset_sim_scenario(self.current_scenario)

        for i in range(len(self.agent_idx_list)):
            self.agent_current_infor[i] = copy.deepcopy(agent_setting["init_agent_params"][i])
            self.set_agent_params(i, agent_setting["init_agent_params"][i])

        # reset env in viewer
        if self.visual:
            self.viewer = Viewer(wind_size=(int(scenario["wind_size"][0]*self.draw_scale), int(scenario["wind_size"][1]*self.draw_scale)),
                                 checker=(int(scenario["wind_size"][0]*self.draw_scale), int(scenario["wind_size"][1]*self.draw_scale), [235,235,235]))
            self.reset_viewer(self.current_scenario, self.agent_current_infor)
        else:
            self.viewer = None

        return

    ######------ functions related to simulator------######
    def sim_prepare(self):
        return rvo2.PyRVOSimulator(timeStep=1,
                                   neighborDist=AGNET_PARAM_DEFAULT["nb_Dist"],
                                   maxNeighbors=AGNET_PARAM_DEFAULT["max_nbs"],
                                   timeHorizon=AGNET_PARAM_DEFAULT["timeH"],
                                   timeHorizonObst=AGNET_PARAM_DEFAULT["timeH_Obst"],
                                   radius=AGNET_PARAM_DEFAULT["radius"],
                                   maxSpeed=AGNET_PARAM_DEFAULT["maxSpd"])

    def reset_sim_scenario(self, scenario):
        self.sim.clearObstacle()
        obs_list = self.get_all_obstacles_from_scenario(scenario_in=copy.deepcopy(scenario))
        for obs_i in obs_list:
            self.sim.addObstacle(make_ccw([tuple(p) for p in obs_i]))
        self.sim.processObstacles()

    def sim_step(self):
        self.sim.doStep()

    def perform_action(self, actions):
        pre_ps = self.get_current_positions()
        actions = np.array(actions).reshape(-1, 2)
        # perform action
        for i in range(len(self.agent_idx_list)):
            dx=actions[i][0]
            dy=actions[i][1]
            len_a=math.sqrt(dx*dx+dy*dy)
            if len_a>self.agent_current_infor[i]["maxSpd"]:
                dx*= self.agent_current_infor[i]["maxSpd"] / len_a
                dy*= self.agent_current_infor[i]["maxSpd"] / len_a
            self.sim.setAgentPrefVelocity(self.agent_idx_list[i], (dx,dy))
        self.sim_step()

        # update infor
        self.time_step += 1
        for i in range(len(self.agent_idx_list)):
            curr_p_i = self.sim.getAgentPosition(self.agent_idx_list[i])
            self.agent_current_infor[i]["pos"] = np.array(curr_p_i).tolist()
            self.agent_current_infor[i]["traj_history"].append([pre_ps[i].tolist(), actions[i].tolist()])

    ######------ functions related to agent setting ------######
    def add_agent_sim(self, px, py):
        return self.sim.addAgent(pos=tuple([px, py]))


    def set_agent_params(self, a_idx, agent_param_dict):
        # set param in sim
        agent_id = self.agent_idx_list[a_idx]
        if agent_param_dict["pos"] is not None:
            self.sim.setAgentPosition(agent_id, tuple(agent_param_dict["pos"]))
        if agent_param_dict["nb_Dist"] is not None:
            self.sim.setAgentNeighborDist(agent_id, agent_param_dict["nb_Dist"])
        if agent_param_dict["max_nbs"] is not None:
            self.sim.setAgentMaxNeighbors(agent_id, agent_param_dict["max_nbs"])
        if agent_param_dict["timeH"] is not None:
            self.sim.setAgentTimeHorizon(agent_id, agent_param_dict["timeH"])
        if agent_param_dict["timeH_Obst"] is not None:
            self.sim.setAgentTimeHorizonObst(agent_id, agent_param_dict["timeH_Obst"])
        if agent_param_dict["radius"] is not None:
            self.sim.setAgentRadius(agent_id, agent_param_dict["radius"])
        if agent_param_dict["maxSpd"] is not None:
            self.sim.setAgentMaxSpeed(agent_id, agent_param_dict["maxSpd"])
        if agent_param_dict["vlcty"] is not None:
            self.sim.setAgentVelocity(agent_id, agent_param_dict["vlcty"])
        # update current agent infor
        for key_i in agent_param_dict.keys():
            if agent_param_dict[key_i] is not None:
                self.agent_current_infor[a_idx][key_i] = agent_param_dict[key_i]

    def set_agent_position(self, a_idx, pos):
        agent_id = self.agent_idx_list[a_idx]
        # set agent position in simulator
        self.sim.setAgentPosition(agent_id, tuple(pos))
        # update current agent infor
        self.agent_current_infor[a_idx]["pos"] = copy.deepcopy(pos)

    ######------ functions related to agent information ------######
    def get_current_positions(self):
        current_positions = []
        for agent_ in self.agent_idx_list:
            pos = self.sim.getAgentPosition(agent_)
            current_positions.append([pos[0], pos[1]])
        current_positions = np.array(current_positions)
        return current_positions

    def update_current_positions(self):
        for i, agent_ in enumerate(self.agent_idx_list):
            pos = self.sim.getAgentPosition(agent_)
            self.agent_current_infor[i]["pos"]= [pos[0], pos[1]]
        return

    ######------ functions related to viewer ------######
    def reset_viewer(self, scenario_init, agent_setting_init):
        if "extended" in scenario_init.keys() and scenario_init["extended"]:
            scenario = copy.deepcopy(scenario_init)
        else:
            scenario = self.get_scenario_extension(copy.deepcopy(scenario_init))
        agent_setting = copy.deepcopy(agent_setting_init)
        self.viewer.reset_array()

        # set scenarios
        if "obs_list" in scenario.keys():
            for obs_i in scenario["obs_list"]:
                if obs_i["type"] == "rectangle" or obs_i["type"] == "triangle":
                    self.viewer.add_obs((copy.deepcopy(np.array(obs_i["params"]["vertexes"]))*self.draw_scale).reshape(1, -1)[0].tolist())
                elif obs_i["type"] == "circle":
                    self.viewer.add_obs((copy.deepcopy(np.array(obs_i["ext_params"]["edges"]))*self.draw_scale).reshape(1, -1)[0].tolist())
        if "zebra_crossing_list" in scenario.keys():
            for zebra_crs_i in scenario["zebra_crossing_list"]:
                for box_i in zebra_crs_i["ext_params"]["zebra_lines_boxes"]:
                    self.viewer.add_zebra_box((copy.deepcopy(np.array(box_i))*self.draw_scale).reshape(1, -1)[0].tolist())

        if "passages_list" in scenario.keys():
            for passage_i in scenario["passages_list"]:
                for psg_obs_i in passage_i["ext_params"]["obstacles"]:
                    self.viewer.add_obs((copy.deepcopy(np.array(psg_obs_i))*self.draw_scale).reshape(1, -1)[0].tolist())

        if "areas_list" in scenario.keys():
            for area_i in scenario["areas_list"]:
                if area_i["type"]=="entrance":
                    area_color = [140, 235, 205]
                elif area_i["type"]=="exit":
                    area_color = [250, 155, 155]
                self.viewer.add_door_box((copy.deepcopy(np.array(area_i["ext_params"]["whole_box"]))*self.draw_scale).reshape(1, -1)[0].tolist(), area_color)


        # set agents
        for agent_id in range(len(agent_setting)):
            agent_i_setting = agent_setting[agent_id]
            self.viewer.add_agent(pos=tuple(np.array(agent_i_setting["pos"])*self.draw_scale),
                                  rad=agent_i_setting["radius"]*self.draw_scale, color=agent_i_setting["color"])
            if agent_i_setting["draw_goal"]:
                self.viewer.add_goal(pos=tuple(np.array(agent_i_setting["goal_pos"])*self.draw_scale),
                                     goal_size=agent_i_setting["radius"]/2*self.draw_scale, color=agent_i_setting["color"])
            else:
                self.viewer.add_goal(pos=None)

        # set sensor
        if hasattr(self, 'sensor') and self.sensor is not None:
            self.viewer.sensor = self.sensor


    def render(self):
        for idx, agent_i_setting in enumerate(self.agent_current_infor):
            self.viewer.agent_pos_array[idx] = (np.array(agent_i_setting["pos"])*self.draw_scale).tolist()
            if agent_i_setting["draw_goal"]:
                self.viewer.goal_pos_array[idx] = (np.array(agent_i_setting["goal_pos"])*self.draw_scale).tolist()
            else:
                self.viewer.goal_pos_array[idx] = None
        self.viewer.render()


    ######------ other functions ------######
    @staticmethod
    def get_scenario_extension(scenario_):
        if "extended" in scenario_.keys() and scenario_["extended"]:
            print("The scenario has already been extended.")
            return
        scenario_extend = copy.deepcopy(scenario_)
        scenario_extend["extended"] = True
        if "obs_list" in scenario_.keys():
            for obs_id, obs_i in enumerate(scenario_["obs_list"]):
                scenario_extend["obs_list"][obs_id]["ext_params"] = ORCA_Env.get_obs_infor(obs_i)
        if "zebra_crossing_list" in scenario_.keys():
            for zebra_id, zebra_i in enumerate(scenario_["zebra_crossing_list"]):
                scenario_extend["zebra_crossing_list"][zebra_id]["ext_params"] = ORCA_Env.get_zebra_crossing_infor(zebra_i)
        if "passages_list" in scenario_.keys():
            for psg_id, psg_i in enumerate(scenario_["passages_list"]):
                scenario_extend["passages_list"][psg_id]["ext_params"] = ORCA_Env.get_passage_infor(psg_i)
        if "areas_list" in scenario_.keys():
            for area_id, area_i in enumerate(scenario_["areas_list"]):
                scenario_extend["areas_list"][area_id]["ext_params"] = ORCA_Env.get_area_infor(area_i)
        return scenario_extend

    @staticmethod
    def get_obs_infor(obs):
        if obs["type"] == "rectangle" or obs["type"] == "triangle":
            rec_vs = np.array(copy.deepcopy(obs["params"]["vertexes"]))
            rec_center = np.sum(rec_vs, axis=0) / len(rec_vs)
            return {"center": rec_center.tolist()}
        elif obs["type"] == "circle":
            circle_edges = np.array(get_circle(r=obs["params"]["radius"], fill=False, RES=int(obs["params"]["radius"]/10)*16)).reshape(-1, 2) + np.array(obs["params"]["center"])
            return {"edges": circle_edges.tolist()}

    @staticmethod
    def get_zebra_crossing_infor(zebra_crossing):
        w_ = zebra_crossing["params"]["width"]
        h_ = zebra_crossing["params"]["height"]
        whole_box = np.array([[-w_/2, -h_/2],[w_/2, -h_/2],[w_/2, h_/2],[-w_/2, h_/2]])
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(whole_box[vc_id], zebra_crossing["params"]["rotation"]/180*math.pi)
        whole_box += np.array(zebra_crossing["params"]["center"])

        in_out_lines = np.array([[copy.deepcopy(whole_box[0]), copy.deepcopy(whole_box[3])], [copy.deepcopy(whole_box[1]), copy.deepcopy(whole_box[2])]])

        boxes_ = []
        x_ = - zebra_crossing["params"]["width"] / 2
        y_ = - zebra_crossing["params"]["height"] / 2
        lw_ = zebra_crossing["params"]["zb_line_width"]
        lh_ = zebra_crossing["params"]["height"]
        while(x_+lw_<=zebra_crossing["params"]["width"]/2):
            boxes_.append([[x_, y_], [x_+lw_, y_], [x_+lw_, y_+lh_], [x_, y_+lh_]])
            x_ += lw_*2
        boxes_ = np.array(boxes_)
        for box_id in range(len(boxes_)):
            for vc_id in range(len(boxes_[0])):
                boxes_[box_id][vc_id] = vector_rotation(boxes_[box_id][vc_id], zebra_crossing["params"]["rotation"]/180*math.pi)
        boxes_ += np.array(zebra_crossing["params"]["center"])

        return {"whole_box":whole_box.tolist(), "in_out_lines":in_out_lines.tolist(), "zebra_lines_boxes":boxes_.tolist()}

    @staticmethod
    def get_passage_infor(psg):
        w_ = psg["params"]["width"]
        h_ = psg["params"]["height"]
        free_w = psg["params"]["passage_width"]
        whole_box = np.array([[-w_/2, -h_/2],[w_/2, -h_/2],[w_/2, h_/2],[-w_/2, h_/2]])
        free_box = np.array([[-free_w/2, -h_/2],[free_w/2, -h_/2],[free_w/2, h_/2],[-free_w/2, h_/2]])
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(whole_box[vc_id], psg["params"]["rotation"]/180*math.pi)
            free_box[vc_id] = vector_rotation(free_box[vc_id], psg["params"]["rotation"]/180*math.pi)
        whole_box += np.array(psg["params"]["center"])
        free_box += np.array(psg["params"]["center"])
        obstacles = np.array([[copy.deepcopy(whole_box[0]), copy.deepcopy(free_box[0]), copy.deepcopy(free_box[3]), copy.deepcopy(whole_box[3])],
                              [copy.deepcopy(free_box[1]), copy.deepcopy(whole_box[1]), copy.deepcopy(whole_box[2]), copy.deepcopy(free_box[2])]])
        in_out_lines = np.array([[copy.deepcopy(free_box[0]), copy.deepcopy(free_box[1])], [copy.deepcopy(free_box[3]), copy.deepcopy(free_box[2])]])
        return {"whole_box": whole_box.tolist(), "obstacles": obstacles.tolist(), "free_space": free_box.tolist(), "in_out_lines": in_out_lines.tolist()}

    @staticmethod
    def get_area_infor(area):
        w_ = area["params"]["width"]
        h_ = area["params"]["height"]
        whole_box = np.array([[-w_/2, -h_/2],[w_/2, -h_/2],[w_/2, h_/2],[-w_/2, h_/2]])
        for vc_id in range(len(whole_box)):
            whole_box[vc_id] = vector_rotation(whole_box[vc_id], area["params"]["rotation"]/180*math.pi)
        whole_box += np.array(area["params"]["center"])
        return {"whole_box": whole_box.tolist()}

    @staticmethod
    def get_all_obstacles_from_scenario(scenario_in):
        scenario_ = None
        if "extended" in scenario_in.keys() and scenario_in["extended"]:
            scenario_ = copy.deepcopy(scenario_in)
        else:
            scenario_ = ORCA_Env.get_scenario_extension(copy.deepcopy(scenario_in))

        all_obs_list = []
        if "obs_list" in scenario_.keys():
            for obs_i in scenario_["obs_list"]:
                if obs_i["type"] == "rectangle" or obs_i["type"] == "triangle":
                    all_obs_list.append(copy.deepcopy(np.array(obs_i["params"]["vertexes"])).tolist())
                elif obs_i["type"] == "circle":
                    all_obs_list.append(copy.deepcopy(np.array(obs_i["ext_params"]['edges'])).tolist())
        if "passages_list" in scenario_.keys():
            for passage_i in scenario_["passages_list"]:
                for psg_obs_i in passage_i["ext_params"]["obstacles"]:
                    all_obs_list.append(copy.deepcopy(np.array(psg_obs_i)).tolist())

        return all_obs_list



if __name__ == '__main__':
    orca_env = ORCA_Env()
    while(1):
        orca_env.perform_action([[1, 1], [0.5, 1.4], [1, 1.4], [1.2 ,1], [1.4, 1]])
        orca_env.render()
        time.sleep(0.01)


