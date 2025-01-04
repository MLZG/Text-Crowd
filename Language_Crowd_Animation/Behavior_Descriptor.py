import math
import random
from Utils import *
import numpy as np
import shapely.geometry as geom
import copy
from Simulators.ORCA_Env import ORCA_Env


SYMBOL_TO_TEXT = {
    # obj names
    "rectangle": ["rectangle", "oblong", "quadrilateral"], "triangle": ["triangle", "triangular shape"], "circle": ["circle", "ring", "round"],
    "zebra_crossing": ["zebra crossing", "pedestrian crossing", "crosswalk"],
    "passage": ["passage", "pathway", "corridor"],
    "entrance": ["entrance", "entry"], "exit": ["exit", "export"],
    # actions
    "enter_a": ["enters from", "gets in from", "moves from"], "exit_a": ["exits through", "leaves through", "quits through"],
    "pass_by": ["passes", "passes by", "moves past"], "pass_by_eg": ["passes", "passes by", "moves past"], "around": ["bypasses", "circles around"],
    "cross": ["crosses", "passes", "walks across", "moves across"], "through": ["moves through", "walks through", "passes through", "travels through"],
    # locations
    "right": ["right"], "upper_right": ["upper right"], "upper": ["upper"], "upper_left": ["upper left"], "left": ["left"],
    "lower_left": ["lower left"], "lower": ["lower"], "lower_right": ["lower right"], "top_right": ["top right"], "top": ["top"],
    "top_left": ["top left"], "bottom_left": ["bottom left"], "bottom": ["bottom"], "bottom_right": ["bottom right"], "middle": ["middle", "center"],
    # directions
    "anticlockwise": ["anticlockwise", "counterclockwise"], "clockwise": ["clockwise"],
    # group size
    "tiny": ["tiny"],
    "small": ["small"],
    "big": ["big"],
    "large": ["large"],
}


class Behavior_Descriptor:
    def __init__(self, snr_mid_scl=0.25, snr_inner_scl=0.66, global_adj_num=8, local_adj_num=8, dir_adj_num=8,
                 drop_out_ps={"action_loc": 0.0, "obj_loc": 0.0, "action_dir": 0.0}):
        self.mid_scl = snr_mid_scl
        self.inner_scl = snr_inner_scl
        self.global_adj_num = global_adj_num
        self.local_adj_num = local_adj_num
        self.dir_adj_num = dir_adj_num
        self.dropout_ps = drop_out_ps

    def path_to_descriptor(self, scenario_, path_v, path_e):
        itr_objs_list = []
        itr_v_list = []
        itr_e_list = []

        v_curr_id = 0
        while(v_curr_id<len(path_v)):
            if scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]] is None:
                v_curr_id += 1
                continue
            obj_key = scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]]["key"]
            obj_id = scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]]["obj_id"]
            vs = [path_v[v_curr_id]]
            es = []
            while(1):
                v_curr_id += 1
                if v_curr_id >= len(path_v) or scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]] is None:
                    break
                obj_key_nxt = scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]]["key"]
                obj_id_nxt = scenario_["roadmap"]["vertexes_idx_inS"][path_v[v_curr_id]]["obj_id"]
                if obj_key_nxt==obj_key and obj_id_nxt==obj_id:
                    vs.append(copy.deepcopy(path_v[v_curr_id]))
                    es.append(copy.deepcopy(path_e[v_curr_id-1]))
                else:
                    break
            itr_objs_list.append(copy.deepcopy(scenario_[obj_key][obj_id]))
            itr_v_list.append(copy.deepcopy(vs))
            itr_e_list.append(copy.deepcopy(es))

        path_descriptor = [{"group_size": random.choice(["tiny", "small", "big", "large"])}]
        for obj_id in range(len(itr_objs_list)):
            path_descriptor.append(self.get_itr_descriptor(copy.deepcopy(scenario_), copy.deepcopy(itr_objs_list[obj_id]),
                                                           copy.deepcopy(itr_v_list[obj_id]), copy.deepcopy(itr_e_list[obj_id])))
        return path_descriptor


    def descriptor_to_text(self, dsc_l, dropout_ps=None, symbol2text=None):
        # A [group_size] group
        # [action] ([action_loc]) ([obj_loc]) [obj_name] ([action_dir])
        # [action] ([action_loc]) ([obj_loc]) [obj_name] ([action_dir])
        # [action] ([action_loc]) ([obj_loc]) [obj_name] ([action_dir])
        # ...

        if dropout_ps is None:
            dropout_ps = copy.deepcopy(self.dropout_ps)
        if symbol2text is not None:
            s2t = copy.deepcopy(symbol2text)
        else:
            s2t = copy.deepcopy(SYMBOL_TO_TEXT)
        text_ = ""

        for dscid, dsc_i in enumerate(dsc_l):
            if dscid == 0:
                text_ += ("A " + random.choice(s2t[dsc_i["group_size"]]) + " group")
                continue
            itr_text = " " + random.choice(s2t[dsc_i["action"]]) + " the"
            if random.random() > dropout_ps["action_loc"] and dsc_i["action_loc"] is not None:
                itr_text += (" " + random.choice(s2t[dsc_i["action_loc"]]) + " side of the")
            if random.random() > dropout_ps["obj_loc"] and dsc_i["obj_loc"] is not None:
                itr_text += (" " + random.choice(s2t[dsc_i["obj_loc"]]))
            itr_text += (" " + random.choice(s2t[dsc_i["obj"]]))
            if random.random() > dropout_ps["action_dir"] and dsc_i["action_dir"] is not None:
                if dsc_i["action_dir"][0:4] == "from":
                    dir1 = copy.deepcopy(dsc_i["action_dir"][dsc_i["action_dir"].index("from")+5: dsc_i["action_dir"].index("to")-1])
                    dir2 = copy.deepcopy(dsc_i["action_dir"][dsc_i["action_dir"].index("to")+3:])
                    itr_text += (" from " + random.choice(s2t[dir1]) + " to " + random.choice(s2t[dir2]))
                else:
                    itr_text += (" " + random.choice(s2t[dsc_i["action_dir"]]))
            if dscid != len(dsc_l)-1:
                itr_text += ","
            else:
                itr_text += "."
            text_ += itr_text

        return text_


    def get_itr_descriptor(self, scenario_, obj, itr_points, itr_edges):
        itr_ppos_list = []
        itr_egps_list = []
        for p_i in itr_points:
            itr_ppos_list.append(copy.deepcopy(scenario_["roadmap"]["vertexes"][p_i]))
        for e_i in itr_edges:
            itr_egps_list.append([copy.deepcopy(scenario_["roadmap"]["vertexes"][e_i["edge"][0]]),
                                  copy.deepcopy(scenario_["roadmap"]["vertexes"][e_i["edge"][1]])])

        itr_desc_ = {
            "obj": None,
            "obj_loc": None,
            "action": None,
            "action_loc": None,
            "action_dir": None,
        }
        # obj type & location
        itr_desc_["obj"] = copy.deepcopy(obj["type"])
        obj_ctr = None
        if obj["type"] == "rectangle" or obj["type"] == "triangle":
            obj_ctr = ORCA_Env.get_obs_infor(copy.deepcopy(obj))["center"]
        else:
            obj_ctr = copy.deepcopy(obj["params"]["center"])
        itr_desc_["obj_loc"] = self.get_global_location(wind_size=copy.deepcopy(scenario_["wind_size"]), p_=copy.deepcopy(obj_ctr),
                                                        mid_scl=self.mid_scl, inner_scale=self.inner_scl,
                                                        adj_num=self.global_adj_num)
        # interaction type, location, and direction
        if obj["type"] == "entrance":
            itr_desc_["action"] = "enter_a"
        elif obj["type"] == "exit":
            itr_desc_["action"] = "exit_a"
        elif obj["type"] == "rectangle" or obj["type"] == "triangle" or obj["type"] == "circle":
            if len(itr_ppos_list)==1 or len(itr_ppos_list)==2:
                itr_desc_["action"] = "pass_by" if len(itr_ppos_list)==1 else "pass_by_eg"
                itr_desc_["action_loc"] = self.get_local_location(ctr_=copy.deepcopy(obj_ctr),
                                                                  p_=(np.sum(np.array(copy.deepcopy(itr_ppos_list)), axis=0)/len(itr_ppos_list)).tolist(),
                                                                  adj_num=self.local_adj_num)
                itr_desc_["action_dir"] = self.get_edges_dir(copy.deepcopy(itr_egps_list), adj_num=self.dir_adj_num)
            else:
                itr_desc_["action"] = "around"
                mid_ps = copy.deepcopy(itr_ppos_list[int((len(itr_ppos_list)-1)/2):int(len(itr_ppos_list)/2)+1])
                itr_desc_["action_loc"] = self.get_local_location(ctr_=copy.deepcopy(obj_ctr),
                                                                  p_=(np.sum(np.array(copy.deepcopy(mid_ps)), axis=0)/len(mid_ps)).tolist(),
                                                                  adj_num=self.local_adj_num)
                itr_desc_["action_dir"] = self.get_edges_dir(copy.deepcopy(itr_egps_list), adj_num=self.dir_adj_num)
        elif obj["type"] == "zebra_crossing" or obj["type"] == "passage":
            if len(itr_ppos_list)==1:
                itr_desc_["action"] = "pass_by"
                itr_desc_["action_loc"] = self.get_local_location(ctr_=copy.deepcopy(obj_ctr),
                                                                  p_=(np.sum(np.array(copy.deepcopy(itr_ppos_list)), axis=0)/len(itr_ppos_list)).tolist(),
                                                                  adj_num=self.local_adj_num)
            else:
                itr_desc_["action"] = "cross" if obj["type"] == "zebra_crossing" else "through"
                itr_desc_["action_dir"] = self.get_edges_dir(copy.deepcopy(itr_egps_list), adj_num=self.dir_adj_num)

        return itr_desc_


    def get_global_location(self, wind_size, p_, mid_scl=None, inner_scale=None, adj_num=None):
        if adj_num is None:
            adj_num = self.global_adj_num
        if mid_scl is None:
            mid_scl = self.mid_scl
        if inner_scale is None:
            inner_scale = self.inner_scl

        p_ori = (np.array(p_) - np.array([wind_size[0]/2, wind_size[1]/2])).tolist()
        if abs(p_ori[0])<=wind_size[0]*mid_scl/2 and abs(p_ori[1])<=wind_size[1]*mid_scl/2:
            return "middle"
        else:
            if adj_num == 8:
                st_vec = [math.cos(-math.pi*2/16), math.sin(-math.pi*2/16)]
                ag_ = (get_signed_angle([[0, 0], copy.deepcopy(st_vec)], [[0, 0], np.array(p_ori).tolist()]) + 360) % 360
                if abs(p_ori[0])<=wind_size[0]*inner_scale/2 and abs(p_ori[1])<=wind_size[1]*inner_scale/2:
                    ar_list = ["right", "upper_right", "upper", "upper_left", "left", "lower_left", "lower", "lower_right"]
                else:
                    ar_list = ["right", "top_right", "top", "top_left", "left", "bottom_left", "bottom", "bottom_right"]
                return ar_list[int(ag_/(360/8))]
            elif adj_num == 4:
                st_vec = [math.cos(-math.pi*2/8), math.sin(-math.pi*2/8)]
                ag_ = (get_signed_angle([[0, 0], copy.deepcopy(st_vec)], [[0, 0], np.array(p_ori).tolist()]) + 360) % 360
                if abs(p_ori[0])<=wind_size[0]*inner_scale/2 and abs(p_ori[1])<=wind_size[1]*inner_scale/2:
                    ar_list = ["right", "upper", "left",  "lower"]
                else:
                    ar_list = ["right", "top", "left", "bottom"]
                return ar_list[int(ag_/(360/4))]


    def get_local_location(self, ctr_, p_, adj_num=None):
        if adj_num is None:
            adj_num = self.local_adj_num
        p_ori = (np.array(p_) - np.array(ctr_)).tolist()
        if adj_num == 8:
            st_vec = [math.cos(-math.pi*2/16), math.sin(-math.pi*2/16)]
            ag_ = (get_signed_angle([[0, 0], copy.deepcopy(st_vec)], [[0, 0], np.array(p_ori).tolist()]) + 360) % 360
            ar_list = ["right", "upper_right", "upper", "upper_left", "left", "lower_left", "lower", "lower_right"]
            return ar_list[int(ag_/(360/8))]
        elif adj_num == 4:
            st_vec = [math.cos(-math.pi*2/8), math.sin(-math.pi*2/8)]
            ag_ = (get_signed_angle([[0, 0], copy.deepcopy(st_vec)], [[0, 0], np.array(p_ori).tolist()]) + 360) % 360
            ar_list = ["right", "upper", "left", "lower"]
            return ar_list[int(ag_/(360/4))]

    def get_edges_dir(self, edges_, adj_num=None):
        if adj_num is None:
            adj_num = self.dir_adj_num
        if edges_ is None or len(edges_)==0:
            return None
        elif len(edges_)==1:
            eg_ctr = (np.sum(np.array(copy.deepcopy(edges_[0])), axis=0)/len(edges_[0])).tolist()
            adj_from = self.get_local_location(ctr_=copy.deepcopy(eg_ctr), p_=copy.deepcopy(edges_[0][0]), adj_num=adj_num)
            adj_to = self.get_local_location(ctr_=copy.deepcopy(eg_ctr), p_=copy.deepcopy(edges_[0][1]), adj_num=adj_num)
            return "from_" + adj_from + "_to_" + adj_to
        else:
            vec1 = np.array(edges_[0][1]) - np.array(edges_[0][0])
            vec2 = np.array(edges_[1][1]) - np.array(edges_[1][0])
            dir_ = "anticlockwise" if np.cross(vec1, vec2)>=0 else "clockwise"
            return dir_
