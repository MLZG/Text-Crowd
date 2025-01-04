import math
import random
import time
import rasterio
from rasterio.features import geometry_mask
from Utils import *
import numpy as np
import shapely.geometry as geom
from Simulators.ORCA_Env import ORCA_Env, SCENARIO_DEFAULT, AGENT_SETTING_DEFAULT
import copy

EXAMPLE_SCENARIO_CONFIG = {
    "wind_size": [1024, 1024],
    "bound_srk_scale": 1/5,
    "safe_dis" : 80,
    "objects": {
        "rectangle": {"num": 1, "size_range":[40, 60], "graph_buffer": 20},
        "triangle": {"num": 1, "size_range":[40, 60], "graph_buffer": 20},
        "circle": {"num": 1, "size_range":[40, 60], "graph_buffer": 20},
        "zebra_crossing": {"num": 1, "size_range":[180, 230], "graph_buffer": 20},
        "passage": {"num": 1, "size_range":[160, 180], "graph_buffer": 20},
        "entrance": {"num": 2, "size_range":[60, 60], "graph_buffer": None},
        "exit": {"num": 1, "size_range":[60, 60], "graph_buffer": None}
    }
}

ALL_OBJECT_SEMANTICS = ["rectangle", "triangle", "circle", "zebra_crossing", "passage_obs", "passage_free", "entrance", "exit"]

SCENARIO_INIT = {
    "wind_size": None,
    "obs_list": [],
    "zebra_crossing_list": [],
    "passages_list": [],
    "areas_list": [],
}


class Scenario_Generator:
    def __init__(self):
        pass

    def random_scenario(self, scenario_configs):
        wind_size = copy.deepcopy(scenario_configs["wind_size"])
        srk_scale = scenario_configs["bound_srk_scale"]
        safe_dis = scenario_configs["safe_dis"]

        scenario_ = copy.deepcopy(SCENARIO_INIT)
        scenario_["wind_size"] = copy.deepcopy(wind_size)

        w_rg = [wind_size[0]*srk_scale/2, wind_size[0]*(1-srk_scale/2)]
        h_rg = [wind_size[1]*srk_scale/2, wind_size[1]*(1-srk_scale/2)]
        scenario_["wind_size_sub"] = [copy.deepcopy(w_rg), copy.deepcopy(h_rg)]

        boundary = geom.LineString(geom.Polygon(get_box_ll(x=wind_size[0]*(1-srk_scale),y=wind_size[1]*(1-srk_scale),
                                                           lowerleft=(w_rg[0],h_rg[0]))).exterior.coords).buffer(1, cap_style=3, join_style=2)
        obj_polys_list = [boundary]

        for obj_name in scenario_configs["objects"].keys():
            obj_num = scenario_configs["objects"][obj_name]["num"]
            obj_size_range = copy.deepcopy(scenario_configs["objects"][obj_name]["size_range"])
            for r_i in range(obj_num):
                obj, obj_poly, obj_fkey = None, None, None
                # get obj (except for attributes)
                loop_cnt = 0
                while(1):
                    loop_cnt += 1
                    if loop_cnt >= 30000:
                        return None
                    if obj_name == "entrance" or obj_name == "exit":
                        obj, obj_poly, obj_fkey = self.get_rand_obj(obj_name, copy.deepcopy(obj_size_range), [copy.deepcopy(w_rg), copy.deepcopy(h_rg)])
                        jd = True
                        for a_i in scenario_["areas_list"]:
                            dis_lim = ((wind_size[0]*(1-srk_scale) + wind_size[1]*(1-srk_scale))/2)/\
                                      (max(scenario_configs["objects"]["entrance"]["num"], scenario_configs["objects"]["exit"]["num"]))
                            dis_lim = max(dis_lim, scenario_configs["objects"][a_i["type"]]["size_range"][1]/2+scenario_configs["objects"][obj_name]["size_range"][1]/2)
                            if np.linalg.norm(np.array(obj["params"]["center"])-np.array(a_i["params"]["center"]))<dis_lim:
                                jd = False
                                break
                        if jd:
                            break
                    else:
                        obj, obj_poly, obj_fkey = self.get_rand_obj(obj_name, copy.deepcopy(obj_size_range), [copy.deepcopy(w_rg), copy.deepcopy(h_rg)])
                        if poly_collision_check(obj_polys_list, obj_poly.buffer(safe_dis)):
                            break

                # get obj's attributes
                obj_graph_ps, obj_graph_egs, obj_graph_box = self.get_obj_graph(copy.deepcopy(obj), scenario_configs["objects"][obj_name]["graph_buffer"])
                obj["attributes"]["graph"] = {"points": copy.deepcopy(obj_graph_ps), "edges": copy.deepcopy(obj_graph_egs),
                                              "graph_box": copy.deepcopy(obj_graph_box)}


                obj_polys_list.append(obj_poly)
                scenario_[obj_fkey].append(copy.deepcopy(obj))

        return scenario_


    def get_rand_obj(self, obj_type, objsize_range, area_):
        obj_infor = {"type": obj_type, "params": {}, "attributes": {}}
        if obj_type == "rectangle":
            ctr_ = [random.uniform(area_[0][0], area_[0][1]), random.uniform(area_[1][0], area_[1][1])]
            r_, ag_ = random.uniform(objsize_range[0], objsize_range[1])/2, random.uniform(30., 60.)
            w_, h_ = r_*math.cos(ag_/180.*math.pi)*2, r_*math.sin(ag_/180.*math.pi)*2
            rot = random.randint(-180, 180)
            box_ = vectors_rotation(np.array(get_box(w_, h_, [0., 0.])).reshape(-1, 2).tolist(), rot/180.*math.pi)
            box_ = (np.array(box_) + np.array(ctr_)).tolist()
            obj_infor["params"]["vertexes"] = copy.deepcopy(box_)
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(box_)])
            return obj_infor, obj_poly, "obs_list"
        elif obj_type == "triangle":
            ctr_ = [random.uniform(area_[0][0], area_[0][1]), random.uniform(area_[1][0], area_[1][1])]
            r_ = random.uniform(objsize_range[0], objsize_range[1])/2
            tri_ = []
            for rg_i in [[0., 60.], [120., 180.], [240., 300.]]:
                ag = random.uniform(rg_i[0], rg_i[1])
                tri_.append([r_*math.cos(ag/180.*math.pi), r_*math.sin(ag/180.*math.pi)])
            tri_ = (np.array(tri_) + np.array(ctr_)).tolist()
            obj_infor["params"]["vertexes"] = copy.deepcopy(tri_)
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(tri_)])
            return obj_infor, obj_poly, "obs_list"
        elif obj_type == "circle":
            ctr_ = [random.uniform(area_[0][0], area_[0][1]), random.uniform(area_[1][0], area_[1][1])]
            r_ = random.uniform(objsize_range[0], objsize_range[1])/2
            circle_ps = (np.array(get_circle(r_, fill=False, RES=32)).reshape(-1, 2) + np.array(ctr_)).tolist()
            obj_infor["params"] = {"center": copy.deepcopy(ctr_), "radius": r_}
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(circle_ps)])
            return obj_infor, obj_poly, "obs_list"
        elif obj_type == "zebra_crossing":
            ctr_ = [random.uniform(area_[0][0], area_[0][1]), random.uniform(area_[1][0], area_[1][1])]
            r_, ag_ = random.uniform(objsize_range[0], objsize_range[1])/2, random.uniform(20., 40.)
            w_, h_ = r_*math.cos(ag_/180.*math.pi)*2, r_*math.sin(ag_/180.*math.pi)*2
            zb_w_ = w_/15-1e-6
            rot = random.randint(-180, 180)
            box_ = vectors_rotation(np.array(get_box(w_, h_, [0., 0.])).reshape(-1, 2).tolist(), rot/180.*math.pi)
            box_ = (np.array(box_) + np.array(ctr_)).tolist()
            obj_infor["params"] = {"center": copy.deepcopy(ctr_), "width": w_, "height": h_, "zb_line_width": zb_w_, "rotation": rot}
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(box_)])
            return obj_infor, obj_poly, "zebra_crossing_list"
        elif obj_type == "passage":
            ctr_ = [random.uniform(area_[0][0], area_[0][1]), random.uniform(area_[1][0], area_[1][1])]
            r_, ag_ = random.uniform(objsize_range[0], objsize_range[1])/2, random.uniform(40., 60.)
            w_, h_ = r_*math.cos(ag_/180.*math.pi)*2, r_*math.sin(ag_/180.*math.pi)*2
            ps_w_ = w_ * random.uniform(0.75, 0.85)
            rot = random.randint(-180, 180)
            box_ = vectors_rotation(np.array(get_box(w_, h_, [0., 0.])).reshape(-1, 2).tolist(), rot/180.*math.pi)
            box_ = (np.array(box_) + np.array(ctr_)).tolist()
            obj_infor["params"] = {"center": copy.deepcopy(ctr_), "width": w_, "height": h_, "passage_width": ps_w_, "rotation": rot}
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(box_)])
            return obj_infor, obj_poly, "passages_list"
        elif obj_type == "entrance" or obj_type == "exit":
            w_ = random.uniform(objsize_range[0], objsize_range[1])
            h_ = w_ * 1.0
            edge_id = random.randint(0, 3)
            if edge_id == 0:
                ctr_ = [random.uniform(area_[0][0], area_[0][1]), area_[1][0]-h_/2]
                rot = 0
            elif edge_id == 1:
                ctr_ = [random.uniform(area_[0][0], area_[0][1]), area_[1][1]+h_/2]
                rot = 0
            elif edge_id == 2:
                ctr_ = [area_[0][0]-h_/2, random.uniform(area_[1][0], area_[1][1])]
                rot = 90
            elif edge_id == 3:
                ctr_ = [area_[0][1]+h_/2, random.uniform(area_[1][0], area_[1][1])]
                rot = 90
            box_ = vectors_rotation(np.array(get_box(w_, h_, [0., 0.])).reshape(-1, 2).tolist(), rot/180.*math.pi)
            box_ = (np.array(box_) + np.array(ctr_)).tolist()
            obj_infor["params"] =  {"center": copy.deepcopy(ctr_), "width": w_, "height": h_, "rotation": rot}
            obj_poly = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(box_)])
            return obj_infor, obj_poly, "areas_list"


    def get_obj_graph(self, obj, graph_buffer):
        edges_list = []
        points_list = []
        if obj["type"] == "rectangle":
            box_ = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(obj["params"]["vertexes"])]).buffer(graph_buffer, cap_style=3, join_style=2)
            points_list = np.array(list(box_.exterior.coords)[:-1]).tolist()
            for p_id in range(len(points_list)):
                edges_list.append([p_id, (p_id+1)%len(points_list)])
                edges_list.append([(p_id+1)%len(points_list), p_id])
            box_inner = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(obj["params"]["vertexes"])]).buffer(graph_buffer-1e-3, cap_style=3, join_style=2)
            return points_list, edges_list, copy.deepcopy(np.array(list(box_inner.exterior.coords)[:-1]).tolist())
        elif obj["type"] == "triangle":
            tri_ = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(obj["params"]["vertexes"])]).buffer(graph_buffer, cap_style=3, join_style=2)
            points_list = np.array(list(tri_.exterior.coords)[:-1]).tolist()
            for p_id in range(len(points_list)):
                edges_list.append([p_id, (p_id+1)%len(points_list)])
                edges_list.append([(p_id+1)%len(points_list), p_id])
            tri_inner = geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(obj["params"]["vertexes"])]).buffer(graph_buffer-1e-3, cap_style=3, join_style=2)
            return points_list, edges_list, copy.deepcopy(np.array(list(tri_inner.exterior.coords)[:-1]).tolist())
        elif obj["type"] == "circle":
            bbox_ = np.array(get_box(obj["params"]["radius"]*2,obj["params"]["radius"]*2,copy.deepcopy(obj["params"]["center"]))).reshape(-1, 2).tolist()
            box_ = geom.Polygon([[p[0],p[1]] for p in bbox_]).buffer(graph_buffer, cap_style=3, join_style=2)
            points_list = np.array(list(box_.exterior.coords)[:-1]).tolist()
            for p_id in range(len(points_list)):
                edges_list.append([p_id, (p_id+1)%len(points_list)])
                edges_list.append([(p_id+1)%len(points_list), p_id])
            box_inner = geom.Polygon([[p[0],p[1]] for p in bbox_]).buffer(graph_buffer-1e-3, cap_style=3, join_style=2)
            return points_list, edges_list, copy.deepcopy(np.array(list(box_inner.exterior.coords)[:-1]).tolist())
        elif obj["type"] == "zebra_crossing":
            obj_buffer = copy.deepcopy(obj)
            obj_buffer["params"]["width"] += graph_buffer*2
            obj_buffer["params"]["height"] += graph_buffer*2
            obj_ex_infor = ORCA_Env.get_zebra_crossing_infor(obj_buffer)
            for l_i in obj_ex_infor["in_out_lines"]:
                alpha_ = random.uniform(0.4, 0.6)
                points_list.append((np.array(l_i[0])*alpha_+np.array(l_i[1])*(1-alpha_)).tolist())
            edges_list.append([0, 1])
            edges_list.append([1, 0])
            side_lines = [[copy.deepcopy(obj_ex_infor["whole_box"][0]), copy.deepcopy(obj_ex_infor["whole_box"][1])],
                          [copy.deepcopy(obj_ex_infor["whole_box"][3]), copy.deepcopy(obj_ex_infor["whole_box"][2])]]
            for l_i in side_lines:
                alpha_ = random.uniform(0.4, 0.6)
                points_list.append((np.array(l_i[0])*alpha_+np.array(l_i[1])*(1-alpha_)).tolist())
            edges_list.append([2, 3])
            edges_list.append([3, 2])
            obj_buffer_inner = copy.deepcopy(obj)
            obj_buffer_inner["params"]["width"] += (graph_buffer-1e-3)*2
            obj_buffer_inner["params"]["height"] += (graph_buffer-1e-3)*2
            obj_ex_infor_inner = ORCA_Env.get_zebra_crossing_infor(obj_buffer_inner)
            return points_list, edges_list, copy.deepcopy(obj_ex_infor_inner["whole_box"])
        elif obj["type"] == "passage":
            obj_buffer = copy.deepcopy(obj)
            obj_buffer["params"]["width"] += graph_buffer*2
            obj_buffer["params"]["height"] += graph_buffer*2
            obj_ex_infor = ORCA_Env.get_passage_infor(obj_buffer)
            for l_i in obj_ex_infor["in_out_lines"]:
                alpha_ = 0.5
                points_list.append((np.array(l_i[0])*alpha_+np.array(l_i[1])*(1-alpha_)).tolist())
            edges_list.append([0, 1])
            edges_list.append([1, 0])
            obj_buffer_inner = copy.deepcopy(obj)
            obj_buffer_inner["params"]["width"] += (graph_buffer-1e-3)*2
            obj_buffer_inner["params"]["height"] += (graph_buffer-1e-3)*2
            obj_ex_infor_inner = ORCA_Env.get_passage_infor(obj_buffer_inner)
            return points_list, edges_list, copy.deepcopy(obj_ex_infor_inner["whole_box"])
        elif obj["type"] == "entrance" or obj["type"] == "exit":
            points_list = [copy.deepcopy(obj["params"]["center"])]
            edges_list = []
            return points_list, edges_list, []


    def get_feature(self, obj_semantic_name):
        all_obj_smtcs = copy.deepcopy(ALL_OBJECT_SEMANTICS)
        objsm_id = all_obj_smtcs.index(obj_semantic_name)
        feature_ = np.zeros(len(all_obj_smtcs))
        feature_[objsm_id] = 1
        return feature_

    def get_semantic_map(self, scenario_in, grid_width):
        if "extended" in scenario_in.keys() and scenario_in["extended"]:
            scenario_ = copy.deepcopy(scenario_in)
        else:
            scenario_ = ORCA_Env.get_scenario_extension(copy.deepcopy(scenario_in))

        grid_size = [int(scenario_["wind_size"][0]/grid_width), int(scenario_["wind_size"][1]/grid_width)]
        semantic_map_ = np.zeros((grid_size[0], grid_size[1], len(copy.deepcopy(ALL_OBJECT_SEMANTICS))))
        transform_ = rasterio.transform.from_bounds(0, 0, scenario_["wind_size"][0], scenario_["wind_size"][1], grid_size[0], grid_size[1])

        for key_i in scenario_.keys():
            if key_i == "wind_size" or key_i == "wind_size_sub" or key_i == "roadmap" or key_i == "extended":
                continue
            for obj_id in range(len(scenario_[key_i])):
                if scenario_[key_i][obj_id]["type"] == "passage":
                    poly_free = geom.Polygon(copy.deepcopy(scenario_[key_i][obj_id]["ext_params"]["free_space"]))
                    mask_free = rasterio.features.geometry_mask([poly_free], out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
                    mask_free = np.flip(mask_free, axis=0).transpose(1, 0)
                    semantic_map_[~mask_free] = np.array(self.get_feature("passage_free"))
                    for obs_id in range(2):
                        poly_obs = geom.Polygon(copy.deepcopy(scenario_[key_i][obj_id]["ext_params"]["obstacles"][obs_id]))
                        mask_obs = rasterio.features.geometry_mask([poly_obs], out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
                        mask_obs = np.flip(mask_obs, axis=0).transpose(1, 0)
                        semantic_map_[~mask_obs] = np.array(self.get_feature("passage_obs"))
                else:
                    if scenario_[key_i][obj_id]["type"] == "rectangle" or scenario_[key_i][obj_id]["type"] == "triangle":
                        poly_ = geom.Polygon(copy.deepcopy(scenario_[key_i][obj_id]["params"]["vertexes"]))
                    elif scenario_[key_i][obj_id]["type"] == "circle":
                        poly_ = geom.Polygon(copy.deepcopy(scenario_[key_i][obj_id]["ext_params"]["edges"]))
                    else:
                        poly_ = geom.Polygon(copy.deepcopy(scenario_[key_i][obj_id]["ext_params"]["whole_box"]))
                    geom_mask = rasterio.features.geometry_mask([poly_], out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
                    geom_mask = np.flip(geom_mask, axis=0).transpose(1, 0)
                    semantic_map_[~geom_mask] = np.array(self.get_feature(scenario_[key_i][obj_id]["type"]))

        return semantic_map_


    def get_sg_distb(self, scenario_in, grid_width, sg_areas):
        if "extended" in scenario_in.keys() and scenario_in["extended"]:
            scenario_ = copy.deepcopy(scenario_in)
        else:
            scenario_ = ORCA_Env.get_scenario_extension(copy.deepcopy(scenario_in))

        grid_size = [int(scenario_["wind_size"][0]/grid_width), int(scenario_["wind_size"][1]/grid_width)]
        sg_distribution = np.zeros((grid_size[0], grid_size[1], 2))
        transform_ = rasterio.transform.from_bounds(0, 0, scenario_["wind_size"][0], scenario_["wind_size"][1], grid_size[0], grid_size[1])

        for obj_id in range(len(scenario_["areas_list"])):
            if obj_id == sg_areas["start_area"] or obj_id == sg_areas["goal_area"]:
                poly_ = geom.Polygon(copy.deepcopy(scenario_["areas_list"][obj_id]["ext_params"]["whole_box"]))
                geom_mask = rasterio.features.geometry_mask([poly_], out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
                geom_mask = np.flip(geom_mask, axis=0).transpose(1, 0)
                sg_distribution[~geom_mask] = np.array([1, 0]) if obj_id == sg_areas["start_area"] else np.array([0, 1])

        return sg_distribution
