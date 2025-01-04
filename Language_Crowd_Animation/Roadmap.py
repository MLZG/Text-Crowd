from Simulators.ORCA_Env import ORCA_Env
from Viewer.Viewer import Viewer
from Utils import *

import numpy as np
import shapely.geometry as geom
import math,time,os,random,trimesh,copy
from scipy.spatial import distance
import matplotlib.pyplot as plt

class Roadmap:
    def __init__(self):
        pass

    # Build a roadmap based on the scenario (window size, objects, object subgraphs)
    def build_roadmap_PRM(self, scenario_, sample_num=20, nb_dis=500):
        obj_poly_list = []
        scenario_expand = copy.deepcopy(scenario_)

        # Initialize the roadmap based on subgraphs in scenario_
        vertexes_ = []
        v_idx_inS = []
        edges_ = []
        e_idx_inS = []
        for key_i in scenario_.keys():
            if key_i == "wind_size" or key_i == "wind_size_sub" or key_i == "roadmap" or key_i == "extended":
                continue
            for obj_id in range(len(scenario_[key_i])):
                obj_graph = copy.deepcopy(scenario_[key_i][obj_id]["attributes"]["graph"])
                if "points_idx_inRM" not in scenario_expand[key_i][obj_id]["attributes"]["graph"].keys():
                    scenario_expand[key_i][obj_id]["attributes"]["graph"]["points_idx_inRM"] = []
                if "edges_idx_inRM" not in scenario_expand[key_i][obj_id]["attributes"]["graph"].keys():
                    scenario_expand[key_i][obj_id]["attributes"]["graph"]["edges_idx_inRM"] = []
                for e_id in range(len(obj_graph["edges"])):
                    edges_.append([obj_graph["edges"][e_id][0]+len(vertexes_), obj_graph["edges"][e_id][1]+len(vertexes_)])
                    e_idx_inS.append({"key": copy.deepcopy(key_i), "obj_id": obj_id, "edge_id": e_id})
                    scenario_expand[key_i][obj_id]["attributes"]["graph"]["edges_idx_inRM"].append(len(edges_)-1)
                for p_id in range(len(obj_graph["points"])):
                    vertexes_.append(copy.deepcopy(obj_graph["points"][p_id]))
                    v_idx_inS.append({"key": copy.deepcopy(key_i), "obj_id": obj_id, "point_id": p_id})
                    scenario_expand[key_i][obj_id]["attributes"]["graph"]["points_idx_inRM"].append(len(vertexes_)-1)
                obj_poly_list.append(geom.Polygon([[p[0],p[1]] for p in copy.deepcopy(obj_graph["graph_box"])]))

        # add random points
        rand_range = copy.deepcopy(scenario_["wind_size_sub"])
        for r_pi in range(sample_num):
            while(1):
                rand_p = [random.uniform(rand_range[0][0], rand_range[0][1]),
                          random.uniform(rand_range[1][0], rand_range[1][1])]
                if poly_collision_check(obj_poly_list, geom.Point(rand_p)):
                    break
            vertexes_.append(copy.deepcopy(rand_p))
            v_idx_inS.append(None)

        # connect neighbor points
        dist_mat = distance.cdist(np.array(vertexes_), np.array(vertexes_))
        new_edges = np.argwhere(dist_mat<=nb_dis)
        for neg_i in new_edges:
            neg_i_vs = [copy.deepcopy(vertexes_[neg_i[0]]), copy.deepcopy(vertexes_[neg_i[1]])]
            if (neg_i[0] == neg_i[1]) or (not poly_collision_check(obj_poly_list, geom.LineString(np.array(neg_i_vs).tolist()))):
                continue
            if np.array(neg_i).tolist() not in edges_:
                edges_.append(np.array(copy.deepcopy(neg_i)).tolist())
                e_idx_inS.append(None)

        scenario_expand["roadmap"] = {"vertexes": copy.deepcopy(vertexes_),
                                      "vertexes_idx_inS": copy.deepcopy(v_idx_inS),
                                      "edges": copy.deepcopy(edges_),
                                      "edges_idx_inS": copy.deepcopy(e_idx_inS)}

        return copy.deepcopy(scenario_expand), copy.deepcopy(scenario_expand["roadmap"])

    # Get a path based on the roadmap and several constrains
    def sample_path(self, roadmap, constrains):
        graph_ = [[] for i in range(len(roadmap["vertexes"]))]
        for eg_id in range(len(roadmap["edges"])):
            graph_[roadmap["edges"][eg_id][0]].append({"v": roadmap["edges"][eg_id][1], "edge_id": eg_id})

        # DFS-based adaptive sampling
        st_ = [[constrains["start_p"], [constrains["start_p"]], []]]
        search_cnt = 0
        while(st_):
            if search_cnt > 1e4:
                break
            search_cnt += 1
            v_, path_v, path_e = st_.pop()
            if v_ == constrains["goal_p"] and (len(path_e)>=constrains["path_len_range"][0] and len(path_e)<=constrains["path_len_range"][1]):
                return path_v , path_e

            # re-sort the nxt_vs randomly with adaptive weight
            nxt_vs = copy.deepcopy(graph_[v_])
            nxt_vs_avl = []
            for nvid, nv_i in enumerate(nxt_vs):
                if nv_i["v"] in path_v:
                    continue
                if roadmap["vertexes_idx_inS"][nv_i["v"]] is not None and roadmap["vertexes_idx_inS"][nv_i["v"]]["key"]=="areas_list" \
                        and nv_i["v"]!=constrains["goal_p"]:
                    continue
                if len(path_e)!=0:
                    pre_e = [copy.deepcopy(roadmap["vertexes"][path_e[-1]["edge"][1]]),
                             copy.deepcopy(roadmap["vertexes"][path_e[-1]["edge"][0]])]
                    cur_e = [copy.deepcopy(roadmap["vertexes"][v_]),
                             copy.deepcopy(roadmap["vertexes"][nv_i["v"]])]
                    if get_angle(np.array(pre_e), np.array(cur_e)) < constrains["line_angle_lim"]:
                        continue
                inter_ = False
                for eg_pre_i in copy.deepcopy(path_e)[:-1]:
                    cur_e = [copy.deepcopy(roadmap["vertexes"][v_]),
                             copy.deepcopy(roadmap["vertexes"][nv_i["v"]])]
                    eg_ = [copy.deepcopy(roadmap["vertexes"][eg_pre_i["edge"][0]]),
                           copy.deepcopy(roadmap["vertexes"][eg_pre_i["edge"][1]])]
                    if geom.LineString(np.array(copy.deepcopy(cur_e)).tolist()).\
                            intersects(geom.LineString(np.array(copy.deepcopy(eg_)).tolist()).buffer(constrains["safe_dis"])):
                        inter_ = True
                        break
                if inter_:
                    continue
                nxt_vs_avl.append(copy.deepcopy(nv_i))

            nxt_vs_w = []
            for nv_i in nxt_vs_avl:
                w_ = 1.
                if ((roadmap["vertexes_idx_inS"][nv_i["v"]] is not None)):
                    w_ *= constrains["adaptive_w"]["v_w"]
                else:
                    w_ *= (1-constrains["adaptive_w"]["v_w"])
                if (roadmap["edges_idx_inS"][nv_i["edge_id"]] is not None):
                    w_ *= constrains["adaptive_w"]["e_w"]
                else:
                    w_ *= (1-constrains["adaptive_w"]["e_w"])
                nxt_vs_w.append(w_)

            nxt_vs_new = []
            while len(nxt_vs_avl) != 0:
                v_idx = nxt_vs_avl.index(random.choices(nxt_vs_avl, weights=nxt_vs_w)[0])
                nxt_vs_new.append(copy.deepcopy(nxt_vs_avl[v_idx]))
                del nxt_vs_avl[v_idx]
                del nxt_vs_w[v_idx]

            for nxt_vi in list(reversed(nxt_vs_new)):
                st_.append([nxt_vi["v"], path_v + [nxt_vi["v"]], path_e + [{"edge": [v_, nxt_vi["v"]], "edge_id": nxt_vi["edge_id"]}]])

        return None, None


    def path_postprocess(self, scenario_, path_v, path_e, rlx_dis):
        G_vs = copy.deepcopy(scenario_["roadmap"]["vertexes"])
        G_es = copy.deepcopy(scenario_["roadmap"]["edges"])
        path_v_ = copy.deepcopy(path_v)
        path_e_ = copy.deepcopy(path_e)

        while(1):
            rlx_cnt = 0
            for vid, v_i in enumerate(G_vs):
                if vid in path_v_:
                    continue
                if scenario_["roadmap"]["vertexes_idx_inS"][vid] is None or scenario_["roadmap"]["vertexes_idx_inS"][vid]["key"]=="areas_list":
                    continue

                dist_list = []
                for eg_id, eg_i in enumerate(path_e_):
                    dist_list.append(geom.Point(list(copy.deepcopy(v_i))).distance(geom.LineString([list(copy.deepcopy(G_vs[eg_i["edge"][0]])),
                                                                                          list(copy.deepcopy(G_vs[eg_i["edge"][1]]))])))
                sorted_egids = np.argsort(np.array(dist_list))
                for min_dis_egid in sorted_egids:
                    min_dis = dist_list[min_dis_egid]
                    if min_dis <= rlx_dis:
                        # try relax
                        pre_eg = copy.deepcopy(path_e_[min_dis_egid])
                        eg_1 = [pre_eg["edge"][0], vid]
                        eg_2 = [vid, pre_eg["edge"][1]]
                        if eg_1 in G_es and eg_2 in G_es:
                            new_eg_1 = {"edge": eg_1, "edge_id": G_es.index(eg_1)}
                            new_eg_2 = {"edge": eg_2, "edge_id": G_es.index(eg_2)}
                            path_v_.insert(min_dis_egid+1, vid)
                            path_e_.insert(min_dis_egid, new_eg_1)
                            path_e_.insert(min_dis_egid+1, new_eg_2)
                            del path_e_[min_dis_egid+2]
                            rlx_cnt += 1
                            break
            if rlx_cnt==0:
                break
        return path_v_, path_e_
