import copy
import signal
from Language_Crowd_Animation.Behavior_Descriptor import Behavior_Descriptor
from Language_Crowd_Animation.Roadmap import Roadmap
from Language_Crowd_Animation.Scenario_Generator import Scenario_Generator
from Simulators.ORCA_Env import ORCA_Env
from Simulators.Field_Generators.CurveTracking_Field import CurveTracking_Field
from Simulators.Field_Generators.Navigation_Field import Navigation_Field
from Simulators.Field_Env import Field_Env
from Utils import *
import numpy as np
import os
import time
import argparse
import random
import matplotlib.pyplot as plt

parameters_simple = {
    # scenario generation params
    "obj_nums": [0, 1],
    "group_nums": [1, 2, 3],
    "area_max_num": 3,
    "default_scenario_param": {
        "wind_size": [1024, 1024],
        "bound_srk_scale": 1/5,
        "safe_dis" : 100,
        "objects": {
            "rectangle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "triangle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "circle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "zebra_crossing": {"num": 0, "size_range":[170, 200], "graph_buffer": 30},
            "passage": {"num": 0, "size_range":[170, 200], "graph_buffer": 30},
            "entrance": {"num": 0, "size_range":[80, 80], "graph_buffer": None},
            "exit": {"num": 0, "size_range":[80, 80], "graph_buffer": None}
        }
    },
    # roadmap params
    "roadmap_sample_n": 0,
    "roadmap_nb_dis": 2048,
    "path_constrains_default": {"start_p": None, "goal_p": None, "line_angle_lim": 89,
                                "adaptive_w": {"v_w": 0.95, "e_w": 0.7}, "safe_dis": 100, "path_len_range": [1, 9]},
    "path_relax_dis": 30,

    # Behavior params
    "mid_scale": 0.25,
    "inner_scale": 0.66,
    "global_adj_num": 8,
    "local_adj_num": 8,
    "dir_adj_num": 8,
    "drop_out_ps": {"action_loc": 0.6, "obj_loc": 0.1, "action_dir": 0.4},
    "group_size_range": {"tiny":[2, 3], "small":[4, 7], "big":[8, 15], "large":[16, 30]},

    # field params
    "field_use": "CTF", # CurveTracking Field (CTF)
    # (Using CurveTracking Field)
    "reverse_direction": False,
    "vr": 1,
    "kf": 0.05, # 0.008 | 0.015 | 0.05    # convergence rate
    "flag_follow_obstacle": True,
    "epsilon": 0.,  # 0.
    "switch_dist_0": 40.,  # 60.
    "switch_dist": 40.,  # 60.
    "lidar_N": 256,
    "default_constrains_CTF": {
        "filter_path_n_average": 0,
        "closed_path_flag": False,
        # distance between two discrete points in path
        "pt_step_len": 15,  # 15-30
        # smooth condition for trajectory smoothing
        "smooth_condition": 600,
    },
    "default_guidance_CTF": {
        "type": "lines",
        "params": {
            "lines": None,
            "width": 100,  # useless
            "decay_rate": 0.95, # useless
        }
    },

    # env params
    "grid_width_map": 16,
    "grid_width_field": 16,
}

parameters_hard = {
    # scenario generation params
    "obj_nums": [2, 3, 4, 5],
    "group_nums": [1, 2, 3],
    "area_max_num": 3,
    "default_scenario_param": {
        "wind_size": [1024, 1024],
        "bound_srk_scale": 1/5,
        "safe_dis" : 100,
        "objects": {
            "rectangle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "triangle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "circle": {"num": 0, "size_range":[70, 100], "graph_buffer": 30},
            "zebra_crossing": {"num": 0, "size_range":[170, 200], "graph_buffer": 30},
            "passage": {"num": 0, "size_range":[170, 200], "graph_buffer": 30},
            "entrance": {"num": 0, "size_range":[80, 80], "graph_buffer": None},
            "exit": {"num": 0, "size_range":[80, 80], "graph_buffer": None}
        }
    },
    # roadmap params
    "roadmap_sample_n": 0,
    "roadmap_nb_dis": 2048,
    "path_constrains_default": {"start_p": None, "goal_p": None, "line_angle_lim": 89,
                                "adaptive_w": {"v_w": 0.95, "e_w": 0.7}, "safe_dis": 100, "path_len_range": [1, 9]},
    "path_relax_dis": 30,

    # Behavior params
    "mid_scale": 0.25,
    "inner_scale": 0.66,
    "global_adj_num": 8,
    "local_adj_num": 8,
    "dir_adj_num": 8,
    "drop_out_ps": {"action_loc": 0.6, "obj_loc": 0.1, "action_dir": 0.4},
    "group_size_range": {"tiny":[2, 3], "small":[4, 7], "big":[8, 15], "large":[16, 30]},

    # field params
    "field_use": "CTF", # CurveTracking Field (CTF)
    # (Using CurveTracking Field)
    "reverse_direction": False,
    "vr": 1,
    "kf": 0.05, # 0.008 | 0.015 | 0.05    # convergence rate
    "flag_follow_obstacle": True,
    "epsilon": 0.,  # 0.
    "switch_dist_0": 40.,  # 60.
    "switch_dist": 40.,  # 60.
    "lidar_N": 256,
    "default_constrains_CTF": {
        "filter_path_n_average": 0,
        "closed_path_flag": False,
        # distance between two discrete points in path
        "pt_step_len": 15,
        # smooth condition for trajectory smoothing
        "smooth_condition": 600,
    },
    "default_guidance_CTF": {
        "type": "lines",
        "params": {
            "lines": None,
            "width": 100,  # useless
            "decay_rate": 0.95, # useless
        }
    },

    # env params
    "grid_width_map": 16,
    "grid_width_field": 16,
}


def dataset_generation(args):
    param_ = copy.deepcopy(parameters_simple) if args.mode == "simple" else copy.deepcopy(parameters_hard)
    SG_ = Scenario_Generator()
    RM_ = Roadmap()
    BD_ = Behavior_Descriptor(snr_mid_scl=param_["mid_scale"], snr_inner_scl=param_["inner_scale"], global_adj_num=param_["global_adj_num"],
                              local_adj_num=param_["local_adj_num"], dir_adj_num=param_["dir_adj_num"], drop_out_ps=copy.deepcopy(param_["drop_out_ps"]))
    assert param_["field_use"] == "CTF"
    FG_ = CurveTracking_Field(reverse_direction=param_["reverse_direction"], vr=param_["vr"], kf=param_["kf"],
                              flag_follow_obstacle=param_["flag_follow_obstacle"], epsilon=param_["epsilon"],
                              switch_dist_0=param_["switch_dist_0"], switch_dist=param_["switch_dist"], lidar_N=param_["lidar_N"])

    for obj_n in param_["obj_nums"]:
        for group_n in param_["group_nums"]:
            print("### Generating data for obj_num =", obj_n, "and group_num =", group_n)
            case_id = 0
            dataset_obj_group = {"param": copy.deepcopy(param_), "obj_num": obj_n, "group_num": group_n, "data": []}
            while(case_id<args.dataset_size_per_case):
                start_t = time.time()
                print("case id:", case_id)

                # get a random scenario parameter and generate a scenario with roadmap
                while(1):
                    snr_param = copy.deepcopy(param_["default_scenario_param"])
                    for obj_id in range(obj_n):
                        obj_name = random.choice(["rectangle","triangle","circle","zebra_crossing","passage"])
                        snr_param["objects"][obj_name]["num"] += 1
                    snr_param["objects"]["entrance"]["num"] = random.randint(group_n, param_["area_max_num"])
                    snr_param["objects"]["exit"]["num"] = random.randint(1, param_["area_max_num"])
                    # generate the scenario and roadmap
                    snr_init = SG_.random_scenario(copy.deepcopy(snr_param))
                    if snr_init is not None:
                        break
                    else:
                        print("Time out in generating scenario. Retrying...")
                scenario_, roadmap_ =  RM_.build_roadmap_PRM(scenario_=copy.deepcopy(snr_init),
                                                             sample_num=param_["roadmap_sample_n"], nb_dis=param_["roadmap_nb_dis"])

                groups_paths = []
                groups_descriptors = []
                groups_descriptions = []
                groups_sizes = []
                valid_path = True
                for group_id in range(group_n):
                    # get the entrance_i and exit_i for group_i
                    etc_id = copy.deepcopy(group_id)
                    exit_id = snr_param["objects"]["entrance"]["num"]+random.randint(0, snr_param["objects"]["exit"]["num"]-1)
                    start_p = copy.deepcopy(scenario_["areas_list"][etc_id]["attributes"]["graph"]["points_idx_inRM"][0])
                    goal_p = copy.deepcopy(scenario_["areas_list"][exit_id]["attributes"]["graph"]["points_idx_inRM"][0])
                    # set the path constrains
                    constrains_ = copy.deepcopy(param_["path_constrains_default"])
                    constrains_["start_p"] = copy.deepcopy(start_p)
                    constrains_["goal_p"] = copy.deepcopy(goal_p)
                    # sample a path randomly and postprocess it
                    path_vs_init, path_es_init = RM_.sample_path(roadmap=copy.deepcopy(scenario_["roadmap"]), constrains=copy.deepcopy(constrains_))
                    if path_vs_init is None or path_es_init is None:
                        valid_path = False
                        break
                    path_vs, path_es = RM_.path_postprocess(scenario_=copy.deepcopy(scenario_), path_v=copy.deepcopy(path_vs_init),
                                                            path_e=copy.deepcopy(path_es_init), rlx_dis=param_["path_relax_dis"])
                    groups_paths.append([copy.deepcopy(path_vs), copy.deepcopy(path_es)])

                    # get the language description of the path
                    behavior_ = BD_.path_to_descriptor(copy.deepcopy(scenario_), copy.deepcopy(path_vs), copy.deepcopy(path_es))
                    groups_descriptors.append(copy.deepcopy(behavior_))

                    g_size_rg = param_["group_size_range"][behavior_[0]["group_size"]]
                    g_size = random.randint(g_size_rg[0], g_size_rg[1])
                    groups_sizes.append(g_size)

                    description_ = BD_.descriptor_to_text(dsc_l=copy.deepcopy(behavior_), dropout_ps=copy.deepcopy(param_["drop_out_ps"]))
                    groups_descriptions.append(copy.deepcopy(description_))

                if not valid_path:
                    print("No valid path, re-generate the scenario.")
                    continue

                ### get the input sem_map
                smap_ = SG_.get_semantic_map(scenario_in=copy.deepcopy(scenario_), grid_width=param_["grid_width_map"])

                ### get all start_goal_distributions
                groups_distrs = []
                for g_id in range(len(groups_paths)):
                    s_vid_ = groups_paths[g_id][0][0]
                    g_vid_ = groups_paths[g_id][0][-1]
                    sg_areas = {"start_area": scenario_["roadmap"]["vertexes_idx_inS"][s_vid_]["obj_id"],
                                "goal_area": scenario_["roadmap"]["vertexes_idx_inS"][g_vid_]["obj_id"]}
                    sg_distr = SG_.get_sg_distb(scenario_in=copy.deepcopy(scenario_), grid_width=param_["grid_width_map"],
                                                sg_areas=copy.deepcopy(sg_areas))
                    groups_distrs.append(copy.deepcopy(sg_distr))

                ### get the fields ###
                group_fields = []
                FG_.reset(scenario=copy.deepcopy(scenario_), grid_width=param_["grid_width_field"])
                for g_id in range(len(groups_paths)):
                    lines = []
                    for e_i in groups_paths[g_id][1]:
                        lines.append([copy.deepcopy(scenario_["roadmap"]["vertexes"][e_i["edge"][0]]),
                                      copy.deepcopy(scenario_["roadmap"]["vertexes"][e_i["edge"][1]])])
                    guidance_ = copy.deepcopy(param_["default_guidance_CTF"])
                    guidance_["params"]["lines"] = copy.deepcopy(lines)
                    field_constrains = copy.deepcopy(param_["default_constrains_CTF"])
                    field_ = FG_.get_field(guidance=copy.deepcopy(guidance_), constrains=copy.deepcopy(field_constrains))
                    group_fields.append(copy.deepcopy(field_))

                dt_i = {"scenario": scenario_, "obj_num": obj_n, "group_num": group_n,
                        "group_descriptors": groups_descriptors, "group_descriptions": groups_descriptions, "semantic_map": smap_,
                        "group_sizes": groups_sizes, "group_sg_distrs": groups_distrs, "group_paths": groups_paths, "group_fields": group_fields}
                dataset_obj_group["data"].append(copy.deepcopy(dt_i))

                case_id += 1

            file_path = args.data_path + "Data_Full_V1/" + "Obj" + str(obj_n) + "_" + "Group" + str(group_n) + "/"
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            fst_p = int(len(dataset_obj_group["data"])*args.dataset_proportion[0])
            scd_p = int(len(dataset_obj_group["data"])*(args.dataset_proportion[0]+args.dataset_proportion[1]))
            dt_training = {}
            dt_validation = {}
            dt_testing = {}
            for key_i in dataset_obj_group.keys():
                if key_i == "data":
                    dt_training[key_i] = copy.deepcopy(dataset_obj_group[key_i][0:fst_p])
                    dt_validation[key_i] = copy.deepcopy(dataset_obj_group[key_i][fst_p:scd_p])
                    dt_testing[key_i] = copy.deepcopy(dataset_obj_group[key_i][scd_p:])
                else:
                    dt_training[key_i] = copy.deepcopy(dataset_obj_group[key_i])
                    dt_validation[key_i] = copy.deepcopy(dataset_obj_group[key_i])
                    dt_testing[key_i] = copy.deepcopy(dataset_obj_group[key_i])
            np.save(file_path+"dt_training.npy", dt_training)
            np.save(file_path+"dt_validation.npy", dt_validation)
            np.save(file_path+"dt_testing.npy", dt_testing)


def field_control_visualization(scenario, group_distrs, group_paths, group_sizes, group_fields, show_path=True, show_fields=True):
    from Simulators.ORCA_Env import AGNET_PARAM_DEFAULT
    from Simulators.Field_Generators.Base_Field import Base_Field

    agent_n = 200
    agent_radius = 5
    agent_prefV = 2
    group_n = len(group_sizes)
    groups_colors = [[255, 128, 0], [135, 200, 240], [0, 255, 0]][0:group_n]
    gfields_for_ctrl = []
    g_guidances = []
    grid_width = scenario["wind_size"][0] / len(group_fields[0])
    grid_size = [len(group_fields[0]), len(group_fields[0][0])]
    base_field = Base_Field()
    base_field.reset(scenario, grid_width)

    agent_params = {"init_agent_params": []}
    aid = 0
    for gid in range(group_n):
        gf_i = {"agent_ids": [], "field": None, "grid": None}
        gi_st_distr = group_distrs[gid][:, :, 0].reshape(1, -1)[0]
        gi_g_distr = group_distrs[gid][:, :, 1].reshape(1, -1)[0]
        st_poses = random.choices(list(range(len(gi_st_distr))), weights=gi_st_distr, k=group_sizes[gid])
        g_poses = random.choices(list(range(len(gi_g_distr))), weights=gi_g_distr, k=group_sizes[gid])
        for gaid in range(group_sizes[gid]):
            ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
            ai_params["pos"] = [int(st_poses[gaid]/grid_size[0])*grid_width+grid_width/2+random.uniform(0, 1),
                                int(st_poses[gaid]%grid_size[0])*grid_width+grid_width/2+random.uniform(0, 1)]
            ai_params["goal_pos"] = [int(g_poses[gaid]/grid_size[0])*grid_width+grid_width/2+random.uniform(0, 1),
                                     int(g_poses[gaid]%grid_size[0])*grid_width+grid_width/2+random.uniform(0, 1)]
            ai_params["radius"] = agent_radius
            ai_params["pref_speed"] = agent_prefV
            ai_params["color"] = groups_colors[gid]
            agent_params["init_agent_params"].append(copy.deepcopy(ai_params))
            gf_i["agent_ids"].append(aid)
            aid += 1
        gf_i["field"] = copy.deepcopy(group_fields[gid])
        gf_i["grid"] = copy.deepcopy(base_field.grid)
        lines = []
        if show_path:
            for e_i in group_paths[gid][1]:
                lines.append([copy.deepcopy(scenario["roadmap"]["vertexes"][e_i["edge"][0]]),
                              copy.deepcopy(scenario["roadmap"]["vertexes"][e_i["edge"][1]])])
        g_guidance_i = {"type": "lines", "params":{"lines": lines}}
        if show_fields:
            base_field.field_visualization(field=gf_i["field"], guidance=g_guidance_i)
        g_guidances.append(g_guidance_i)
        gfields_for_ctrl.append(copy.deepcopy(gf_i))

    for rst_aid in range(aid, agent_n):
        ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
        ai_params["pos"] = [-1000, -1000]
        ai_params["radius"] = agent_radius
        agent_params["init_agent_params"].append(copy.deepcopy(ai_params))

    fld_env = Field_Env(agent_num=agent_n, visual=True, draw_scale=1.0)


    fld_env.reset(scenario=copy.deepcopy(scenario), agent_setting=copy.deepcopy(agent_params))
    if show_path:
        for gid, (group_path_v_i, group_path_e_i) in enumerate(group_paths):
            color_i = [[255, 0, 0], [0, 0, 255], [255, 255, 0]][gid]
            for edge_i in group_path_e_i:
                fld_env.viewer.add_line(p1=scenario["roadmap"]["vertexes"][edge_i["edge"][0]],
                                        p2=scenario["roadmap"]["vertexes"][edge_i["edge"][1]], color=color_i)
    while(1):
        fld_env.render()
        if fld_env.viewer.entered:
            break
    keyboard = pyglet.window.key.KeyStateHandler()
    while(1):
        fld_env.viewer.push_handlers(keyboard)
        if keyboard[pyglet.window.key.Q]:
            fld_env.set_viewer_field(gfields_for_ctrl[0]["grid"], gfields_for_ctrl[0]["field"])
        elif keyboard[pyglet.window.key.W]:
            fld_env.set_viewer_field(gfields_for_ctrl[1]["grid"], gfields_for_ctrl[1]["field"])
        elif keyboard[pyglet.window.key.E]:
            fld_env.set_viewer_field(gfields_for_ctrl[2]["grid"], gfields_for_ctrl[2]["field"])
        elif  keyboard[pyglet.window.key.P]:
            fld_env.set_viewer_field(None, None)
        elif keyboard[pyglet.window.key.A]:
            fld_env.set_viewer_guidance(g_guidances[0])
        elif keyboard[pyglet.window.key.S]:
            fld_env.set_viewer_guidance(g_guidances[1])
        elif keyboard[pyglet.window.key.D]:
            fld_env.set_viewer_guidance(g_guidances[2])
        elif keyboard[pyglet.window.key.L]:
            fld_env.set_viewer_guidance(None)

        fld_env.render()
        for gid in range(group_n):
            for aid in gfields_for_ctrl[gid]["agent_ids"]:
                if np.linalg.norm(np.array(fld_env.agent_current_infor[aid]["pos"])-np.array(fld_env.agent_current_infor[aid]["goal_pos"]))<0. or \
                        fld_env.agent_current_infor[aid]["pos"][0]<=0 or fld_env.agent_current_infor[aid]["pos"][0]>=scenario["wind_size"][0] or \
                        fld_env.agent_current_infor[aid]["pos"][1]<=0 or fld_env.agent_current_infor[aid]["pos"][1]>=scenario["wind_size"][1]:
                    ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
                    ai_params["pos"] = [-1000, -1000]
                    fld_env.set_agent_params(aid, ai_params)
                    gfields_for_ctrl[gid]["agent_ids"].remove(aid)
        fld_env.perform_action(gfields_for_ctrl)

        if fld_env.viewer.closed:
            break
        time.sleep(0.001)



def dataset_visual(args, data_path):
    data_ = np.load(data_path, allow_pickle=True).item()

    orca_env = ORCA_Env(agent_num=100, draw_scale=1.0)
    for key_i in data_.keys():
        print(key_i, len(data_[key_i]))

    for dt_id in range(len(data_["texts"])):
        scenario_ = copy.deepcopy(data_["scenarios"][dt_id])
        paths_ = copy.deepcopy(data_["paths"][dt_id])
        smap_ = copy.deepcopy(data_["sem_maps"][dt_id])
        sg_distr = copy.deepcopy(data_["sg_distrs"][dt_id])
        text_ = copy.deepcopy(data_["texts"][dt_id])
        g_field = copy.deepcopy(data_["g_fields"][dt_id])


        print(text_)
        sg_colors = np.array([[0,255,0],[0,0,255]])
        map_colors = np.concatenate([np.random.randint(0, 255, (len(smap_[0][0])-2, 3)), sg_colors], axis=0)
        smap_cvimg = cv_visual_map(map_in=copy.deepcopy(smap_), colors=map_colors, save_nm=None)
        distr_cvimg = cv_visual_map(map_in=copy.deepcopy(sg_distr), colors=sg_colors, save_nm=None)
        field_cvimg = cv_visual_field(g_field, grid_width=10, save_nm=None)
        enlarge_scale = int(len(field_cvimg)/len(smap_cvimg))
        smap_large = cv2.resize(smap_cvimg, None, fx=enlarge_scale, fy=enlarge_scale, interpolation=cv2.INTER_CUBIC)
        distr_large = cv2.resize(distr_cvimg, None, fx=enlarge_scale, fy=enlarge_scale, interpolation=cv2.INTER_CUBIC)
        interval = np.ones((len(smap_large), 20, 3))*255
        cat_img = cv2.hconcat([interval, smap_large, interval, distr_large, interval, field_cvimg, interval])
        cv2.imwrite("./Dataset/data.jpg", cat_img)

        # visualization
        from Simulators.ORCA_Env import AGENT_SETTING_DEFAULT
        orca_env.reset(scenario=copy.deepcopy(scenario_), agent_setting={"init_agent_params": copy.deepcopy(orca_env.agent_current_infor)})
        # plot the roadmap
        for vid, v_i in enumerate(scenario_["roadmap"]["vertexes"]):
            v_color = [255, 0, 0] if scenario_["roadmap"]["vertexes_idx_inS"][vid] is not None else [0, 0, 0]
            orca_env.viewer.add_waypoint(pos=v_i, waypoint_size=10, color=v_color)
        for eid, e_i in enumerate(scenario_["roadmap"]["edges"]):
            e_color = [0, 255, 0] if scenario_["roadmap"]["edges_idx_inS"][eid] is not None else [90, 90, 90]
            orca_env.viewer.add_line(p1=scenario_["roadmap"]["vertexes"][e_i[0]], p2=scenario_["roadmap"]["vertexes"][e_i[1]], color=e_color)
        for gid, (group_path_v_i, group_path_e_i) in enumerate(paths_):
            color_i = [[255, 0, 0], [0, 0, 255], [255, 255, 0]][gid]
            for edge_i in group_path_e_i:
                orca_env.viewer.add_line(p1=scenario_["roadmap"]["vertexes"][edge_i["edge"][0]], p2=scenario_["roadmap"]["vertexes"][edge_i["edge"][1]], color=color_i)
        while(1):
            orca_env.render()
            time.sleep(0.06)
            if orca_env.viewer.closed:
                break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="hard") # "simple" or "hard"
    parser.add_argument('--data_path', default="./Dataset/")
    parser.add_argument('--dataset_size_per_case', default=4000)
    parser.add_argument('--dataset_proportion', default=[0.6, 0.2, 0.2])  #[training, validation, testing]
    args = parser.parse_args()

    dataset_generation(args)



    # ### Dataset checking and visualization
    # path_ = args.data_path + "Data_Full_V1/" + "Obj5_Group2/"
    # data_training = np.load(path_ + "dt_training.npy", allow_pickle=True).item()
    # data_validation = np.load(path_ + "dt_validation.npy", allow_pickle=True).item()
    # data_testing = np.load(path_ + "dt_testing.npy", allow_pickle=True).item()
    # print(len(data_training["data"]))
    # print(len(data_validation["data"]))
    # print(len(data_testing["data"]))
    #
    # dt_for_vis = data_training["data"][2]
    # for key_i in dt_for_vis.keys():
    #     print(key_i, end=" ")
    # print()
    # print("obj num:", dt_for_vis["obj_num"])
    # print("group num:", dt_for_vis["group_num"])
    # print("group sizes:", dt_for_vis["group_sizes"])
    # for gid in range(dt_for_vis["group_num"]):
    #     print(dt_for_vis["group_descriptors"][gid])
    #     print(dt_for_vis["group_descriptions"][gid])
    #
    # smap = dt_for_vis["semantic_map"]
    # cv_visual_map(smap, show=True)
    #
    # field_control_visualization(scenario=dt_for_vis["scenario"], group_distrs=dt_for_vis["group_sg_distrs"],
    #                             group_paths=dt_for_vis["group_paths"], group_sizes=dt_for_vis["group_sizes"], group_fields=dt_for_vis["group_fields"])
    #
