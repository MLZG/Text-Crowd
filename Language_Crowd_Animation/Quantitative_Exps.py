import sys
import os, time, math, random, copy

import cv2
import pyglet
import numpy as np
import matplotlib.pyplot as plt
from fastgrab import screenshot
from dataclasses import dataclass

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler

from Generation_Pipelines import SgDistr_Generation_Pipeline
from Generation_Pipelines import Field_Generation_Pipeline
from Utils import *

from Simulators.Field_Env import Field_Env
from Simulators.Field_Generators.Base_Field import Base_Field
from Simulators.ORCA_Env import ORCA_Env, AGNET_PARAM_DEFAULT, AGENT_SETTING_DEFAULT

from Scenario_Generator import Scenario_Generator

from dtaidistance import dtw, dtw_ndim


@dataclass
class Quantitative_Exp_Config:
    # dataset
    data_path = "./Dataset/Data_Full_V2/"
    obj_nums = [0, 1, 2, 3, 4, 5]
    group_nums = [1, 2, 3]
    data_scale = 1.0

    # 2d sim
    visual = True
    warm_up_steps = 20
    agent_radius = 5
    agent_prefV = 4
    max_path_len_ratio = 1.5
    pos_rand_range = 5
    reach_dis = 60

    # metrics
    path_bound = 80
    strict_agent_success_ratio = 0.7
    strict_group_success_ratio = 0.8
    relaxed_agent_success_ratio = 0.5
    relaxed_group_success_ratio = 0.7

    use_complete_text = True

# ------------------------ Configs for Model V2 ------------------------ #
@dataclass
class Config_SgDistr_V2:
    data_path = "./Dataset/Data_Full_V2/"
    obj_nums = [5, 4, 3, 2, 1, 0]
    group_nums = [1, 2, 3]
    data_scale = 1.0

    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    revision = None
    output_dir = "Models_Server_ForTest/SgDistr-Full-V1/checkpoint-67000/"
    logging_dir = "logs"
    tracker_project_name = "SgDistr"
    report_to = "tensorboard"
    seed = 0

    map_size = 64
    smap_channels = 9
    sg_distr_channels = 2

    mixed_precision = "no"

    num_inference_steps=20
    guidance_scale=1

@dataclass
class Config_Field_V2:
    data_path = "./Dataset/Data_Full_V2/"
    obj_nums = [5, 4, 3, 2, 1, 0]
    group_nums = [1, 2, 3]
    data_scale = 1.0

    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    revision = None
    output_dir = "Models_Server_ForTest/Field-Full-V2/checkpoint-270000/"
    logging_dir = "logs"
    tracker_project_name = "Field"
    report_to = "tensorboard"
    seed = 0

    map_size = 64
    smap_channels = 9
    sg_distr_channels = 2
    field_channels = 2

    mixed_precision = "no"

    num_inference_steps=20
    guidance_scale=1
# ---------------------------------------------------------------------- #


def Sim2D_WithField(exp_config, scenario,
                    group_distrs, group_sizes, group_fields, group_paths,
                    show_fields=True, show_paths=True, record_video_path=None, save_sim_path=None):
    agent_n = np.sum(np.array(group_sizes))
    agent_radius = exp_config.agent_radius
    agent_prefV = exp_config.agent_prefV
    group_n = len(group_sizes)
    groups_colors = [[255, 128, 0], [135, 200, 240], [0, 255, 0], [124, 79, 13], [255, 192, 203], [128, 0, 128]][0:group_n]

    grid_width = scenario["wind_size"][0] / len(group_fields[0])
    grid_size = [len(group_fields[0]), len(group_fields[0][0])]
    base_field = Base_Field()
    base_field.reset(scenario, grid_width)
    # visualize the fields
    if exp_config.visual and show_fields:
        for grp_id in range(group_n):
            base_field.field_visualization(field=group_fields[grp_id], guidance=None)


    def sample_poses(current_poses, safe_dis, distr, grid_width, rand_buffer, boundary, obs_list, sample_num):
        distr_1d = np.array(distr).reshape(1, -1)[0]
        pos_list = []
        for smp_id in range(sample_num):
            while(1):
                pos_loc = random.choices(list(range(len(distr_1d))), weights=distr_1d, k=1)[0]
                pos = [int(pos_loc/len(distr))*grid_width+grid_width/2 + random.uniform(-rand_buffer, rand_buffer),
                       int(pos_loc%len(distr))*grid_width+grid_width/2 + random.uniform(-rand_buffer, rand_buffer)]
                pos[0] = 0 if pos[0]<0 else pos[0]
                pos[0] = boundary[0] if pos[0]>boundary[0] else pos[0]
                pos[1] = 0 if pos[1]<0 else pos[1]
                pos[1] = boundary[1] if pos[1]>boundary[1] else pos[1]
                if len(current_poses) == 0:
                    distances = np.array([np.inf])
                else:
                    distances = np.linalg.norm(np.array(current_poses) - np.array(pos), axis=1)
                if len(pos_list) == 0:
                    distances_ = np.array([np.inf])
                else:
                    distances_ = np.linalg.norm(np.array(pos_list)-np.array(pos), axis=1)
                if np.min(distances) >= safe_dis and np.min(distances_) >= safe_dis and collision_checker(pos=pos, obs_list=obs_list, buf=safe_dis):
                    pos_list.append(pos)
                    break
        return pos_list


    # set group fields
    gfields_for_ctrl = []
    tp = 0
    for gid in range(group_n):
        gfields_for_ctrl.append({"agent_ids": list(range(tp, tp+group_sizes[gid])), "field": copy.deepcopy(group_fields[gid]), "grid": copy.deepcopy(base_field.grid)})
        tp += group_sizes[gid]

    # set all agents' params and reset the scenario
    fld_env = Field_Env(agent_num=agent_n, visual=exp_config.visual, draw_scale=1.0)
    # Add boundary to scenario_bound
    scenario_bound = copy.deepcopy(scenario)
    if "obs_list" not in scenario_bound.keys():
        scenario_bound["obs_list"] = []
    wind_size = copy.deepcopy(scenario_bound["wind_size"])
    thick = 50
    scenario_bound["obs_list"].append({"type": "rectangle", "params": {"vertexes": get_box_ll(x=wind_size[0]+thick*2, y=thick, lowerleft=(-thick,-thick))}, "attributes": {}})
    scenario_bound["obs_list"].append({"type": "rectangle", "params": {"vertexes": get_box_ll(x=wind_size[0]+thick*2, y=thick, lowerleft=(-thick,wind_size[1]))}, "attributes": {}})
    scenario_bound["obs_list"].append({"type": "rectangle", "params": {"vertexes": get_box_ll(x=thick, y=wind_size[1]+thick*2, lowerleft=(-thick,-thick))}, "attributes": {}})
    scenario_bound["obs_list"].append({"type": "rectangle", "params": {"vertexes": get_box_ll(x=thick, y=wind_size[1]+thick*2, lowerleft=(wind_size[0],-thick))}, "attributes": {}})
    all_obs = fld_env.get_all_obstacles_from_scenario(copy.deepcopy(scenario_bound))

    agent_params = {"init_agent_params": []}
    for gid in range(group_n):
        add_agent_n = group_sizes[gid]
        agent_pos_list = sample_poses([], safe_dis=0, distr=group_distrs[gid][:, :, 0],
                                      grid_width=grid_width, rand_buffer=exp_config.pos_rand_range, boundary=scenario["wind_size"], obs_list=copy.deepcopy(all_obs),
                                      sample_num=add_agent_n)
        agent_goal_list = sample_poses([], safe_dis=0, distr=group_distrs[gid][:, :, 1],
                                       grid_width=grid_width, rand_buffer=exp_config.pos_rand_range, boundary=scenario["wind_size"], obs_list=copy.deepcopy(all_obs),
                                       sample_num=add_agent_n)
        for idx in range(add_agent_n):
            ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
            ai_params["pos"] = agent_pos_list[idx]
            ai_params["goal_pos"] = agent_goal_list[idx]
            ai_params["radius"] = agent_radius
            ai_params["pref_speed"] = agent_prefV
            ai_params["color"] = groups_colors[gid]
            agent_params["init_agent_params"].append(copy.deepcopy(ai_params))


    ### Start simulation
    fld_env.reset(scenario=copy.deepcopy(scenario_bound), agent_setting=copy.deepcopy(agent_params))
    max_steps = -1
    for gid, (group_path_v_i, group_path_e_i) in enumerate(group_paths):
        path_len_group_i = 0
        for edge_i in group_path_e_i:
            line_p1 = copy.deepcopy(scenario["roadmap"]["vertexes"][edge_i["edge"][0]])
            line_p2 = copy.deepcopy(scenario["roadmap"]["vertexes"][edge_i["edge"][1]])
            path_len_group_i += np.linalg.norm(np.array(line_p2)-np.array(line_p1))
            if exp_config.visual and show_paths:
                fld_env.viewer.add_line(p1=line_p1, p2=line_p2, color=groups_colors[gid])
        max_steps = max(max_steps, int(path_len_group_i * exp_config.max_path_len_ratio / exp_config.agent_prefV))

    # # wait at beginning
    # if exp_config.visual:
    #     while(1):
    #         fld_env.render()
    #         if fld_env.viewer.entered:
    #             break

    # warm up
    for stp_id in range(exp_config.warm_up_steps):
        if exp_config.visual:
            fld_env.render()
            # time.sleep(0.003)
        super(Field_Env, fld_env).perform_action(np.zeros((agent_n, 2)).tolist())
    for agt_id in range(agent_n):
        fld_env.agent_current_infor[agt_id]["traj_history"] = fld_env.agent_current_infor[agt_id]["traj_history"][:-exp_config.warm_up_steps]
        assert len(fld_env.agent_current_infor[agt_id]["traj_history"])==0

    video_frms = []
    step = 0
    all_agent_trajs = [[] for i_ in range(group_n)]
    removed_agent_n = 0
    while(1):
        if exp_config.visual:
            fld_env.render()
            time.sleep(0.006)
            if record_video_path is not None and not fld_env.viewer.closed:
                loc = fld_env.viewer.get_location()
                im_cv2 = np.array(screenshot.Screenshot().capture((loc[0],loc[1],fld_env.viewer.width,fld_env.viewer.height)))[:, :, 0:3]
                video_frms.append(im_cv2)

        # remove agents that reach the goal or get out of the scenario, and record its trajectory
        for gid in range(group_n):
            for aid in gfields_for_ctrl[gid]["agent_ids"]:
                if np.linalg.norm(np.array(fld_env.agent_current_infor[aid]["pos"])-np.array(fld_env.agent_current_infor[aid]["goal_pos"]))<exp_config.reach_dis or \
                        fld_env.agent_current_infor[aid]["pos"][0]<=0 or fld_env.agent_current_infor[aid]["pos"][0]>=scenario["wind_size"][0] or \
                        fld_env.agent_current_infor[aid]["pos"][1]<=0 or fld_env.agent_current_infor[aid]["pos"][1]>=scenario["wind_size"][1]:
                    all_agent_trajs[gid].append({"agent_id": aid, "agent_trajs": np.array(fld_env.agent_current_infor[aid]["traj_history"])[:, 0, :]})
                    fld_env.set_agent_position(aid, [-1e5, -1e5])
                    gfields_for_ctrl[gid]["agent_ids"].remove(aid)
                    removed_agent_n += 1

        fld_env.perform_action_fast(gfields_for_ctrl)

        step += 1
        if removed_agent_n>=agent_n or step>=max_steps:
            break

    if exp_config.visual and not fld_env.viewer.closed:
        fld_env.viewer.close()

    # handle the rest agents
    for gid in range(group_n):
        for aid in gfields_for_ctrl[gid]["agent_ids"]:
            all_agent_trajs[gid].append({"agent_id": aid, "agent_trajs": np.array(fld_env.agent_current_infor[aid]["traj_history"])[:, 0, :]})

    # save record video
    if exp_config.visual and record_video_path is not None:
        video_wt = cv2.VideoWriter(record_video_path, cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), 280/agent_prefV, (fld_env.viewer.width, fld_env.viewer.height))
        for frm in video_frms:
            video_wt.write(frm)
        video_wt.release()
    # save simulation data
    if save_sim_path is not None:
        sim_data = {
            "scenario": scenario,
            "agent_num": agent_n,
            "agent_infor": fld_env.agent_current_infor
        }
        np.save(save_sim_path, sim_data)

    return all_agent_trajs



def main(config_exp, config_sgdistr, config_field):
    # Load Model
    print("Loading models for start-goal-distribution generation...")
    sg_text_encoder = CLIPTextModel.from_pretrained(config_sgdistr.pretrained_model_name_or_path,
                                                    subfolder="text_encoder",
                                                    revision=config_sgdistr.revision,
                                                    use_safetensors=True)
    sg_tokenizer = CLIPTokenizer.from_pretrained(config_sgdistr.pretrained_model_name_or_path,
                                                 subfolder="tokenizer",
                                                 revision=config_sgdistr.revision)
    sg_noise_scheduler = DDPMScheduler.from_pretrained(config_sgdistr.pretrained_model_name_or_path,
                                                       subfolder="scheduler")
    sg_unet = UNet2DConditionModel.from_pretrained(config_sgdistr.output_dir,
                                                   subfolder="unet",
                                                   use_safetensors=True)
    sg_unet.set_attention_slice("max")
    sgdistr_pipeline = SgDistr_Generation_Pipeline(text_encoder=sg_text_encoder, tokenizer=sg_tokenizer,
                                                   scheduler=sg_noise_scheduler, unet=sg_unet,
                                                   config=config_sgdistr)

    print("Loading models for field generation...")
    field_text_encoder = CLIPTextModel.from_pretrained(config_field.pretrained_model_name_or_path,
                                                       subfolder="text_encoder",
                                                       revision=config_field.revision,
                                                       use_safetensors=True)
    field_tokenizer = CLIPTokenizer.from_pretrained(config_field.pretrained_model_name_or_path,
                                                    subfolder="tokenizer",
                                                    revision=config_field.revision)
    field_noise_scheduler = DDPMScheduler.from_pretrained(config_field.pretrained_model_name_or_path,
                                                          subfolder="scheduler")
    field_unet = UNet2DConditionModel.from_pretrained(config_field.output_dir,
                                                      subfolder="unet",
                                                      use_safetensors=True)
    field_unet.set_attention_slice("max")
    field_pipeline = Field_Generation_Pipeline(text_encoder=field_text_encoder, tokenizer=field_tokenizer,
                                               scheduler=field_noise_scheduler, unet=field_unet,
                                               config=config_field)

    # Run Quantitative Experiments for each data piece
    for obj_n in config_exp.obj_nums:
        for group_n in config_exp.group_nums:
            experiment_result_og = {"strict_success_group_num": 0, "relaxed_success_group_num": 0, "total_group_num": 0,
                                    "average_dtw": 0, "exp_data": []}

            # read data piece
            print(f"# ------ Reading data of object_num {obj_n}, group_num {group_n}...")
            dt_path_og = config_exp.data_path + f"Obj{obj_n}_Group{group_n}/dt_testing.npy"
            print(f"   data path: {dt_path_og}")
            dt_og = np.load(dt_path_og, allow_pickle=True).item()

            # choose sub-dataset according to data_scale if data_scale is not equal to 1
            assert abs(int(len(dt_og["data"])*config_exp.data_scale) - len(dt_og["data"])*config_exp.data_scale) < 1e-6
            data_size_foruse = int(len(dt_og["data"])*config_exp.data_scale)
            print("   total cases in data: %d | cases for use: %d" %(len(dt_og["data"]), data_size_foruse))
            dt_og["data"] = dt_og["data"][0:data_size_foruse]

            # iterate over each data case
            for data_id, dt_i in enumerate(dt_og["data"]):
                print(f"case id: {data_id}")
                # ground truth
                gt_scenario = dt_i["scenario"]
                gt_group_descriptors = dt_i["group_descriptors"]
                gt_group_descriptions = dt_i["group_descriptions"]
                gt_smap = dt_i["semantic_map"]
                gt_group_sizes = dt_i["group_sizes"]
                gt_group_paths = dt_i["group_paths"]

                gt_group_sgdistrs = dt_i["group_sg_distrs"]
                gt_group_fields = dt_i["group_fields"]

                # prepare inputs for this case
                # input text
                input_prompts = copy.deepcopy(gt_group_descriptions)
                if config_exp.use_complete_text:
                    from Behavior_Descriptor import Behavior_Descriptor
                    drop_out_ps = {"action_loc": 0, "obj_loc": 0, "action_dir": 0}
                    BD_ = Behavior_Descriptor(snr_mid_scl=0.25, snr_inner_scl=0.66, global_adj_num=8,
                                              local_adj_num=8, dir_adj_num=8, drop_out_ps=copy.deepcopy(drop_out_ps))
                    input_prompts = []
                    for descriptor_i in gt_group_descriptors:
                        input_prompts.append(BD_.descriptor_to_text(dsc_l=copy.deepcopy(descriptor_i), dropout_ps=copy.deepcopy(drop_out_ps)))

                # input semantic map
                input_smaps = []
                for group_id in range(group_n):
                    input_smaps.append(copy.deepcopy(gt_smap))

                # do inference
                print("Inferring start and goal distributions...")
                pred_group_sgdistrs = sgdistr_pipeline.inference(smaps=copy.deepcopy(np.array(input_smaps)),
                                                                 prompts=copy.deepcopy(input_prompts),
                                                                 num_inference_steps=config_sgdistr.num_inference_steps,
                                                                 guidance_scale=config_sgdistr.guidance_scale,
                                                                 save_path=None, show=False)
                pred_group_sgdistrs = np.clip(pred_group_sgdistrs, a_min=0., a_max=1.0)
                pred_group_sgdistrs[pred_group_sgdistrs<0.2] = 0.
                pred_group_sgdistrs[pred_group_sgdistrs>0.8] = 1.

                print("Inferring fields...")
                pred_group_fields = field_pipeline.inference(smaps=copy.deepcopy(np.array(input_smaps)),
                                                             prompts=copy.deepcopy(input_prompts),
                                                             sg_distrs=copy.deepcopy(pred_group_sgdistrs),
                                                             num_inference_steps=config_field.num_inference_steps,
                                                             guidance_scale=config_field.guidance_scale,
                                                             save_path=None, show=False)
                # normalize the fields and remove vectors in obstacle
                grid_size = [config_field.map_size, config_field.map_size]
                grid_width = gt_scenario["wind_size"][0] / grid_size[0]
                base_field = Base_Field()
                base_field.reset(scenario=copy.deepcopy(gt_scenario), grid_width=grid_width)
                grid_map = base_field.grid["grid_map"]
                obs_coords = np.argwhere(grid_map==1)
                for group_id in range(group_n):
                    pred_group_fields[group_id] = np.nan_to_num(pred_group_fields[group_id] / (np.linalg.norm(pred_group_fields[group_id], axis=2).reshape((grid_size[0], grid_size[1], 1))), 0)
                    pred_group_fields[group_id][obs_coords[:, 0], obs_coords[:, 1]] = np.array([0., 0.])

                # do simulation
                agent_trajs = Sim2D_WithField(exp_config=copy.deepcopy(config_exp), scenario=copy.deepcopy(gt_scenario),
                                              group_distrs=pred_group_sgdistrs, group_sizes=gt_group_sizes, group_fields=pred_group_fields, group_paths=gt_group_paths,
                                              show_fields=False, show_paths=True, record_video_path=None, save_sim_path=None)

                # compute metrics
                # for each group, compute the distances between each agent (that belongs to that group) and the groundtruth path
                case_exp_data = []
                for grp_idx in range(group_n):
                    group_exp_data = {"ground_truth_path_points": None, "group_agent_infor": [],
                                      "group_success_agentN_strict": 0, "group_success_agentN_relaxed":0, "group_total_agent_N": 0,
                                      "group_success_strict": None, "group_success_relaxed": None, "group_average_dtw": 0}
                    # get groundtruth path points
                    (group_path_v_i, group_path_e_i) = gt_group_paths[grp_idx]
                    gt_path_points = []
                    for eg_id, edge_i in enumerate(group_path_e_i):
                        pt1 = copy.deepcopy(gt_scenario["roadmap"]["vertexes"][edge_i["edge"][0]])
                        pt2 = copy.deepcopy(gt_scenario["roadmap"]["vertexes"][edge_i["edge"][1]])
                        length_ = np.linalg.norm(np.array(pt2)-np.array(pt1))
                        pt_step = config_exp.agent_prefV
                        for step_ in range(int(length_/pt_step)):
                            pct = step_/int(length_/pt_step)
                            pt_ = pct*np.array(pt2)+(1-pct)*np.array(pt1)
                            gt_path_points.append([pt_[0], pt_[1]])
                        if eg_id == len(group_path_e_i) - 1:
                            gt_path_points.append([pt2[0], pt2[1]])
                    group_exp_data["ground_truth_path_points"] = copy.deepcopy(gt_path_points)

                    # get agent_trajs which belongs to this group
                    group_agent_trajs = agent_trajs[grp_idx]
                    for agent_traj_i in group_agent_trajs:
                        agent_exp_infor = {"traj": copy.deepcopy(agent_traj_i), "average_dtw": None, "in_path_ratio": None,
                                           "success_strict":None, "success_relaxed":None}

                        ai_traj = agent_traj_i["agent_trajs"]
                        # compute similarity between ai_traj and gt_path_points
                        dis_useless, dtw_paths = dtw_ndim.warping_paths_fast(np.array(gt_path_points, dtype=np.double),
                                                                             np.array(ai_traj, dtype=np.double))
                        best_dtw_path = np.array(dtw.best_path(dtw_paths))
                        dis_list = np.linalg.norm(np.array(gt_path_points)[best_dtw_path[:, 0]] - np.array(ai_traj)[best_dtw_path[:, 1]], axis=1)
                        dis_list -= config_exp.path_bound
                        dis_list[dis_list<0] = 0
                        agent_exp_infor["average_dtw"] = np.sum(dis_list)/len(dis_list)
                        group_exp_data["group_total_agent_N"] += 1
                        group_exp_data["group_average_dtw"] += agent_exp_infor["average_dtw"]

                        out_path_points = np.argwhere(dis_list>0).reshape(1, -1)[0]
                        in_path_points = np.argwhere(dis_list<=0).reshape(1, -1)[0]
                        assert len(out_path_points)+len(in_path_points)==len(dis_list)
                        in_ratio = len(in_path_points) / len(dis_list)
                        agent_exp_infor["in_path_ratio"] = in_ratio


                        if in_ratio >= config_exp.strict_agent_success_ratio:
                            agent_exp_infor["success_strict"] = True
                            group_exp_data["group_success_agentN_strict"] += 1
                        else:
                            agent_exp_infor["success_strict"] = False

                        if in_ratio >= config_exp.relaxed_agent_success_ratio:
                            agent_exp_infor["success_relaxed"] = True
                            group_exp_data["group_success_agentN_relaxed"] += 1
                        else:
                            agent_exp_infor["success_relaxed"] = False

                        group_exp_data["group_agent_infor"].append(agent_exp_infor)


                    group_exp_data["group_average_dtw"] /= group_exp_data["group_total_agent_N"]
                    experiment_result_og["average_dtw"] += group_exp_data["group_average_dtw"]
                    experiment_result_og["total_group_num"] += 1

                    if group_exp_data["group_success_agentN_strict"] / group_exp_data["group_total_agent_N"] >= config_exp.strict_group_success_ratio:
                        group_exp_data["group_success_strict"] = True
                        experiment_result_og["strict_success_group_num"] += 1
                    else:
                        group_exp_data["group_success_strict"] = False

                    if group_exp_data["group_success_agentN_relaxed"] / group_exp_data["group_total_agent_N"] >= config_exp.relaxed_group_success_ratio:
                        group_exp_data["group_success_relaxed"] = True
                        experiment_result_og["relaxed_success_group_num"] += 1
                    else:
                        group_exp_data["group_success_relaxed"] = False

                    case_exp_data.append(group_exp_data)

                    print(f"   group {grp_idx} | success (strict): {group_exp_data['group_success_strict']} |"
                          f" success (relaxed): {group_exp_data['group_success_relaxed']} | group dtw: {group_exp_data['group_average_dtw']}")
                    print(f"   - OVERALL -   | success group num (strict): {experiment_result_og['strict_success_group_num']} | "
                          f"success group num (relaxed): {experiment_result_og['relaxed_success_group_num']} | total group num: {experiment_result_og['total_group_num']}")
                    print(f"   - OVERALL -   | success rate (strict): {experiment_result_og['strict_success_group_num']/experiment_result_og['total_group_num']} | "
                          f"success rate (relaxed): {experiment_result_og['relaxed_success_group_num']/experiment_result_og['total_group_num']} | "
                          f"average_dtw: {experiment_result_og['average_dtw']/experiment_result_og['total_group_num']}")
                    print()

                experiment_result_og["exp_data"].append(case_exp_data)

            experiment_result_og["average_dtw"] /= experiment_result_og["total_group_num"]

            # save the experiment data for this object num and group num
            np.save(f"./Quantitative_Exp_Data/exp_obj{obj_n}_group{group_n}.npy", {"all_exp_data": experiment_result_og})


def Quantitative_Exp_Result_from_Data(exp_config, data_path):
    for obj_n in exp_config.obj_nums:
        for group_n in exp_config.group_nums:
            print(f"# ------ Result for object {obj_n}, group {group_n} ------ #")
            dt_og_path = os.path.join(data_path, f"exp_obj{obj_n}_group{group_n}.npy")
            print(f"data path: {dt_og_path}")
            result_og = np.load(dt_og_path, allow_pickle=True).item()["all_exp_data"]
            print(f"   - OVERALL -   | success group num (strict): {result_og['strict_success_group_num']} | "
                  f"success group num (relaxed): {result_og['relaxed_success_group_num']} | total group num: {result_og['total_group_num']}")
            print(f"   - OVERALL -   | success rate (strict): {result_og['strict_success_group_num']/result_og['total_group_num']} | "
                  f"success rate (relaxed): {result_og['relaxed_success_group_num']/result_og['total_group_num']} | "
                  f"average_dtw: {result_og['average_dtw']}")


if __name__ == '__main__':
    main(config_exp=Quantitative_Exp_Config(), config_sgdistr=Config_SgDistr_V2(), config_field=Config_Field_V2())
    # Quantitative_Exp_Result_from_Data(exp_config=Quantitative_Exp_Config(), data_path="./Quantitative_Exp_Data/result_full/")