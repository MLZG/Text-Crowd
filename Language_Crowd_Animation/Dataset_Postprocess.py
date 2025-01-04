import sys
sys.path.append('../')

import os, time, random, copy
from Simulators.ORCA_Env import ORCA_Env, AGNET_PARAM_DEFAULT
from Simulators.Field_Env import Field_Env
from Simulators.Field_Generators.Base_Field import Base_Field
from Utils import *
import numpy as np
from dataclasses import dataclass

@dataclass
class Postprocess_Config:
    # information of dataset for postprocessing
    data_path = "./Dataset/Data_Full_V1/"
    obj_nums = [0, 1, 2, 3, 4, 5]
    group_nums = [1, 2, 3]
    data_category = "training"

    target_path = "./Dataset/Data_Full_V2/"

    # params for postprocessing
    crowd_size = 50
    visual = False

    alpha = 1.5   # weight of gradient for field updating
    update_iters = 15   # num of iterations
    warm_up_steps = 20   # num of warmup steps before simulation

    agent_color = [255, 128, 0]   # color of agents
    agent_radius = 5   # radius of agents
    agent_prefV = 4   # prefer speed of agents

    max_path_len_ratio = 1.5   # the simulation steps should not exceed path_len * max_path_len_ratio / agent_prefV
    pos_rand_range = 5   # the noise range for randomizing the initial positions
    reach_dis = 40   # distance for determining the arrival of the target


def dataset_generation_v2(config):
    if not os.path.exists(config.target_path):
        os.mkdir(config.target_path)

    for obj_n in config.obj_nums:
        for group_n in config.group_nums:
            # load one piece of dataset v1
            print(f"###### reading data of obj {obj_n}, group {group_n}...")
            dt_path_og = config.data_path + f"Obj{obj_n}_Group{group_n}/dt_{config.data_category}.npy"
            print(f"### data path: {dt_path_og}")
            data_v1_og = np.load(dt_path_og, allow_pickle=True).item()

            # copy the piece of dataset v1 to the target
            data_target_og = copy.deepcopy(data_v1_og)

            # postprocess all fields in data_v1_og, and put the postprocessed fields into data_target_og
            print("### data size:", len(data_v1_og["data"]))
            for data_id in range(len(data_v1_og["data"])):
                for group_id in range(data_v1_og["data"][data_id]["group_num"]):
                    print(f"# ------  data {data_id}, group {group_id} ------ #")
                    data_target_og["data"][data_id]["group_fields"][group_id] = field_postprocess(scenario=copy.deepcopy(data_v1_og["data"][data_id]["scenario"]),
                                                                                                  sgdistrs=copy.deepcopy(data_v1_og["data"][data_id]["group_sg_distrs"][group_id]),
                                                                                                  paths=copy.deepcopy(data_v1_og["data"][data_id]["group_paths"][group_id]),
                                                                                                  initial_field=copy.deepcopy(data_v1_og["data"][data_id]["group_fields"][group_id]),
                                                                                                  config=copy.deepcopy(config))

            # save the new dataset to dataset_v2
            dt_path_og_v2 = config.target_path + f"Obj{obj_n}_Group{group_n}/"
            if not os.path.exists(dt_path_og_v2):
                os.mkdir(dt_path_og_v2)
            np.save(dt_path_og_v2+f"dt_{config.data_category}.npy", data_target_og)


def field_postprocess(scenario, sgdistrs, paths, initial_field, config):
    agent_n = config.crowd_size
    grid_width = scenario["wind_size"][0] / len(initial_field)
    grid_size = [len(initial_field), len(initial_field[0])]
    base_field = Base_Field()
    base_field.reset(scenario, grid_width)

    # prepare params for each agent
    st_distr = sgdistrs[:, :, 0].reshape(1, -1)[0]
    goal_point = copy.deepcopy(scenario["roadmap"]["vertexes"][paths[0][-1]])
    agent_params = {"init_agent_params": []}
    for aid in range(config.crowd_size):
        ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
        ai_params["pos"] = None
        ai_params["goal_pos"] = None
        ai_params["radius"] = config.agent_radius
        ai_params["pref_speed"] = config.agent_prefV
        ai_params["color"] = copy.deepcopy(config.agent_color)
        agent_params["init_agent_params"].append(copy.deepcopy(ai_params))
    # remove other useless agents (if have)
    for rst_aid in range(config.crowd_size, agent_n):
        ai_params = copy.deepcopy(AGNET_PARAM_DEFAULT)
        ai_params["pos"] = [-1000, -1000]
        ai_params["radius"] = config.agent_radius
        agent_params["init_agent_params"].append(copy.deepcopy(ai_params))

    # other vars for visualization
    lines = []
    for e_i in paths[1]:
        lines.append([copy.deepcopy(scenario["roadmap"]["vertexes"][e_i["edge"][0]]),
                      copy.deepcopy(scenario["roadmap"]["vertexes"][e_i["edge"][1]])])
    guidance = {"type": "lines", "params":{"lines": lines}}

    # calculate the allowed max steps per iteration
    path_len = 0
    for line_i in lines:
        path_len += np.linalg.norm(np.array(line_i[1])-np.array(line_i[0]))
    max_steps_per_iter = int(path_len * config.max_path_len_ratio / config.agent_prefV)

    # start postprocess loop
    start_time = time.time()
    current_field = copy.deepcopy(initial_field)
    fld_env = Field_Env(agent_num=agent_n, visual=config.visual, draw_scale=1.0)
    iter_cnt = 0
    while(1):
        iter_stt_time = time.time()
        # visualize current field
        if config.visual:
            base_field.field_visualization(field=current_field, guidance=guidance)

        # set the groups controlled by current field
        group_field_dict = {"agent_ids": list(range(config.crowd_size)), "field": copy.deepcopy(current_field), "grid": copy.deepcopy(base_field.grid)}

        # reset agent's initial positions
        st_poses = random.choices(list(range(len(st_distr))), weights=st_distr, k=config.crowd_size)
        for agt_id in range(config.crowd_size):
            agent_params["init_agent_params"][agt_id]["pos"] = [int(st_poses[agt_id]/grid_size[0])*grid_width+grid_width/2+random.uniform(-config.pos_rand_range, config.pos_rand_range),
                                                                int(st_poses[agt_id]%grid_size[0])*grid_width+grid_width/2+random.uniform(-config.pos_rand_range, config.pos_rand_range)]
            agent_params["init_agent_params"][agt_id]["goal_pos"] = copy.deepcopy(goal_point)

        # reset scenario and agents
        fld_env.reset(scenario=copy.deepcopy(scenario), agent_setting=copy.deepcopy(agent_params))


        ### ------ start simulation ------ ###
        # render wait until entered
        if config.visual:
            while(1):
                fld_env.render()
                if fld_env.viewer.entered:
                    break

        # warm up
        for stp_id in range(config.warm_up_steps):
            if config.visual:
                fld_env.render()
                time.sleep(0.003)
            super(Field_Env, fld_env).perform_action(np.zeros((agent_n, 2)).tolist())

        if config.visual:
            keyboard = pyglet.window.key.KeyStateHandler()
        field_before_update = copy.deepcopy(current_field)
        upd_map = np.zeros((grid_size[0], grid_size[1]))
        step = 0
        while(1):
            if config.visual:
                fld_env.viewer.push_handlers(keyboard)
                if keyboard[pyglet.window.key.Q]:
                    fld_env.set_viewer_field(group_field_dict["grid"], group_field_dict["field"])
                elif keyboard[pyglet.window.key.P]:
                    fld_env.set_viewer_field(None, None)
                elif keyboard[pyglet.window.key.A]:
                    fld_env.set_viewer_guidance(guidance)
                elif keyboard[pyglet.window.key.L]:
                    fld_env.set_viewer_guidance(None)
                fld_env.render()
                time.sleep(0.003)

            # remove the agents that get out of the window range or reach the goal
            for aid in group_field_dict["agent_ids"]:
                if np.linalg.norm(np.array(fld_env.agent_current_infor[aid]["pos"])-np.array(fld_env.agent_current_infor[aid]["goal_pos"]))<config.reach_dis or \
                        fld_env.agent_current_infor[aid]["pos"][0]<=1e-6 or fld_env.agent_current_infor[aid]["pos"][0]>=scenario["wind_size"][0]-1e-6 or \
                        fld_env.agent_current_infor[aid]["pos"][1]<=1e-6 or fld_env.agent_current_infor[aid]["pos"][1]>=scenario["wind_size"][1]-1e-6:
                    fld_env.set_agent_position(aid, [-1000, -1000])
                    group_field_dict["agent_ids"].remove(aid)

            # perform action based on the group_field
            pre_agent_positions = fld_env.get_current_positions()
            fld_env.perform_action_fast([group_field_dict])
            aft_agent_positions = fld_env.get_current_positions()

            ### Get the true velocity of each agent in group_field_dict["agent_ids"] and update the current field
            # get useful agents' positions before and after executing
            pre_positions_grp = np.array(pre_agent_positions)[group_field_dict["agent_ids"]]
            aft_positions_grp = np.array(aft_agent_positions)[group_field_dict["agent_ids"]]
            # get the indexs of each agent on the grid
            agt_grid_idxs = (np.array(pre_positions_grp) / grid_width).astype(int)
            agt_grid_idxs[agt_grid_idxs<0] = 0
            agt_grid_idxs[:,0][agt_grid_idxs[:,0]>(grid_size[0]-1)] = (grid_size[0]-1)
            agt_grid_idxs[:,1][agt_grid_idxs[:,1]>(grid_size[1]-1)] = (grid_size[1]-1)
            # get the gradient map
            grads_ = np.array(aft_positions_grp) - np.array(pre_positions_grp)
            grads_ = np.nan_to_num(grads_ / (np.linalg.norm(grads_, axis=1).reshape((-1, 1))), 0.)  # normalization
            gradient_map = np.zeros_like(current_field)
            gradient_map[agt_grid_idxs[:, 0], agt_grid_idxs[:, 1]] = grads_
            upd_map[agt_grid_idxs[:, 0], agt_grid_idxs[:, 1]] = 1
            # update current field
            current_field += config.alpha * gradient_map
            curr_field_norm = np.linalg.norm(current_field, axis=2)
            curr_field_norm = curr_field_norm.reshape((curr_field_norm.shape[0], curr_field_norm.shape[1], 1))
            current_field = np.nan_to_num(current_field / curr_field_norm, 0)

            step += 1

            if len(group_field_dict["agent_ids"]) == 0 or step >= max_steps_per_iter:
                break

        total_aver_grads = np.sum(np.linalg.norm(np.array(current_field)-np.array(field_before_update), axis=2)) / np.sum(upd_map)

        print(f"iteration {iter_cnt} | simulated steps: {step} | total_aver_grads: {total_aver_grads}"
              f" | time: {time.time()-iter_stt_time}s | total time: {time.time()-start_time}")

        iter_cnt += 1
        if iter_cnt >= config.update_iters:
            break

    if config.visual and not fld_env.viewer.closed:
        fld_env.viewer.close()

    # remove the vectors that inside the obstacles in the grid map
    obs_idx = (np.argwhere(np.array(base_field.grid["grid_map"]) == 1))
    current_field[obs_idx[:, 0], obs_idx[:, 1]] = np.array([0., 0.])

    return current_field


def data_visual(config, obj_n, group_n, rand_data_num=10):
    from Dataset_Generation import field_control_visualization
    data_og = np.load(config.target_path+f"Obj{obj_n}_Group{group_n}/dt_{config.data_category}.npy", allow_pickle=True).item()

    for i in range(rand_data_num):
        data_id = random.randint(0, len(data_og["data"])-1)
        data_for_visual = data_og["data"][data_id]

        for group_id in range(data_for_visual["group_num"]):
            print(data_for_visual["group_descriptors"][group_id])
            print(data_for_visual["group_descriptions"][group_id])
        print("group sizes:", data_for_visual["group_sizes"])

        cv_visual_map(map_in=data_for_visual["semantic_map"], colors=None, save_nm=None, show=True)

        field_control_visualization(scenario=data_for_visual["scenario"], group_distrs=data_for_visual["group_sg_distrs"],
                                    group_paths=data_for_visual["group_paths"], group_sizes=data_for_visual["group_sizes"],
                                    group_fields=data_for_visual["group_fields"], show_path=True, show_fields=True)


if __name__ == '__main__':
    config = Postprocess_Config()
    dataset_generation_v2(config)

    # data_visual(config, 5, 3, rand_data_num=2)
