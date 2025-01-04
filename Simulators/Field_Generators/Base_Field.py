from Simulators.ORCA_Env import ORCA_Env
from Viewer.Viewer import Viewer
from Utils import *
import numpy as np
import shapely.geometry as geom
import math,time,os, random
import rasterio
from rasterio.features import geometry_mask
import copy

class Base_Field:
    def __init__(self):
        pass

    def reset(self, scenario, grid_width):
        self.scenario = copy.deepcopy(scenario)
        self.grid_width = grid_width
        self.grid = self.space_discretization()  # keys: wind_size, grid_width, grid_size, grid_map, grid_infor

    def space_discretization(self):
        grid = {}
        wind_size = copy.deepcopy(self.scenario["wind_size"])
        grid["wind_size"] = copy.deepcopy(wind_size)
        grid["grid_width"] = self.grid_width
        assert (int(wind_size[0]/self.grid_width))*self.grid_width == wind_size[0]
        assert (int(wind_size[1]/self.grid_width))*self.grid_width == wind_size[1]
        grid_size = [int(wind_size[0]/self.grid_width), int(wind_size[1]/self.grid_width)]
        grid["grid_size"] = copy.deepcopy(grid_size)

        grid_map = np.zeros((grid_size[0], grid_size[1]), dtype=int)
        grid_infor = []

        obs_list = ORCA_Env.get_all_obstacles_from_scenario(scenario_in=self.scenario)
        if len(obs_list) != 0:
            obs_polys = []
            for obs_i in obs_list:
                obs_polys.append(geom.Polygon(obs_i))
            transform_ = rasterio.transform.from_bounds(0, 0, wind_size[0], wind_size[1], grid_size[0], grid_size[1])
            geom_mask = rasterio.features.geometry_mask(obs_polys, out_shape=(grid_size[1], grid_size[0]), transform=transform_, all_touched=True)
            geom_mask = np.flip(geom_mask, axis=0).transpose(1, 0)
            grid_map[~geom_mask] = 1
        for i in range(grid_size[0]):
            grid_infor_rowi = []
            for j in range(grid_size[1]):
                center_ = [i*self.grid_width+self.grid_width/2, j*self.grid_width+self.grid_width/2]
                edge_len = self.grid_width
                box_ = [[center_[0]-edge_len/2, center_[1]-edge_len/2], [center_[0]+edge_len/2, center_[1]-edge_len/2],
                        [center_[0]+edge_len/2, center_[1]+edge_len/2], [center_[0]-edge_len/2, center_[1]+edge_len/2]]
                grid_infor_rowi.append({"center": center_, "edge_len": edge_len, "box": box_, "free": grid_map[i][j]})
            grid_infor.append(grid_infor_rowi)
        grid_infor = np.array(grid_infor)
        grid["grid_map"] = grid_map
        grid["grid_infor"] = grid_infor

        return grid


    def get_field(self, guidance, constrains=None):
        pass

    def field_visualization(self, field, guidance=None):
        # visualization
        wind_size = copy.deepcopy(self.grid["wind_size"])
        grid_size = copy.deepcopy(self.grid["grid_size"])
        grid_width = self.grid_width

        viewer = Viewer(wind_size=tuple((int(wind_size[0]), int(wind_size[1]))))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if self.grid["grid_infor"][i][j]["free"] == 1:
                    viewer.add_obs(np.array(self.grid["grid_infor"][i][j]["box"]).reshape(1, -1)[0].tolist())
                else:
                    viewer.add_obs(np.array(self.grid["grid_infor"][i][j]["box"]).reshape(1, -1)[0].tolist(), [200, 200, 200])
        arrows = []
        arrow_colors = []
        for i in range(len(field)):
            for j in range(len(field[0])):
                vec = field[i][j]
                arrow = np.array([[0, 0], vec])
                arrow = (arrow - vec/2)*grid_width
                arrow += np.array(self.grid["grid_infor"][i][j]["center"])
                arrows.append(arrow)
                arrow_colors.append([0, 0, 0])
        viewer.set_arrows(np.array(arrows), np.array(arrow_colors))

        if guidance is not None and guidance["type"] == "lines":
            lines = guidance["params"]["lines"]
            trajs = []
            traj_colors = []
            for lidx, line_i in enumerate(lines):
                trajs.append(np.array(line_i).reshape(1, -1)[0].tolist())
                traj_colors.append([255, 0, 0])
            viewer.set_traj(np.array(trajs), np.array(traj_colors))

        while(1):
            viewer.render()
            if viewer.closed:
                break

    def fields_blending(self, fields, weights):
        return np.average(np.array(fields), axis=0, weights=np.array(weights)/np.sum(weights))


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

    fld = Base_Field()
    fld.reset(scenario_test, grid_width)
    fld.field_visualization(np.zeros((fld.grid["grid_size"][0], fld.grid["grid_size"][1], 2)), None)

