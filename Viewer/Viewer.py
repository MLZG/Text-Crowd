import time
import torch
import numpy as np
from Utils import *
import math,pyglet,cv2,os

class Viewer(pyglet.window.Window):
    def __init__(self, wind_size=(800,800), checker=(800,800,[140,140,140])):
        config = pyglet.gl.Config(sample_buffers=1, samples=16)
        super(Viewer, self).__init__(config=config, resizable=False,
                                     width=wind_size[0], height=wind_size[1],
                                     caption='viewer',
                                     vsync=True)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()
        self.closed = False
        self.entered = False

        self.batch_checker = None
        if checker is not None:
            vss=[]
            for x in range(0,wind_size[0],checker[0]):
                for y in range(0,wind_size[1],checker[1]):
                    if (x//checker[0]+y//checker[1])%2==0:
                        vss+=get_box(checker[0], checker[1], lowerleft=(x,y), triangle=True)
            self.batch_checker=self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLES, None, ('v2f', vss), ('c3B', tuple(checker[2])*(len(vss)//2)))

        self.reset_array()


    def reset_array(self):
        self.agent_pos_array = []
        self.batch_agent_list = []
        self.batch_agent_edge_list = []

        self.goal_pos_array = []
        self.batch_goal_list = []
        self.batch_goal_edge_list = []

        self.batch_obs_list = []
        self.batch_obs_edge_list = []

        self.waypoint_pos_array = []
        self.batch_waypoint_list = []
        self.batch_waypoint_edge_list = []

        self.batch_line_list = []

        self.batch_zebrabox_list = []
        self.batch_zebrabox_edge_list = []
        self.batch_door_list = []
        self.batch_door_edge_list = []

        self.traj_history_list = []
        self.arrows = []
        self.particle_circles = []
        self.particle_circles_ps = []

        self.agent_vis_inner_list = []
        self.agent_vis_edge_list = []

    def add_zebra_box(self, box_points):
        vss = box_points
        self.batch_zebrabox_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss),
                                                       ('c3B', tuple([255, 255, 255]) * (len(vss) // 2))))
        self.batch_zebrabox_edge_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss),
                                                            ('c3B', tuple([130, 130, 130]) * (len(vss) // 2))))

    def add_door_box(self, box_points, color=[140, 140, 140]):
        vss = box_points
        self.batch_door_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss),
                                                       ('c3B', tuple(color) * (len(vss) // 2))))
        self.batch_door_edge_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss),
                                                            ('c3B', tuple(color) * (len(vss) // 2))))

    def add_waypoint(self, pos, waypoint_size=10, color=[0, 0, 255]):
        self.waypoint_pos_array.append(list(pos))
        vss = get_box(waypoint_size, waypoint_size)
        self.batch_waypoint_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss),
                                                       ('c3B', tuple(color) * (len(vss) // 2))))
        self.batch_waypoint_edge_list.append(self.batch.add(len(vss) // 2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss),
                                                            ('c3B', tuple([0, 0, 0]) * (len(vss) // 2))))

    def add_line(self, p1, p2, color=[0, 255, 0]):
        vss = np.array([p1, p2]).reshape(1, -1)[0].tolist()
        self.batch_line_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))

    def add_agent(self, pos, rad, color=[124,79,13]):
        self.agent_pos_array.append(list(pos))
        vss=get_circle(rad)
        self.batch_agent_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))
        vss=get_circle(rad, fill=False)
        self.batch_agent_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c3B', tuple([0,0,0])*(len(vss)//2))))

    def add_goal(self, pos, goal_size=10, color=[230,13,13]):
        if pos is None:
            self.goal_pos_array.append(None)
            self.batch_goal_list.append(None)
            self.batch_goal_edge_list.append(None)
            return
        self.goal_pos_array.append(list(pos))
        vss=get_box(goal_size, goal_size)
        self.batch_goal_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))
        self.batch_goal_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_LOOP, None, ('v2f', vss), ('c3B', tuple([0,0,0])*(len(vss)//2))))

    def add_obs(self, vss, color=[100,100,100]):
        self.batch_obs_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c3B', tuple(color)*(len(vss)//2))))
        self.batch_obs_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c3B', tuple([70,70,70])*(len(vss)//2))))

    def set_traj(self, trajs, colors):
        self.traj_history_list = []
        for tj, color in zip(trajs, colors):
            self.traj_history_list.append(self.batch.add(len(tj)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', tj), ('c3B', tuple(color)*(len(tj)//2))))

    def set_arrows(self, arrows, colors):
        self.arrows = []
        for arrow, color in zip(arrows, colors):
            vss1, vss2 = get_arrow(arrow[0], arrow[1])
            self.arrows.append(self.batch.add(len(vss1)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss1), ('c3B', tuple(color)*(len(vss1)//2))))
            self.arrows.append(self.batch.add(len(vss2)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss2), ('c3B', tuple(color)*(len(vss2)//2))))

    def add_particle(self, radius, color, trail_len, decay):
        alpha = 255
        if decay is None:
            decay = int(255 / trail_len)
        circle_list = []
        circle_pos_list = []
        for t_id in range(trail_len):
            circle_ = get_circle(radius)
            color_4 = (color[0], color[1], color[2], alpha)
            circle_list.append(self.batch.add(len(circle_)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', circle_), ('c4B', tuple(color_4)*(len(circle_)//2))))
            circle_pos_list.append([-1e5, -1e5])
            alpha = max(alpha-decay, 0)
        circle_list.reverse()
        circle_pos_list.reverse()
        self.particle_circles.append(circle_list)
        self.particle_circles_ps.append(circle_pos_list)
        return len(self.particle_circles)-1

    def delete_particle(self, particle_id):
        self.particle_circles[particle_id] = None
        self.particle_circles_ps[particle_id] = None

    def set_particle_trail(self, particle_id, traj):
        self.particle_circles_ps[particle_id] = np.array(traj).tolist()

    def set_agent_vis_circles(self, agents_ps_list, agents_radius_list, agents_color_list):
        assert len(agents_ps_list)==len(agents_radius_list)==len(agents_color_list)
        self.agent_vis_inner_list = []
        self.agent_vis_edge_list = []
        for agent_id in range(len(agents_ps_list)):
            vss = get_circle(agents_radius_list[agent_id])
            vss = (np.array(vss).reshape(-1, 2) + np.array(agents_ps_list[agent_id])).reshape(1, -1)[0].tolist()
            color_4 = agents_color_list[agent_id]
            self.agent_vis_inner_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_TRIANGLE_FAN, None, ('v2f', vss), ('c4B', tuple(color_4)*(len(vss)//2))))
            vss = get_circle(agents_radius_list[agent_id], fill=False)
            vss = (np.array(vss).reshape(-1, 2) + np.array(agents_ps_list[agent_id])).reshape(1, -1)[0].tolist()
            color_4_eg = [0, 0, 0, color_4[-1]]
            self.agent_vis_edge_list.append(self.batch.add(len(vss)//2, pyglet.gl.GL_LINE_STRIP, None, ('v2f', vss), ('c4B', tuple(color_4_eg)*(len(vss)//2))))

    def on_key_press(self, symbol, modifiers):
        if symbol==pyglet.window.key.ESCAPE:
            self.close()
            self.closed = True
        elif symbol==pyglet.window.key.ENTER:
            self.entered = not self.entered

    def on_draw(self):
        self.clear()
        pyglet.gl.glLineWidth(2)
        pyglet.gl.glPointSize(10)

        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

        if self.batch_checker is not None:
            self.batch_checker.draw(pyglet.gl.GL_TRIANGLES)

        for doorbox, doorbox_edge in zip(self.batch_door_list, self.batch_door_edge_list):
            doorbox.draw(pyglet.gl.GL_TRIANGLE_FAN)
            doorbox_edge.draw(pyglet.gl.GL_LINE_LOOP)

        for zebrabox, zebrabox_edge in zip(self.batch_zebrabox_list, self.batch_zebrabox_edge_list):
            zebrabox.draw(pyglet.gl.GL_TRIANGLE_FAN)
            zebrabox_edge.draw(pyglet.gl.GL_LINE_LOOP)

        for vertex_list,edge_list in zip(self.batch_obs_list,self.batch_obs_edge_list):
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)

        for goal_pos,vertex_list,edge_list in zip(self.goal_pos_array,self.batch_goal_list,self.batch_goal_edge_list):
            if goal_pos is None:
                continue
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(goal_pos[0],goal_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)
            pyglet.gl.glPopMatrix()

        for arrow_lines in self.arrows:
            arrow_lines.draw(pyglet.gl.GL_LINE_STRIP)

        for trajs in self.traj_history_list:
            trajs.draw(pyglet.gl.GL_LINE_STRIP)

        for trail_circle_ps, trail_circles in zip(self.particle_circles_ps, self.particle_circles):
            if trail_circle_ps is None or trail_circles is None:
                continue
            for circle_p, circle_ in zip(trail_circle_ps, trail_circles):
                pyglet.gl.glPushMatrix()
                pyglet.gl.glTranslatef(circle_p[0], circle_p[1], 0)
                circle_.draw(pyglet.gl.GL_TRIANGLE_FAN)
                pyglet.gl.glPopMatrix()

        for line_i in self.batch_line_list:
            line_i.draw(pyglet.gl.GL_LINE_STRIP)

        for agent_pos,vertex_list,edge_list in zip(self.agent_pos_array,self.batch_agent_list,self.batch_agent_edge_list):
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(agent_pos[0],agent_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_STRIP)
            pyglet.gl.glPopMatrix()

        for waypoint_pos,vertex_list,edge_list in zip(self.waypoint_pos_array,self.batch_waypoint_list,self.batch_waypoint_edge_list):
            pyglet.gl.glPushMatrix()
            pyglet.gl.glTranslatef(waypoint_pos[0],waypoint_pos[1],0)
            vertex_list.draw(pyglet.gl.GL_TRIANGLE_FAN)
            edge_list.draw(pyglet.gl.GL_LINE_LOOP)
            pyglet.gl.glPopMatrix()

        for vis_agent_inner, vis_agent_edge in zip(self.agent_vis_inner_list, self.agent_vis_edge_list):
            vis_agent_inner.draw(pyglet.gl.GL_TRIANGLE_FAN)
            vis_agent_edge.draw(pyglet.gl.GL_LINE_STRIP)

        if hasattr(self,'sensor') and hasattr(self.sensor,'readings'):
            self.sensor.draw_sensor_reading()

    def render(self):
        try:
            self.switch_to()
            self.dispatch_events()
            self.dispatch_event('on_draw')
            self.flip()
        except:
            self.close()

if __name__=='__main__':
    from Utils import *
    viewer=Viewer()
    viewer.add_agent(pos=(100,100), rad=10)
    viewer.add_goal(pos=(700,700))
    viewer.add_obs(vss=get_box(400, 400, lowerleft=(200,200)))
    pyglet.app.run()