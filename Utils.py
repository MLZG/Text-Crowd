import time
import copy
import shapely.geometry as geom
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

def distance_checker(pos, obs_list):
    polys_exact=[]
    for obs in obs_list:
        polys_exact.append(geom.Polygon([[p[0],p[1]] for p in obs]))
    ret=1e10
    for poly in polys_exact:
        ret=min(ret,poly.distance(geom.Point(pos)))
    return ret

def collision_checker_point(point, obs_list):
    polys=[]
    for obs in obs_list:
        polys.append(geom.Polygon([[p[0],p[1]] for p in obs]))
    for poly in polys:
        if poly.intersects(geom.Point(point)):
            return False
    return True

def collision_checker_line(posa, posb, obs_list, has_buffer=False, dis=0):
    polys_exact=[]
    for obs in obs_list:
        polys_exact.append(geom.Polygon([[p[0],p[1]] for p in obs]))
    for poly in polys_exact:
        if has_buffer:
            line = geom.LineString([posa, posb])
            if poly.intersects(line.buffer(dis)):
                return False
        elif poly.intersects(geom.LineString([posa,posb])):
            return False
    return True

def collision_checker(pos, obs_list, buf=None):
    buffer_value = 0
    if buf is not None:
        buffer_value = buf
    polys=[]
    for obs in obs_list:
        polys.append(geom.Polygon([[p[0],p[1]] for p in obs]).buffer(buffer_value))
    for poly in polys:
        if poly.intersects(geom.Point(pos)):
            return False
    return True

def poly_collision_check(polys, poly_new):
    for ply_i in polys:
        if ply_i.intersects(poly_new):
            return False
    return True

def make_ccw(pts):
    is_ccw=False
    for i in range(len(pts)):
        iLast=(i+len(pts)-1)%len(pts)
        dirLast=(pts[i][0]-pts[iLast][0],pts[i][1]-pts[iLast][1])

        iNext=(i+1)%len(pts)
        dirNext=(pts[iNext][0]-pts[i][0],pts[iNext][1]-pts[i][1])

        nLast=(-dirLast[1],dirLast[0])
        dotLastNext=nLast[0]*dirNext[0]+nLast[1]*dirNext[1]
        if dotLastNext>0:
            is_ccw=True
        elif dotLastNext<0:
            is_ccw=False
            break
    if not is_ccw:
        return [pts[len(pts)-1-i] for i in range(len(pts))]
    else: return pts


def get_circle(r, fill=True, RES=16):
    circle_list = [0.0, 0.0] if fill else []
    for i in range(RES+1):
        circle_list += [r*math.cos(math.pi*2*i/RES), r*math.sin(math.pi*2*i/RES)]
    return circle_list


def get_box(x, y, ctr=None, lowerleft=None, triangle=False):
    if triangle:
        vss=[-x/2, -y/2, x/2, -y/2, x/2, y/2,   -x/2, -y/2, x/2, y/2, -x/2, y/2]
    else: vss=[-x/2, -y/2, x/2, -y/2, x/2, y/2, -x/2, y/2]
    if ctr is not None:
        for i in range(len(vss)//2):
            vss[i*2+0]+=ctr[0]
            vss[i*2+1]+=ctr[1]
    if lowerleft is not None:
        for i in range(len(vss)//2):
            vss[i*2+0]+=lowerleft[0]+x/2
            vss[i*2+1]+=lowerleft[1]+y/2
    return vss


def get_box_ll(x, y, ctr=None, lowerleft=None, triangle=False):
    vss=get_box(x, y, ctr, lowerleft, triangle)
    return [[vss[i],vss[i+1]] for i in range(0,len(vss),2)]


def get_arrow(ori, tgt, arrow_len=None):
    if arrow_len is None:
        len = np.linalg.norm(np.array(tgt)-np.array(ori))/4
    else:
        len = arrow_len
    dre = [1.0, 1.0]
    vec = np.array(tgt) - ori
    cos_ = vec[0]/np.linalg.norm(np.array(vec))
    sin_ = vec[1]/np.linalg.norm(np.array(vec))
    p = [cos_*-len*dre[0], sin_*-len*dre[1]]
    if(vec[0]>=0):
        dre[0] = -1.0
    if(vec[1]>=0):
        dre[1] = -1.0
    line1 = np.array([ori, tgt]).reshape(1, -1)[0].tolist()
    line2 = np.array([p[0]*math.cos(-0.5)-p[1]*math.sin(-0.5)+tgt[0], p[0]*math.sin(-0.5)+p[1]*math.cos(-0.5)+tgt[1],
                      tgt[0], tgt[1],
                      p[0]*math.cos(0.5)-p[1]*math.sin(0.5)+tgt[0], p[0]*math.sin(0.5)+p[1]*math.cos(0.5)+tgt[1]]).tolist()
    return line1, line2

def vector_rotation(vec, angle):
    rot_mat = np.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    return rot_mat.dot(np.array(vec))

def vectors_rotation(vec_list, angle):
    r_vec_list = []
    for vec_ in vec_list:
        r_vec_list.append(vector_rotation(copy.deepcopy(np.array(vec_)), angle))
    r_vec_list = np.array(r_vec_list).tolist()
    return r_vec_list

def grid_discretization(wind_size, grid_width, obs_list):
    grid_ = []
    grid_infor = []
    obs_polys=[]
    for obs_i in obs_list:
        obs_polys.append(geom.Polygon([[p[0],p[1]] for p in obs_i]))
    for i in range(int(wind_size[0]/grid_width)):
        grid_rowi = []
        grid_infor_rowi = []
        for j in range(int(wind_size[1]/grid_width)):
            center_ = [i*grid_width+grid_width/2, j*grid_width+grid_width/2]
            edge_len = grid_width
            box_ = [[center_[0]-edge_len/2, center_[1]-edge_len/2], [center_[0]+edge_len/2, center_[1]-edge_len/2],
                    [center_[0]+edge_len/2, center_[1]+edge_len/2], [center_[0]-edge_len/2, center_[1]+edge_len/2]]
            box_poly = geom.Polygon([[p[0],p[1]] for p in box_])
            free = 0
            for obs_pl in obs_polys:
                if box_poly.intersects(obs_pl):
                    free = 1
                    break
            grid_rowi.append(free)
            grid_infor_rowi.append({"center": center_, "edge_len": edge_len, "box": box_, "free": free})
        grid_.append(grid_rowi)
        grid_infor.append(grid_infor_rowi)
    return grid_, grid_infor


def bilinear_interpolation(x, y, points, values):
    pr_1 = np.array(points)[:, 0]
    pr_2 = np.array(points)[:, 1]
    agst = np.lexsort((pr_2, pr_1))
    points = np.array(points)[agst]
    values = np.array(values)[agst]
    [[x1, y1], [x1_, y2], [x2, y1_], [x2_, y2_]] = points
    v11, v12, v21, v22 = values
    if (x1 != x1_) or (x2 != x2_) or (y1 != y1_) or (y2 != y2_):
        print("Not Rectangle!")
        return None
    if not (x1<=x<=x2) or not (y1<=y<=y2):
        print("Point Not within Rectangle!")
        return None
    ans = (v11*(x2-x)*(y2-y)+v21*(x-x1)*(y2-y)+v12*(x2-x)*(y-y1)+v22*(x-x1)*(y-y1))/((x2-x1)*(y2-y1)+0.0)
    # for checking
    px1 = v11*((x2-x)/(x2-x1)) + v21*((x-x1)/(x2-x1))
    px2 = v12*((x2-x)/(x2-x1)) + v22*((x-x1)/(x2-x1))
    ans1 = px1*((y2-y)/(y2-y1)) + px2*((y-y1)/(y2-y1))
    if np.linalg.norm(np.array(ans1)-np.array(ans)) >= 1e-6:
        print("Func Wrong!")
        return None
    return ans

def interp_grid_fast(x, y, xp, yp, zp):
    """
    Bilinearly interpolate over regular 2D grid.

    `xp` and `yp` are 1D arrays defining grid coordinates of sizes :math:`N_x`
    and :math:`N_y` respectively, and `zp` is the 2D array, shape
    :math:`(N_x, N_y)`, containing the gridded data points which are being
    interpolated from. Note that the coordinate grid should be regular, i.e.
    uniform grid spacing. `x` and `y` are either scalars or 1D arrays giving
    the coordinates of the points at which to interpolate. If these are outside
    the boundaries of the coordinate grid, the resulting interpolated values
    are evaluated at the boundary.

    Parameters
    ----------
    x : 1D array or scalar
        x-coordinates of interpolating point(s).
    y : 1D array or scalar
        y-coordinates of interpolating point(s).
    xp : 1D array, shape M
        x-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    yp : 1D array, shape N
        y-coordinates of data points zp. Note that this should be a *regular*
        grid, i.e. uniform spacing.
    zp : 2D array, shape (M, N)
        Data points on grid from which to interpolate.

    Returns
    -------
    z : 1D array or scalar
        Interpolated values at given point(s).

    """
    # if scalar, turn into array
    scalar = False
    if not isinstance(x, (list, np.ndarray)):
        scalar = True
        x = np.array([x])
        y = np.array([y])

    # grid spacings and sizes
    hx = xp[1] - xp[0]
    hy = yp[1] - yp[0]
    Nx = xp.size
    Ny = yp.size

    # snap beyond-boundary points to boundary
    x[x < xp[0]] = xp[0]
    y[y < yp[0]] = yp[0]
    x[x > xp[-1]] = xp[-1]
    y[y > yp[-1]] = yp[-1]

    # find indices of surrounding points
    i1 = np.floor((x - xp[0]) / hx).astype(int)
    i1[i1 == Nx - 1] = Nx - 2
    j1 = np.floor((y - yp[0]) / hy).astype(int)
    j1[j1 == Ny - 1] = Ny - 2
    i2 = i1 + 1
    j2 = j1 + 1

    # get coords and func at surrounding points
    x1 = xp[i1]
    x2 = xp[i2]
    y1 = yp[j1]
    y2 = yp[j2]
    z11 = zp[i1, j1]
    z21 = zp[i2, j1]
    z12 = zp[i1, j2]
    z22 = zp[i2, j2]

    # interpolate
    t11 = z11 * (x2 - x) * (y2 - y)
    t21 = z21 * (x - x1) * (y2 - y)
    t12 = z12 * (x2 - x) * (y - y1)
    t22 = z22 * (x - x1) * (y - y1)
    z = (t11 + t21 + t12 + t22) / (hx * hy)
    if scalar:
        z = z[0]
    return z


def interp_grid_closest_4(x, y, xp, yp, grid_values):
    # get the 2d grid coords
    gc_x, gc_y = np.meshgrid(xp, yp)
    grid_coords = np.concatenate((gc_x[:, :, None], gc_y[:, :, None]), axis=-1)
    grid_coords = grid_coords.transpose((1,0,2))

    # grid spacings and sizes
    hx = xp[1] - xp[0]
    hy = yp[1] - yp[0]
    Nx = xp.size
    Ny = yp.size

    # snap beyond-boundary points to boundary
    x[x < xp[0]] = xp[0]
    y[y < yp[0]] = yp[0]
    x[x > xp[-1]] = xp[-1]
    y[y > yp[-1]] = yp[-1]

    # find indices of surrounding points
    x_min = np.floor((x - xp[0]) / hx).astype(int)
    x_min[x_min == Nx - 1] = Nx - 2
    y_min = np.floor((y - yp[0]) / hy).astype(int)
    y_min[y_min == Ny - 1] = Ny - 2

    x_min = x_min - 1
    y_min = y_min - 1
    x_min[x_min<0] = 0
    y_min[y_min<0] = 0

    x_max = x_min + 3
    y_max = y_min + 3
    x_max[x_max>(Nx-1)] = Nx - 1
    y_max[y_max>(Ny-1)] = Ny - 1

    ans = []
    assert len(x) == len(y)
    assert np.array(grid_values).shape[2] == 2
    for idx in range(len(x)):
        points = grid_coords[x_min[idx]:x_max[idx]+1, y_min[idx]:y_max[idx]+1]
        points = points.reshape((points.shape[0]*points.shape[1], -1))
        point_values = grid_values[x_min[idx]:x_max[idx]+1, y_min[idx]:y_max[idx]+1]
        point_values = point_values.reshape((point_values.shape[0]*point_values.shape[1], -1))
        p_xy = np.array([x[idx], y[idx]])

        dis_list = np.linalg.norm(np.array(points) - np.array(p_xy), axis=1)
        sorted_ids = np.argsort(dis_list)
        found_ = False
        for id_ in sorted_ids:
            if abs(np.linalg.norm(point_values[id_]) - 0.) >= 1e-6:
                found_ = True
                ans.append(copy.deepcopy(point_values[id_]))
                break
        if not found_:
            ans.append(np.array([0., 0.]))
            # ans.append(np.random.uniform(-1, 1, (2,)))

    return np.array(ans)



def get_angle(vec_1, vec_2):
    v1 = np.array(vec_1)[1] - np.array(vec_1)[0]
    v2 = np.array(vec_2)[1] - np.array(vec_2)[0]
    return np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))/np.pi*180

def get_signed_angle(vec_1, vec_2):
    v1 = np.array(vec_1)[1] - np.array(vec_1)[0]
    v2 = np.array(vec_2)[1] - np.array(vec_2)[0]
    return math.degrees(math.atan2(v1[0]*v2[1]-v1[1]*v2[0], np.dot(v1, v2)))


import signal
class TimeoutException(BaseException):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Time out error!")
def set_timeout(t_scd):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(t_scd)
def clear_timeout():
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)


def cv_visual_map(map_in, colors=None, save_nm=None, show=False):
    map_ = np.flip(np.array(map_in), axis=1).transpose(1, 0, 2)
    if colors is None:
        colors = np.random.randint(0, 255, (len(map_[0][0]), 3))
    img_ = np.zeros((len(map_), len(map_[0]), 3))
    for i in range(len(map_)):
        for j in range(len(map_[0])):
            img_[i][j] = (np.sum(map_[i][j].reshape(len(map_[i][j]), 1) * np.array(colors), axis=0))
    if show:
        cv2.imshow("img", img_/255)
        cv2.waitKey(0)
    if save_nm is not None:
        cv2.imwrite(save_nm, img_)
    return img_

def cv_visual_field(field_in, grid_width=10, save_nm=None, show=False):
    field_ = np.flip(np.array(field_in), axis=1).transpose(1, 0, 2)
    img = np.ones((int(grid_width*len(field_)), int(grid_width*len(field_[0])), 3))*255
    for xid in range(len(field_)):
        for yid in range(len(field_[0])):
            cv2.rectangle(img, (int(yid*grid_width), int(xid*grid_width)),
                          (int((yid+1)*grid_width), int((xid+1)*grid_width)), (0, 0, 0), thickness=1)
            vec = np.array([field_[xid][yid][0], -field_[xid][yid][1]])
            arrow_ = np.array([[0, 0], vec])
            arrow_ = (arrow_-vec/2)*grid_width*0.9
            arrow_ += np.array([int((yid+0.5)*grid_width), int((xid+0.5)*grid_width)])
            arrow_ = arrow_.astype(int)
            cv2.arrowedLine(img, arrow_[0], arrow_[1], (0, 0, 0), line_type=cv2.LINE_AA, tipLength=0.3)
    if show:
        cv2.imshow("img", img/255)
        cv2.waitKey(0)
    if save_nm is not None:
        cv2.imwrite(save_nm, img)
    return img


def traj_smoothing(traj_in, s_=10):
    x = np.array(traj_in)[:, 0]
    y = np.array(traj_in)[:, 1]
    tck, u = splprep([x, y], s=s_)
    new_points = splev(u, tck)
    # # plot
    # plt.plot(x, y, "ro")
    # plt.scatter(new_points[0], new_points[1])
    # plt.show()
    # save to traj_out
    traj_out = np.zeros_like(traj_in)
    traj_out[:, 0] = new_points[0].copy()
    traj_out[:, 1] = new_points[1].copy()
    return traj_out



import pyDeclutter,pyglet,math

class Sensor:
    def __init__(self, agent_r, obs_all, sensorRange=100, N=64):
        self.agent_r = agent_r
        self.obs_all = obs_all
        self.sensorRange=sensorRange
        self.dirs=[(math.cos(math.pi*2*j/N),math.sin(math.pi*2*j/N)) for j in range(N)]
        self.N=N
        self.reset()

    def reset(self):
        #setup obs
        lss=[]
        for obs in self.obs_all:
            for i in range(len(obs)):
                j=(i+1)%len(obs)
                a=pyDeclutter.Vec2(obs[i][0],obs[i][1])
                b=pyDeclutter.Vec2(obs[j][0],obs[j][1])
                lss.append(pyDeclutter.LineSeg2D(a,b))
        if len(lss)==0:
            return
        self.envCpp=pyDeclutter.Environment2D(lss)

    def get_sensor_reading(self, positions):
        if len(self.obs_all) == 0:
            self.readings = [[max(self.sensorRange-self.agent_r, 0) for si in range(self.N)] for ip, p in enumerate(positions)]
            return self.readings
        pss=[pyDeclutter.Vec2(p[0],p[1]) for p in positions]
        self.envCpp.setAgent(pss, self.agent_r)
        self.readings=[[max(v - self.agent_r, 0) for v in self.envCpp.sensorData(p, self.sensorRange, self.N, ip)] for ip, p in enumerate(pss)]
        return self.readings