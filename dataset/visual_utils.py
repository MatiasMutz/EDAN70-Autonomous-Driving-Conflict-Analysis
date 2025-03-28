import matplotlib.pyplot as plt
import zarr
import numpy as np
from scipy.spatial import ConvexHull


def read_scenario(filename, root):
    dt = zarr.open(root+filename, mode='r')

    slices = dt.index[:]
    timestep = dt.timestep[:]
    motion = dt.motion[:]
    type = dt.category[:]
    maps = dt.lane[:]

    return slices, timestep, motion, type, maps


def simplify_trajectory(xx,yy):
    poly = ConvexHull([[x,y] for x,y in zip(xx, yy)]) 
    if (poly.area / np.linalg.norm([xx[-1]-xx[0], yy[-1]-yy[0]])) >= 2.05:
        xx, yy = xx[poly.vertices], yy[poly.vertices]
        if poly.vertices[np.argmin(xx**2+yy**2)-1] > poly.vertices[np.argmin(xx**2+yy**2)]:
            xx = np.flip(xx)
            yy = np.flip(yy)
        idx_end = np.argmax(np.ediff1d(xx, to_end=(xx[0]-xx[-1]))**2+np.ediff1d(yy, to_end=(yy[0]-yy[-1]))**2)
        xx = np.concatenate([xx[idx_end+1:], xx[:idx_end+1]])
        yy = np.concatenate([yy[idx_end+1:], yy[:idx_end+1]])

    mid = int(len(xx)/2)
    x_first, y_first = [xx[mid]], [yy[mid]]
    x_last, y_last = [xx[mid]], [yy[mid]]

    for i in range(mid-1, -1, -1):
        if ((x_first[-1]-xx[i])**2 + (y_first[-1]-yy[i])**2) > 1:
            x_first.append(xx[i])
            y_first.append(yy[i])
    
    for i in range(mid+1, len(xx)):
        if ((x_last[-1]-xx[i])**2 + (y_last[-1]-yy[i])**2) > 1:
            x_last.append(xx[i])
            y_last.append(yy[i])

    x_new = np.concatenate((np.flip(x_first[1:]), x_last))
    y_new = np.concatenate((np.flip(y_first[1:]), y_last))
    
    return x_new, y_new


def identify_direction(xi, yi, ti, xj, yj, tj):
    Ti = ti[np.argmin(np.sqrt(xi**2+yi**2))]
    Tj = tj[np.argmin(np.sqrt(xj**2+yj**2))]

    if Ti <= Tj:
        condition_i = (ti>=(Ti-5))&(ti<=(Tj+5))
        condition_j = (tj>=(Ti-5))&(tj<=(Tj+5))
        xi, yi, ti = xi[condition_i], yi[condition_i], ti[condition_i]
        xj, yj, tj = xj[condition_j], yj[condition_j], tj[condition_j]
        idx_cft_i = np.argmin(np.sqrt(xi**2+yi**2))
        idx_cft_j = np.argmin(np.sqrt(xj**2+yj**2))
        PET = Tj - Ti
        ifirst = True
        if idx_cft_i==0:
            vec = np.array([xi[idx_cft_i:idx_cft_i+3], yi[idx_cft_i:idx_cft_i+3]])
        elif idx_cft_i>=len(xi)-1:
            vec = np.array([xi[idx_cft_i-2:idx_cft_i+1], yi[idx_cft_i-2:idx_cft_i+1]])
        else:
            vec = np.array([xi[idx_cft_i-1:idx_cft_i+2], yi[idx_cft_i-1:idx_cft_i+2]])
        if idx_cft_j<1:
            p_after = np.array([xj[idx_cft_j+1], yj[idx_cft_j+1]])
            direction = (p_after[0]-vec[0,2])*(vec[1,0]-vec[1,2])-(p_after[1]-vec[1,2])*(vec[0,0]-vec[0,2])
        else:
            p_before = np.array([xj[idx_cft_j-1], yj[idx_cft_j-1]])
            direction = (p_before[0]-vec[0,0])*(vec[1,2]-vec[1,0])-(p_before[1]-vec[1,0])*(vec[0,2]-vec[0,0])
    elif Ti > Tj:
        condition_i = (ti>=(Tj-5))&(ti<=(Ti+5))
        condition_j = (tj>=(Tj-5))&(tj<=(Ti+5))
        xi, yi, ti = xi[condition_i], yi[condition_i], ti[condition_i]
        xj, yj, tj = xj[condition_j], yj[condition_j], tj[condition_j]
        idx_cft_i = np.argmin(np.sqrt(xi**2+yi**2))
        idx_cft_j = np.argmin(np.sqrt(xj**2+yj**2))
        PET = Ti - Tj
        ifirst = False
        if idx_cft_j==0:
            vec = np.array([xj[idx_cft_j:idx_cft_j+3], yj[idx_cft_j:idx_cft_j+3]])
        elif idx_cft_j>=len(xj)-1:
            vec = np.array([xj[idx_cft_j-2:idx_cft_j+1], yj[idx_cft_j-2:idx_cft_j+1]])
        else:
            vec = np.array([xj[idx_cft_j-1:idx_cft_j+2], yj[idx_cft_j-1:idx_cft_j+2]])
        if idx_cft_i<1:
            p_after = np.array([xi[idx_cft_i+1], yi[idx_cft_i+1]])
            direction = (p_after[0]-vec[0,2])*(vec[1,0]-vec[1,2])-(p_after[1]-vec[1,2])*(vec[0,0]-vec[0,2])
        else:
            p_before = np.array([xi[idx_cft_i-1], yi[idx_cft_i-1]])
            direction = (p_before[0]-vec[0,0])*(vec[1,2]-vec[1,0])-(p_before[1]-vec[1,0])*(vec[0,2]-vec[0,0])

    xi, yi = simplify_trajectory(xi, yi)
    xj, yj = simplify_trajectory(xj, yj)

    return xi, yi, ti, xj, yj, tj, direction, PET, ifirst


def visualize(filename, root, other_road_users=True, direction=True):
    dt = zarr.open(root+filename, mode='r')

    slices = dt.index[:]
    timestep = dt.timestep[:]
    motion = dt.motion[:]
    types = dt.category[:]
    typelist = ['human-driven vehicles' if t==0 else 'pedestrians' if t==1 else 'motorcyclists' if t==2 else 'cyclists' if t==3 else 'buses' if t==4 else 'autonomous vehicles' if t==10 else 'static backrgound' for t in types]
    maps = dt.lane[:]

    fig, ax = plt.subplots(figsize=(7,7))

    for map in maps:
        ax.plot(map[:,0], map[:,1], c='silver', lw=1, zorder=-1)

    xi, yi, ti = motion[slices[0]:slices[1],0], motion[slices[0]:slices[1],1], timestep[slices[0]:slices[1]]
    xj, yj, tj = motion[slices[1]:slices[2],0], motion[slices[1]:slices[2],1], timestep[slices[1]:slices[2]]
    ax.scatter(xi, yi, c=ti, cmap='viridis', s=1, vmin=0, vmax=11)
    ax.scatter(xj, yj, c=tj, cmap='viridis', s=1, vmin=0, vmax=11)

    if direction:
        xi, yi, ti, xj, yj, tj, _, PET, _ = identify_direction(xi, yi, ti, xj, yj, tj)

        ax.arrow(xi[0], yi[0], xi[1]-xi[0], yi[1]-yi[0], color='r', label=typelist[0], head_width=3, head_length=3)
        ax.arrow(xi[-2], yi[-2], xi[-1]-xi[-2], yi[-1]-yi[-2], color='r', head_width=3, head_length=3)

        ax.arrow(xj[0], yj[0], xj[1]-xj[0], yj[1]-yj[0], color='b', label=typelist[1], head_width=3, head_length=3)
        ax.arrow(xj[-2], yj[-2], xj[-1]-xj[-2], yj[-1]-yj[-2], color='b', head_width=3, head_length=3)
    
    if len(types)>2 and other_road_users:
        for i in range(2,len(types)):
            start, end = slices[i], slices[i+1]
            x = motion[start:end,0]
            y = motion[start:end,1]    
            if types[i] == -1:
                ax.plot(x, y, c='red', zorder=-1, label=typelist[i])
            if types[i] == 10:
                ax.plot(x, y, c='blue', zorder=-1, label='AV')
            if types[i] in [0,1,2,3,4]:
                ax.plot(x, y, c='m', zorder=-1, label='other road users')
    
    ax.set_title('PET:' + str(np.round(PET,1)) + 's')

    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels[2:], handles[2:]))
    handles = handles[:2]
    labels = labels[:2]
    handles.extend(by_label.values())
    labels.extend(by_label.keys())
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25))

    return fig, ax