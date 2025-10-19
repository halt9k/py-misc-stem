import re

import matplotlib
import numpy as np

from matplotlib import pyplot as plt

matplotlib.use('qtagg')
# colorscale = pc.get_colorscale('Viridis')
# color_func = pc.sample_colorscale(colorscale, f_raw, low=0.0, high=1.0)


def plot_3d_volume_mpl(xs, ys, zs, fs):
    
    plot_idx = fs < 100.0
    # mask_high = fs > 2.0
    # high_idx = np.where(mask_high)[0]
    # low_idx = np.where(~mask_high)[0]
    # rng = np.random.default_rng(42)
    # sample_low = rng.choice(low_idx, size=min(1500, len(low_idx)), replace=False)
    # plot_idx = np.concatenate([high_idx, sample_low])
    
    xs_b = xs[plot_idx]
    ys_b = ys[plot_idx]
    zs_b = zs[plot_idx]
    fs_b = fs[plot_idx]
    # f_plot = np.clip(fs_b, 0.0, 1.0)
    f_plot = fs_b
    
    norm = plt.Normalize(vmin=fs_b.min(), vmax=fs_b.max())
    
    cmap = plt.get_cmap('turbo')
    rgba = cmap(norm(f_plot))
    # transparency proportional to f
    rgba[:, 3] = 1 - norm(f_plot)
    
    sizes = 40 * (1 - f_plot) ** 4
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(xs=xs_b, ys=ys_b, zs=zs_b, s=sizes, c=rgba, norm=norm, cmap=cmap, marker='o', linewidths=0) 
    ax.scatter3D(xs=xs_b, ys=ys_b, zs=zs_b, s=sizes, c=fs_b, norm=norm, cmap=cmap, marker='o', linewidths=0)
    # fig.colorbar(sm, ax=ax, shrink=0.6,  label='f(a,b,c)')
    
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    ax.set_title('3D map of f(a,b,c) ∈ [0,1]\nOpacity = f(a,b,c)')
    plt.show()


def plot_3d_volume(f_xyz):
    # draft, does not work properly yet
    import numpy as np
    from skimage import measure
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x, y, z = np.mgrid[-10:10:41j, -10:10:41j, -10:10:41j]
    vol = np.vectorize(f_xyz)(x, y, z)
    iso_val=8.0
    verts, faces, _, _ = measure.marching_cubes(vol, iso_val, spacing=(0.1, 0.1, 0.1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                    # vmin=(-10,-10,-10), vmax=(10,10,10),
                    cmap='Spectral', lw=3, zorder=100)

    # ax.set_xlim([x.min(), x.max()])
    # ax.set_ylim([y.min(), y.max()])
    # ax.set_zlim([z.min(), z.max()])

    plt.show()