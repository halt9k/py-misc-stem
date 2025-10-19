import re

import matplotlib
import numpy as np

import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as pc

pio.renderers.default = 'browser'
# colorscale = pc.get_colorscale('Viridis')
# color_func = pc.sample_colorscale(colorscale, f_raw, low=0.0, high=1.0)

# def to_rgba(color_str, alpha):
#     if color_str.startswith('#'):
#         rgb = pc.hex_to_rgb(color_str)
#     elif color_str.startswith('rgb'):
#         nums = list(map(int, re.findall(r'\d+', color_str)))
#         rgb = tuple(nums[:3])
#     else:
#         rgb = (255, 255, 255)
#     return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'


def _to_rgba(color_str: str, alpha: float) -> str:
    if color_str.startswith('#'):
        r, g, b = pc.hex_to_rgb(color_str)
    elif color_str.startswith('rgb'):
        nums = list(map(int, re.findall(r'\d+', color_str)))
        r, g, b = nums[:3]
    else:
        r, g, b = (255, 255, 255)
    return f'rgba({r},{g},{b},{alpha})'

# def plot_3d_plotly(f_xyz, limits_xyz: tuple[tuple[float, float], 3]):
#     limits_x, limits_y, limits_z = limits_xyz
#     xm, ym, zm = np.meshgrid(limits_x, limits_y, limits_z, indexing='ij')
#     xm, ym, zm = xm.ravel(), ym.ravel(), zm.ravel()
#     
#     f_raw = np.array([f_xyz(a, b, c) for a, b, c in zip(xm, ym, zm)], dtype=float)
#     
#     mask = f_raw > 0.02
#     xm, ym, zm, f_raw = xm[mask], ym[mask], zm[mask], f_raw[mask]
#     
#     colors_rgba = [to_rgba(c, float(f)) for c, f in zip(color_func, f_raw)]
#     sizes = 5 + 30 * f_raw
#     
#     fig = go.Figure(data=[
#         go.Scatter3d(
#             x=xm, y=ym, z=zm,
#             mode='markers',
#             marker=dict(
#                 size=sizes,
#                 color=colors_rgba,
#                 showscale=True,
#                 colorbar=dict(title='f(a,b,c)')
#             )
#         )
#     ])
#     
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='a',
#             yaxis_title='b',
#             zaxis_title='c'
#         ),
#         title='Interactive 3D Map of f(a,b,c)\nColor + Transparency = f(a,b,c)',
#         width=900,
#         height=800
#     )
#     
#     fig.show()


def plot_3d_volume_plotly_1(xs, ys, zs, fs):
    limits_x = xs.min(), xs.max()
    limits_y = ys.min(), ys.max()
    limits_z = zs.min(), zs.max()

    mask = fs > 0.02
    xm = xs[mask]; ym = ys[mask]; zm = zs[mask]; fm = fs[mask]
    if fm.size == 0:
        raise ValueError("No points to plot (all f <= 0.02). Choose wider limits or lower threshold.")

    sizes = 5.0 + 30.0 * fm

    colorscale = pc.get_colorscale('Viridis')
    if fm.max() > fm.min():
        norm_f = (fm - fm.min()) / (fm.max() - fm.min())
    else:
        norm_f = np.zeros_like(fm)
    sampled = pc.sample_colorscale(colorscale, list(norm_f), low=0.0, high=1.0)
    rgba_colors = [_to_rgba(col, float(alpha)) for col, alpha in zip(sampled, fm)]

    main_trace = go.Scatter3d(
        x=xm, y=ym, z=zm,
        mode='markers',
        marker=dict(size=sizes, color=rgba_colors, showscale=False),
        hovertemplate='a=%{x}<br>b=%{y}<br>c=%{z}<br>f=%{marker.color}<extra></extra>'
    )

    colorbar_trace = go.Scatter3d(
        x=xm, y=ym, z=zm,
        mode='markers',
        marker=dict(size=2, color=fm, colorscale='Viridis', showscale=True, colorbar=dict(title='f(x,y,z)')),
        hoverinfo='none',
        showlegend=False,
        opacity=0.0
    )

    fig = go.Figure(data=[main_trace, colorbar_trace])
    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        title='Interactive 3D map of f(x,y,z) (alpha ~= f)',
        width=900, height=800
    )
    
    fig.show()



def plot_3d_plotly_2(xs, ys, zs, fs) -> None:
    # other version of _1, unknown if better
    limits_x = xs.min(), xs.max()
    limits_y = ys.min(), ys.max()
    limits_z = zs.min(), zs.max()

    # Invert f_vals so that lower values (closer to 0) are more opaque
    # This matches the previous matplotlib plot and the user's likely intent
    # where lower error (f_vals closer to 0) is more interesting.
    # We normalize to [0, 1] for the alpha value.
    fs_inverted = 1.0 - (fs - fs.min()) / (fs.max() - fs.min())

    mask = fs_inverted > 0.02
    xm = xs[mask]; ym = ys[mask]; zm = zs[mask]; fm = fs_inverted[mask]
    f_raw_masked = fs[mask] # Keep original f_vals for color scale

    if fm.size == 0:
        raise ValueError("No points to plot (all f <= 0.02). Choose wider limits or lower threshold.")

    sizes = 5.0 + 30.0 * fm

    colorscale = pc.get_colorscale('Viridis')
    # Use the original f_vals for coloring, so that high f_raw means perfect match (yellow)
    if f_raw_masked.max() > f_raw_masked.min():
        norm_f_raw = (f_raw_masked - f_raw_masked.min()) / (f_raw_masked.max() - f_raw_masked.min())
    else:
        norm_f_raw = np.zeros_like(f_raw_masked)

    # Sample colorscale based on original f_raw values
    sampled_colors = pc.sample_colorscale(colorscale, list(norm_f_raw), low=0.0, high=1.0)

    # Apply inverted f_vals (fm) as alpha
    rgba_colors = [_to_rgba(col, float(alpha)) for col, alpha in zip(sampled_colors, fm)]

    main_trace = go.Scatter3d(
        x=xm, y=ym, z=zm,
        mode='markers',
        marker=dict(size=sizes, color=rgba_colors, showscale=False),
        hovertemplate='a=%{x}<br>b=%{y}<br>c=%{z}<br>f=%{marker.color}<extra></extra>'
    )

    # Add a separate trace for the colorbar using original f_vals
    colorbar_trace = go.Scatter3d(
        x=[None], y=[None], z=[None], # Dummy data to show colorbar
        mode='markers',
        marker=dict(size=2, color=f_raw_masked, colorscale='Viridis', showscale=True, colorbar=dict(title='f(a,b,c)')),
        hoverinfo='none',
        showlegend=False,
        opacity=0.0
    )

    fig = go.Figure(data=[main_trace, colorbar_trace])
    fig.update_layout(
        scene=dict(xaxis_title='a', yaxis_title='b', zaxis_title='c'),
        title='Interactive 3D map of f(a,b,c) (alpha ~= 1-normalized_error)',
        width=900, height=800
    )

    fig.show()