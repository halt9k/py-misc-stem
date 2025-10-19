import numpy as np
import pandas as pd

from plots.mpl_3d_volume import plot_3d_volume


def f_solution_proximity_brute(a, b, c):  
    subs = [a, b, c]
    # all possible arrangements of len 5
    x = np.stack(np.meshgrid(*([subs, ] * 6))).reshape(6, -1)

    # * : a | b | c
    # * + a + b = 17
    # * + * + b = 19
    # * + * - c = 10
    # a + a + * = 12
    # known (a, b, c) solutions for * in [0-9]: (5,7,2), (3, 8, 6)
    
    s1 = x[0] + a + b - 17
    s2 = x[1] + x[2] + b - 19
    s3 = x[3] + x[4] - c - 10
    s4 = a + a + x[5] - 12
    
    B = np.array([s1, s2, s3, s4]).T    
    r = np.linalg.norm(B, axis=1)
        
    return r.min()
    # return ((a+2)**2 + (b-5) **2 + (c-0)**2) ** 0.5


lx, ly, lz = np.linspace(-10, 10, 21, dtype=np.int32), np.linspace(-10, 10, 21, dtype=np.int32), np.linspace(-10, 10, 21, dtype=np.int32)
# lx, ly, lz = np.linspace(-10, 10, 2), np.linspace(-10, 10, 2), np.linspace(-10, 10, 2)
pts_x, pts_y, pts_z = np.array(np.meshgrid(lx, ly, lz)).reshape(3, -1)

# f_solution_proximity_brute(3, 4, 5)
f_vals = np.vectorize(f_solution_proximity_brute)(pts_x, pts_y, pts_z)
# plot_3d_plotly(pts_x, pts_y, pts_z, f_vals)
plot_3d_volume(pts_x, pts_y, pts_z, f_vals)

df = pd.DataFrame({'a': pts_x, 'b': pts_y, 'c': pts_z, 'f': f_vals})
df_sorted = df.sort_values('f', ascending=True).reset_index(drop=True)

print("Top 10 closest solutions:")
print(df_sorted.head(10))
