import MDAnalysis as mda
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from scipy.interpolate import make_splprep, splev
from scipy.interpolate import make_lsq_spline


def get_backbone_by_pca(coords, spacing, group_space=15.0, k=3):
    """
    Extract smooth backbone from surfactant coordinates using PCA + 3D B-spline.

    coords: (N,3) array of surfactant atoms
    spacing: target spacing (nm) along backbone
    group_space: bin width along main axis for initial grouping
    k: spline order (3=cubic)
    """
    # --------------------------
    # 1) PCA for main axis
    # --------------------------
    pca = PCA(n_components=1)
    pca.fit(coords)
    axis = pca.components_[0]
    centroid = coords.mean(axis=0)

    d = np.linalg.norm(coords - centroid, axis=1)
    end_idx = np.argmax(d)
    end_point = coords[end_idx]

    axis = axis / np.linalg.norm(axis)

    # Ensure axis points away from the end
    if np.dot(coords[end_idx] - centroid, axis) > 0:
        axis = -axis

    # --------------------------
    # 2) Project coords to axis
    # --------------------------
    proj = np.dot(coords - end_point, axis)

    # --------------------------
    # 3) Bin along projection and compute group COMs
    # --------------------------
    s_min, s_max = proj.min(), proj.max()
    bins = np.arange(s_min, s_max + group_space, group_space)

    backbone = []
    for i in range(len(bins) - 1):
        mask = (proj >= bins[i]) & (proj < bins[i + 1])
        group = coords[mask]
        if len(group) == 0:
            continue
        backbone.append(group.mean(axis=0))

    backbone = np.array(backbone)
    if len(backbone) < 3:
        return None  # too few points

    # --------------------------
    # 4) Compute cumulative distance along backbone
    # --------------------------
    diffs = np.diff(backbone, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0], np.cumsum(seg_len)])
    total_len = s[-1]

    # normalize to u ∈ [0,1] for B-spline parameter
    u = s / total_len

    # --------------------------
    # 5) Build knot vector for B-spline
    # --------------------------
    n = len(backbone)
    n_internal = max(n // 2, 4)
    t_internal = np.linspace(0, 1, n_internal)
    t = np.r_[np.zeros(k), t_internal, np.ones(k)]  # clamped

    # --------------------------
    # 6) Fit 3D B-spline using make_lsq_spline
    # --------------------------
    x, y, z = backbone.T
    cs_x = make_lsq_spline(u, x, t, k)
    cs_y = make_lsq_spline(u, y, t, k)
    cs_z = make_lsq_spline(u, z, t, k)

    # --------------------------
    # 7) Sample uniformly along contour length
    # --------------------------
    n_uniform = max(int(total_len / spacing), 3)
    s_uniform = np.linspace(0, total_len, n_uniform)
    u_uniform = s_uniform / total_len

    backbone_smooth = np.vstack([
        cs_x(u_uniform),
        cs_y(u_uniform),
        cs_z(u_uniform)
    ]).T

    return backbone_smooth, s_uniform


def tangent_corr(path, s_uniform):
    # Compute unit tangent vectors

    #if smoothing_window > 1:
    #    from scipy.ndimage import uniform_filter1d
    #    path_smooth = uniform_filter1d(path, size=smoothing_window, axis=0)
    #else:
    #    path_smooth = path
    
    # Compute tangents from smoothed path
    #tang = np.diff(path_smooth, axis=0)
    tang = np.diff(path, axis=0)
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    #print(tang)
    n = len(tang)
    
    # Δs = 0 → correlation is exactly 1
    C0 = 1.0

    # Δs ≥ 1
    Ck = [
        np.mean(np.sum(tang[:-k] * tang[k:], axis=1))
        for k in range(1, n)
    ]
    
    C = np.concatenate([[C0], Ck])

    # Δs values
    ds = s_uniform[1] - s_uniform[0]
    svals = np.arange(0, n) * ds
    #svals=np.concatenate([[0], s_uniform[:-1]])

    return svals, C



def exp_decay(s, lp):
    return np.exp(-s / lp)


def lp_from_frame(s, C):
    """Fit persistence length for a single frame."""
    # only use positive distances
    mask = (C > 0)
    s_fit = s[mask]
    C_fit = C[mask]
    min_fit = int(0)
    max_fit = int(len(s_fit)*2/3)

    if len(s_fit) < 3:
        return np.nan

    try:
        popt, _ = curve_fit(exp_decay, s_fit[min_fit:max_fit], C_fit[min_fit:max_fit], p0=[20.0])
        return popt[0]
    except:
        return np.nan

##### ---- Main analysis loop ----- #####

#top = "SDS_400_NOPEG_SALT_1end_free_run100ns.gro"
#traj = "SDS_400_NOPEG_SALT_1end_free_cluster.xtc"
#sel = "name H23 or name H123"

#top = "SLE3S_240_NOPEG_SALT_1end_free_run100ns.gro"
#traj = "SLE3S_240_NOPEG_SALT_1end_free_cluster.xtc"
#sel = "name H23 or name H213"

top = "SCMT_300_NOPEG_SALT_1end_run100ns.gro"
traj = "SCMT_300_NOPEG_SALT_1end_free_cluster.xtc"
sel = "name H23 and not index 26631"
   
u = mda.Universe(top, traj)
micelle = u.select_atoms(sel)

all_s = []
all_C = []

lp_list = []

for ts in u.trajectory[:]:
    coords = micelle.positions.copy()

    backbone_out = get_backbone_by_pca(coords, spacing=15.0)
    if backbone_out is None:
        continue

    path, s_uniform = backbone_out

    s, C = tangent_corr(path, s_uniform)

    lp = lp_from_frame(s, C)
    lp_list.append(lp)
    all_C.append(C)



# Average correlations
min_len = min(len(c) for c in all_C)
C_array = np.array([c[:min_len] for c in all_C])
svals_common = s[:min_len]

lp_mean = np.mean(lp_list)
lp_std = np.std(lp_list)

C_avg = np.mean(C_array, axis=0)
C_std = np.std(C_array, axis=0)

mean_lp = lp_from_frame(svals_common, C_avg)

print("Per-frame persistence lengths:")
#print(lp_list)

print("\nFinal persistence length = %.3f ± %.3f nm" % (lp_mean, lp_std))
print("\nPersistence length from average = %.3f" % (mean_lp))


plt.figure()
plt.plot(svals_common, C_avg, 'o', label='Average data')
plt.fill_between(svals_common, C_avg - C_std, C_avg + C_std, color='gray', alpha=0.3, label='±1 std')
plt.plot(svals_common, exp_decay(svals_common, lp_mean), '--', label=f'Fit: l_p={lp_mean:.2f} Å')
plt.plot(svals_common, exp_decay(svals_common, mean_lp), '--', label=f'Fit: l_p={mean_lp:.2f} Å')

plt.xlabel('Δs (Å)',fontsize=16)
plt.ylabel('<t(s)·t(s+Δs)>',fontsize=16)
plt.title("SCMT persistence length",fontsize=18)
plt.legend(fontsize=14)
plt.tick_params(axis="both",labelsize=14)
plt.show()

#print(svals_common,C_avg)
