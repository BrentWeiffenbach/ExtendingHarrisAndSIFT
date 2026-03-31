import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

pcd = o3d.io.read_point_cloud("data/real/bunny.ply")
pts = np.asarray(pcd.points)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2], s=0.5, c=pts[::5, 2], cmap='viridis') # type: ignore
ax.set_title("Stanford Bunny")
plt.tight_layout()
plt.show()