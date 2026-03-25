import os

# Root folder
root = "3D_Keypoint_Detectors"

# Folder structure
folders = [
    "data/synthetic",
    "data/real",
    "src/common",
    "src/voxel",
    "src/pointcloud",
    "src/evaluation",
    "notebooks",
    "reports/figures"
]

# Placeholder files to create
files = [
    "README.md",
    "requirements.txt",
    "run_all.py",
    "src/common/__init__.py",
    "src/common/base_detector.py",
    "src/common/visualization.py",
    "src/common/io.py",
    "src/common/metrics.py",
    "src/voxel/harris3d.py",
    "src/voxel/sift3d.py",
    "src/voxel/params.py",
    "src/pointcloud/harris_pc.py",
    "src/pointcloud/sift_pc.py",
    "src/pointcloud/params.py",
    "src/evaluation/evaluate_voxel.py",
    "src/evaluation/evaluate_pc.py",
    "src/evaluation/plots.py",
    "notebooks/demo_voxel.ipynb",
    "notebooks/demo_pc.ipynb"
]

# Create folders
for folder in folders:
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)

# Create placeholder files
for file in files:
    path = os.path.join(root, file)
    # Create empty file if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# Placeholder for {file}\n")

print(f"Folder structure and placeholder files created under '{root}'")
