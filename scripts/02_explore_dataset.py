"""Phase 2: Browse and inspect a LeRobot dataset from the Hugging Face Hub.

Dataset: lerobot/aloha_sim_transfer_cube_human
  - Bimanual ALOHA (two 6-DoF arms + grippers => 14-dim action) teleoperated
    in MuJoCo to pick a cube with one arm and transfer it to the other.
  - Human-collected demonstrations => this is IMITATION LEARNING data.
"""
import numpy as np

# Import path moved across versions; try the v0.5.x layout first.
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

print(f"Loading {REPO_ID} (downloads on first run, then cached) ...")
# video_backend="pyav" uses PyAV's bundled FFmpeg libs -> no system ffmpeg / sudo needed.
ds = LeRobotDataset(REPO_ID, video_backend="pyav")
meta = ds.meta

print("\n" + "=" * 60)
print("DATASET-LEVEL METADATA")
print("=" * 60)
print(f"repo_id          : {REPO_ID}")
print(f"robot_type       : {getattr(meta, 'robot_type', 'n/a')}")
print(f"fps              : {meta.fps}")
print(f"total_episodes   : {meta.total_episodes}")
print(f"total_frames     : {meta.total_frames}")
print(f"frames/episode   : {meta.total_frames / max(meta.total_episodes,1):.1f} (avg)")
print(f"camera keys      : {getattr(meta, 'camera_keys', 'n/a')}")

print("\n" + "=" * 60)
print("FEATURES (the LeRobotDataset 'schema')")
print("=" * 60)
for key, feat in meta.features.items():
    shape = feat.get("shape")
    dtype = feat.get("dtype")
    print(f"  {key:30s} dtype={str(dtype):10s} shape={shape}")

print("\n" + "=" * 60)
print("A SINGLE FRAME (ds[0]) -> what the policy actually consumes")
print("=" * 60)
sample = ds[0]
for key, val in sample.items():
    if hasattr(val, "shape"):
        print(f"  {key:35s} {tuple(val.shape)}  {val.dtype}")
    else:
        print(f"  {key:35s} {val}")

# Action / state dimensionality
act = sample["action"]
state = sample.get("observation.state")
print("\n" + "=" * 60)
print("ACTION & STATE SPACES")
print("=" * 60)
print(f"action dim       : {tuple(act.shape)}  -> 14 = [L arm 6 joints + L gripper] + [R arm 6 + R gripper]")
if state is not None:
    print(f"obs.state dim    : {tuple(state.shape)}  (proprioception: current joint positions)")

# Episode-0 structure: pull its frame range and show action trajectory stats.
print("\n" + "=" * 60)
print("EPISODE 0 STRUCTURE")
print("=" * 60)
# v0.5.x: episode frame ranges live in meta.episodes (a HF Dataset).
ep0 = ds.meta.episodes[0]
ep0_from = ep0["dataset_from_index"]
ep0_to = ep0["dataset_to_index"]
print(f"episode 0 spans frames [{ep0_from}, {ep0_to})  => {ep0_to - ep0_from} timesteps")
print(f"at {meta.fps} fps that is ~{(ep0_to - ep0_from)/meta.fps:.1f} seconds of robot motion")
actions = np.stack([ds[i]["action"].numpy() for i in range(ep0_from, ep0_to)])
print(f"action trajectory array shape : {actions.shape}  (timesteps, action_dim)")
print(f"per-joint action range (min..max across episode):")
for j in range(actions.shape[1]):
    print(f"    joint {j:2d}: {actions[:,j].min():+.3f} .. {actions[:,j].max():+.3f}")

print("\nDONE.")
