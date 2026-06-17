"""Phase 1 verification: confirm the LeRobot + MuJoCo stack imports and runs."""
import importlib

print("=" * 50)
print("Phase 1: Environment Verification")
print("=" * 50)

import sys
print(f"Python      : {sys.version.split()[0]}")

import torch
print(f"torch       : {torch.__version__}  (CUDA avail: {torch.cuda.is_available()})")

import lerobot
print(f"lerobot     : {lerobot.__version__}")

import mujoco
print(f"MuJoCo      : {mujoco.__version__}")

import gymnasium
print(f"gymnasium   : {gymnasium.__version__}")

# Optional sim env (may not be installed yet)
try:
    import gym_aloha  # noqa: F401
    print("gym_aloha   : OK")
except Exception as e:
    print(f"gym_aloha   : NOT INSTALLED ({type(e).__name__})")

# Prove MuJoCo can actually build and step a model (headless, CPU).
print("-" * 50)
xml = """
<mujoco>
  <worldbody>
    <body pos='0 0 1'>
      <joint type='free'/>
      <geom type='box' size='.1 .1 .1' rgba='0 .9 0 1'/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
for _ in range(100):
    mujoco.mj_step(model, data)
print(f"MuJoCo sim step OK -> falling box z = {data.qpos[2]:.4f} (started at 1.0)")
print("=" * 50)
print("ALL CHECKS PASSED" )
