"""Phase 4b: Offline inference check -- run the trained ACT policy on held-out
frames from the dataset and compare PREDICTED actions vs GROUND-TRUTH (expert)
actions. This needs no simulator/rendering, so it always works and is a good
sanity check that the checkpoint loads and produces sane outputs.

NOTE: this measures open-loop ACTION ERROR (how close the policy's action is to
the expert's on states the expert actually visited). It is NOT the same as task
success -- a closed-loop rollout (04_eval_sim.sh) can still fail even when this
error is small, because at deployment the policy visits its OWN states.
"""
import sys
import numpy as np
import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/train/act_aloha_cube/checkpoints/last/pretrained_model"
REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

print(f"Loading policy from {CKPT} ...")
policy = ACTPolicy.from_pretrained(CKPT)
policy.eval()
device = "cpu"
policy.to(device)

ds = LeRobotDataset(REPO_ID, video_backend="pyav")

# Evaluate on the LAST episode (held-out-ish: tiny training run barely saw it).
ep = ds.meta.episodes[ds.meta.total_episodes - 1]
lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
print(f"Evaluating open-loop on episode {ds.meta.total_episodes - 1}: frames [{lo},{hi}) ({hi-lo} steps)")

errs = []
policy.reset()
N = min(120, hi - lo)  # cap for speed on CPU
with torch.no_grad():
    for i in range(lo, lo + N):
        frame = ds[i]
        batch = {}
        for k, v in frame.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)
        # provide the language task string the policy expects
        batch["task"] = [frame.get("task", "")]
        pred = policy.select_action(batch).squeeze(0).cpu().numpy()
        gt = frame["action"].numpy()
        errs.append(np.abs(pred - gt))

errs = np.stack(errs)  # (N, 14)
print("\n" + "=" * 56)
print(f"OPEN-LOOP ACTION ERROR over {N} steps (|pred - expert|)")
print("=" * 56)
print(f"  mean L1 error (all joints) : {errs.mean():.4f} rad")
print(f"  median L1 error            : {np.median(errs):.4f} rad")
print(f"  worst-joint mean error     : {errs.mean(axis=0).max():.4f} rad")
print(f"  per-joint mean L1 error:")
for j in range(errs.shape[1]):
    bar = "#" * int(errs[:, j].mean() / max(errs.mean(axis=0).max(), 1e-6) * 30)
    print(f"    joint {j:2d}: {errs[:, j].mean():.4f}  {bar}")
print("\n(With only a few hundred CPU training steps, expect LARGE error -- the")
print(" point is the pipeline runs end to end. A full run drives this down.)")
