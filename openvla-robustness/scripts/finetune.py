"""Phase 3 (OPTIONAL): LoRA fine-tune OpenVLA-7B on LIBERO-Object.

A thin launcher around OpenVLA's official ``vla-scripts/finetune.py``. It (1) pulls
the modified LIBERO RLDS dataset to the VM, then (2) runs the LoRA trainer via
torchrun. Per Claude.md: rank 32, batch 8, grad-accum 2, checkpoint every 500 steps.

    colab exec -s openvla-session -f scripts/finetune.py -- --max_steps 2000

SKIP this phase if GPU time is tight — a thorough robustness study on the official
pretrained checkpoint is worth more than a half-trained custom one. Needs a 40GB+
GPU (A100); LoRA rank 32 fine-tunes ~0.1% of the 7B weights, but activations +
the frozen base still dominate VRAM.

--- What LoRA is doing here (for the report / your own notes) ---
Full fine-tuning updates all 7B weights. LoRA freezes them and learns a low-rank
*correction* per target weight matrix: W_eff = W_frozen + (B @ A) * (alpha/rank),
where for a d-by-k matrix W, A is rank-by-k and B is d-by-rank. With rank=32 those
two skinny matrices have ~32*(d+k) params vs d*k for W — a few hundredths of a
percent. The bet (and the empirical finding behind LoRA) is that the *adaptation*
needed to move from "OpenVLA's pretraining mix" to "LIBERO-Object" lives in a
low-dimensional subspace, so a rank-32 bottleneck captures it. It works for domain
adaptation precisely because we are nudging an already-capable policy onto a new
data distribution, not teaching manipulation from scratch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

OPENVLA_DIR = os.environ.get("OPENVLA_DIR", "/content/openvla")
DEFAULT_DATA_DIR = os.environ.get("LIBERO_RLDS_DIR", "/content/modified_libero_rlds")
HF_DATASET = "openvla/modified_libero_rlds"


def download_dataset(data_dir: str):
    """Snapshot the modified LIBERO RLDS dataset to ``data_dir`` (idempotent)."""
    if os.path.isdir(data_dir) and os.listdir(data_dir):
        print(f"[skip] dataset already present at {data_dir}", flush=True)
        return
    print(f"[data] downloading {HF_DATASET} -> {data_dir}", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_DATASET, repo_type="dataset",
                      local_dir=data_dir, local_dir_use_symlinks=False)


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune OpenVLA-7B on LIBERO")
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--dataset_name", default="libero_object_no_noops")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--run_root_dir", default="checkpoints/lora")
    ap.add_argument("--adapter_tmp_dir", default="adapter_tmp")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--nproc_per_node", type=int, default=1)
    args = ap.parse_args()

    download_dataset(args.data_dir)

    finetune_script = os.path.join(OPENVLA_DIR, "vla-scripts", "finetune.py")
    if not os.path.isfile(finetune_script):
        raise SystemExit(f"OpenVLA finetune script not found at {finetune_script}")

    cmd = [
        "torchrun", "--standalone", "--nnodes", "1",
        "--nproc-per-node", str(args.nproc_per_node), finetune_script,
        "--vla_path", args.vla_path,
        "--data_root_dir", args.data_dir,
        "--dataset_name", args.dataset_name,
        "--run_root_dir", args.run_root_dir,
        "--adapter_tmp_dir", args.adapter_tmp_dir,
        "--lora_rank", str(args.lora_rank),
        "--batch_size", str(args.batch_size),
        "--grad_accumulation_steps", str(args.grad_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--max_steps", str(args.max_steps),
        "--save_steps", str(args.save_steps),
        "--image_aug", "True",
        "--use_lora", "True",
        "--wandb_project", "openvla-robustness",
        "--wandb_entity", "none",
    ]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=OPENVLA_DIR)
    print(f"\n[done] LoRA checkpoints under {args.run_root_dir}. Merge the adapter, "
          "then point run_baseline.py --checkpoint at the merged dir to evaluate.",
          flush=True)


if __name__ == "__main__":
    main()
