"""
export_onnx_b0.py — Export B0 PPO actor to ONNX for browser inference.

Usage:
    python export_onnx_b0.py
    python export_onnx_b0.py --model models/b0_best --out models/b0_best.onnx

Output: models/b0_best.onnx
    Input  : obs    shape [1, 106]  float32
    Output : logits shape [1, 11]  float32  (9 dir + 2 kick, concatenated)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class ActorWrapper(nn.Module):
    """Wrap SB3 PPO policy to export actor logits only."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="models/b0_best",      help="SB3 model path (no .zip)")
    p.add_argument("--out",     default="models/b0_best.onnx")
    p.add_argument("--obs-dim", default=106, type=int)
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model, device="cpu")
    policy = model.policy
    policy.eval()

    wrapper = ActorWrapper(policy)
    wrapper.eval()

    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32)

    print(f"Exporting to: {args.out}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            opset_version=11,
            input_names=["obs"],
            output_names=["logits"],
            dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    print("Export done.")

    # ── Verify with onnxruntime ──────────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        test_obs = np.zeros((1, args.obs_dim), dtype=np.float32)
        logits = sess.run(["logits"], {"obs": test_obs})[0]
        print(f"Verify OK - logits shape: {logits.shape}")
        print(f"Sample logits: {logits[0].tolist()}")

        with torch.no_grad():
            pt_logits = wrapper(torch.tensor(test_obs)).numpy()
        max_diff = np.abs(logits - pt_logits).max()
        print(f"Max diff PyTorch vs ONNX: {max_diff:.2e}  {'OK' if max_diff < 1e-4 else 'WARNING'}")
    except ImportError:
        print("onnxruntime not installed - skipping verify (pip install onnxruntime)")


if __name__ == "__main__":
    main()
