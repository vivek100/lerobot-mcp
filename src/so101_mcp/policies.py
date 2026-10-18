"""Pretrained LeRobot policies: a small registry, loading on CPU, and chunk inference.

A policy maps one observation (joint state + camera frames) to a chunk of future joint targets in the
same units the runtime uses (degrees, gripper 0-100). ``camera_map`` maps the policy's image keys
(``observation.images.<key>``) to this backend's camera names.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

STATE_KEY = "observation.state"
IMAGE_PREFIX = "observation.images."


@dataclass(frozen=True)
class PolicySpec:
    name: str
    repo: str
    revision: str
    camera_map: dict[str, str]  # policy image key -> our camera name
    description: str
    real_ok: bool = False  # trained on (or validated for) this real arm and its cameras


REGISTRY: dict[str, PolicySpec] = {
    spec.name: spec
    for spec in [
        PolicySpec(
            "act_cube_sim_chaenn",
            "Chaenn/act_policy_so101_cube_multitask_sim_0827",
            "96e4b70ba0c4b2009f1a1b1eaf1deb1ff3e35510",
            {"side": "front", "wrist": "wrist"},
            "ACT (ResNet-18, 52M params, 100-step chunks) trained on a different SO-101 MuJoCo cube "
            "pick-and-place sim (Chaenn/so101_cube_sim_place_0827). Foreign scene: proves the pipeline, "
            "not expected to solve our task.",
        ),
    ]
}


def download(repo: str, revision: str, attempts: int = 6) -> str:
    """snapshot_download with retries (this network drops Hub connections now and then)."""
    from huggingface_hub import snapshot_download

    for attempt in range(attempts):
        try:
            return snapshot_download(repo, revision=revision, max_workers=1)
        except Exception:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            time.sleep(2 + 2 * attempt)
    raise AssertionError("unreachable")


@dataclass
class LoadedPolicy:
    spec: PolicySpec
    device: str
    policy: object
    preprocessor: object
    postprocessor: object
    image_keys: list[str]
    image_shapes: dict[str, tuple[int, int, int]]
    chunk_size: int
    load_seconds: float
    latencies_ms: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def load(cls, spec: PolicySpec, device: str = "cpu") -> LoadedPolicy:
        import torch  # noqa: F401
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.configs.policies import PreTrainedConfig

        t0 = time.perf_counter()
        path = download(spec.repo, spec.revision)
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        if hasattr(config, "pretrained_backbone_weights"):
            config.pretrained_backbone_weights = None  # the checkpoint carries the trained backbone
        policy = get_policy_class(config.type).from_pretrained(path, config=config)
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        image_keys = [k.removeprefix(IMAGE_PREFIX) for k in config.input_features if k.startswith(IMAGE_PREFIX)]
        missing = [k for k in image_keys if k not in spec.camera_map]
        if missing:
            raise ValueError(f"camera_map for {spec.name} lacks policy cameras {missing}")
        shapes = {k: tuple(config.input_features[IMAGE_PREFIX + k].shape) for k in image_keys}
        chunk = int(getattr(config, "n_action_steps", None) or getattr(config, "chunk_size", 1))
        return cls(spec, device, policy, preprocessor, postprocessor, image_keys, shapes, chunk,
                   time.perf_counter() - t0)

    def predict_chunk(self, state: np.ndarray, frames: dict[str, np.ndarray]) -> np.ndarray:
        """One observation -> (chunk, 6) joint targets. ``frames`` are keyed by policy camera key, HxWx3 uint8."""
        import torch

        batch = {STATE_KEY: torch.from_numpy(np.asarray(state, dtype=np.float32))[None]}
        for key in self.image_keys:
            frame = frames[key]
            c, h, w = self.image_shapes[key]
            if frame.shape[:2] != (h, w):
                import cv2

                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            batch[IMAGE_PREFIX + key] = torch.from_numpy(frame).permute(2, 0, 1)[None].float() / 255.0
        with self.lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                batch = self.preprocessor(batch)
                chunk = self.policy.predict_action_chunk(batch)
                chunk = self.postprocessor(chunk)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)
            del self.latencies_ms[:-200]
        return chunk[0].detach().cpu().numpy().astype(np.float64)

    def camera_for(self, key: str) -> str:
        return self.spec.camera_map[key]

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        values = np.array(self.latencies_ms)
        return {"p50_ms": round(float(np.percentile(values, 50)), 1), "p95_ms": round(float(np.percentile(values, 95)), 1),
                "count": float(len(values))}
