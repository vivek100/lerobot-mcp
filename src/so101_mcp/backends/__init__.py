from so101_mcp.backends.base import BackendMode, JointLimits, RobotBackend, TaskSignals


def make_backend(kind: BackendMode, **kwargs) -> RobotBackend:
    if kind == "sim":
        from so101_mcp.backends.sim import SimBackend

        return SimBackend(**kwargs)
    if kind == "real":
        from so101_mcp.backends.real import RealBackend

        return RealBackend(**kwargs)
    raise ValueError(f"Unknown backend {kind!r}")


__all__ = ["BackendMode", "JointLimits", "RobotBackend", "TaskSignals", "make_backend"]
