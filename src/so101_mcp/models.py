"""Pydantic models for MCP tool inputs and outputs (explicit, shallow, strictly typed)."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", bool(self.model_config.get("exclude_none", False)))
        return super().model_dump(**kwargs)


CameraName = Literal["front", "wrist"]
GestureName = Literal[
    "nod", "shake", "wave", "tada", "bow", "shrug", "point", "look_around", "dance", "excited"
]
ExpressState = Literal["idle", "listening", "thinking", "happy", "sad"]
Mood = Literal["neutral", "happy", "excited", "sad", "curious"]
PlayType = Literal["gesture", "state", "talk"]


class ObserveRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    cameras: Optional[List[CameraName]] = Field(
        None, description="Physical cameras to capture. Omit for every camera the server has (front + wrist)."
    )
    image_width: Optional[Literal[320, 640]] = Field(None, description="Physical camera width in pixels (default 640).")
    previews: Optional[bool] = Field(
        None, description="Include three MuJoCo views of the current joints (front/side/top). Default true."
    )


class MoveRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    forward_cm: Optional[float] = Field(
        None, ge=-20, le=20,
        description="Centimetres away from the base along the table. Negative = back toward the base.",
    )
    left_cm: Optional[float] = Field(
        None, ge=-20, le=20,
        description="Centimetres toward the arm's left. Negative = toward the arm's right.",
    )
    up_cm: Optional[float] = Field(
        None, ge=-20, le=20,
        description="Centimetres up. Negative = down toward the table.",
    )
    tilt_down_deg: Optional[float] = Field(
        None, ge=-45, le=45,
        description="Degrees to tip the fingers more downward. Negative = more level / upward.",
    )
    roll_deg: Optional[float] = Field(
        None, ge=-90, le=90,
        description="Degrees to twist the jaws counter-clockwise looking along the fingers. Negative = clockwise.",
    )


class DetectRequest(StrictModel):
    target: Literal["color", "face"] = Field(..., description="Largest color blob or largest face, reselected each frame.")
    color: Optional[Literal["red", "yellow", "green", "blue"]] = Field(None, description="Required when target=color.")
    camera: CameraName = Field("wrist", description="Image to inspect. No motion is performed.")
    plane_height_cm: Optional[float] = Field(None, ge=0, le=50, description="Optional assumed height of the detected centre above table. With a calibrated camera, projects centre to that plane and returns robot-space offsets. Not measured depth. Sim cubes are 3 cm tall; centre plane 1.5 cm. Real cameras currently uncalibrated.")


class TrackRequest(DetectRequest):
    seconds: float = Field(10, ge=5, le=60, description="Bounded duration including wrist calibration probe; call waits until finished.")
    invert: bool = Field(False, description="Reverse fixed front-camera pan direction. No effect in wrist servo mode.")


class GripperRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    openness: float = Field(..., ge=0, le=100, description="0 = fully closed, 100 = fully open. Does not move the arm.")


class PlayRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    type: PlayType = Field(..., description="gesture: one-shot. state: hold until stop or the next motion. talk: speak.")
    name: Optional[str] = Field(None, description="Gesture or state name (see list_animations). Required for gesture and state.")
    repeat: Optional[int] = Field(None, ge=1, le=5, description="gesture: times to play (default 1)")
    seconds: Optional[float] = Field(None, gt=1, le=60, description="state: how long to hold (default 8)")
    text: Optional[str] = Field(None, min_length=1, max_length=1500, description="talk: what to say")
    mood: Optional[Mood] = Field(None, description="talk: body language (default neutral)")
    voice: Optional[bool] = Field(None, description="talk: speak aloud (default true)")


class JointValues(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    shoulder_pan: float = Field(..., description="degrees")
    shoulder_lift: float = Field(..., description="degrees")
    elbow_flex: float = Field(..., description="degrees")
    wrist_flex: float = Field(..., description="degrees")
    wrist_roll: float = Field(..., description="degrees")
    gripper: float = Field(..., description="0 closed - 100 open")


class JointLimit(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    joint: str = Field(..., description="Joint name")
    lower: float = Field(..., description="Soft lower limit")
    upper: float = Field(..., description="Soft upper limit")


class JointDetail(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    joint: str = Field(..., description="Joint name")
    position_deg: float = Field(..., description="Measured position")
    lower_deg: float = Field(..., description="Soft lower limit")
    upper_deg: float = Field(..., description="Soft upper limit")
    margin_deg: float = Field(..., description="Distance to the nearer limit")


class GripperPose(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    forward_cm: float = Field(..., description="Centimetres forward from the arm base")
    left_cm: float = Field(..., description="Centimetres to the arm's left (negative = right)")
    up_cm: float = Field(..., description="Centimetres above the table top")
    tilt_down_deg: float = Field(..., description="Finger angle: 0 level, 90 straight down, negative up")
    roll_deg: float = Field(..., description="Wrist-roll joint, degrees")
    gripper: float = Field(..., description="0 closed - 100 open")


class RoomDir(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    direction: Literal["forward", "back", "left", "right", "up", "down"] = Field(..., description="World direction from here")
    remaining_cm: float = Field(..., description="Approximate endpoint room on this axis at current orientation; does not guarantee the interpolated path or a combined move is feasible")
    limited_by: str = Field(..., description="What stops it: table, reach or joint limit")


class PoseSnapshot(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    result: Literal["ok", "moved", "refused", "failed", "stopped", "rested", "played", "holding", "tracked"] = Field(
        ..., description="refused = no motion started; failed/stopped = possible partial motion. holding = expressive state running."
    )
    message: str = Field(..., description="One-line summary for the agent")
    mode: Literal["sim", "real"] = Field(..., description="Which backend is driving the arm")
    requested: Optional[str] = Field(None, description="What this call asked for")
    achieved: Optional[str] = Field(None, description="What actually happened, in the same units")
    pose: GripperPose = Field(..., description="Gripper pose now")
    room_left: List[RoomDir] = Field(..., description="Remaining travel in each direction from here")
    joints: JointValues = Field(..., description="Measured joint angles")
    joint_details: List[JointDetail] = Field(..., description="Each joint vs its limits")
    table_clearance_cm: float = Field(..., description="Lowest point of the arm above the table, centimetres")
    warnings: List[str] = Field(..., description="Safety notes from this call")
    faults: List[str] = Field(..., description="Recent runtime faults, newest last")
    cameras: List[str] = Field(..., description="Names of the images that follow this JSON, in order")
    target: Optional[dict[str, Any]] = Field(None, description="Detection in normalized image coordinates, not world centimetres.")
    execution: Optional[dict[str, Any]] = Field(None, description="Motion job state, reached_target and tracking metrics.")


class AnimationInfo(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    name: str = Field(..., description="Name to pass as PlayRequest.name")
    kind: Literal["gesture", "state"] = Field(..., description="gesture: one-shot. state: held until stop or the next motion")
    description: str = Field(..., description="What it looks like")
    seconds: float = Field(..., description="Length of one play before fitting to the speed limits")


class AnimationList(StrictModel):
    model_config = ConfigDict(extra="forbid", exclude_none=True)

    animations: List[AnimationInfo] = Field(..., description="Gestures and held states")
    moods: List[str] = Field(..., description="Moods accepted by type=talk")
    how_to_play: str = Field(..., description="How to call run_animation for each kind")
