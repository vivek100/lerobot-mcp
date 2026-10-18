"""One reference shared by server instructions, the guide tool and MCP prompt."""

ROBOT_GUIDE = """SO-101 agent operating guide

COORDINATES
The axes are fixed to the robot base, never to the viewer, camera or current wrist heading.
+forward is model +X, along the table away from the base in its neutral reach direction.
+left is model +Y; right is negative left. +up is +Z, vertically away from the table.
Image-left is NOT necessarily robot-left. A wrist camera rotates with the arm; its image axes
can be rotated or reversed. A front camera facing the arm also reverses apparent directions.
Use the labelled +forward/+left/+up axes in previews and the numeric pose, not screen direction.
The previews are model estimates from joint readings, not reconstructed camera images.
Their wrist viewing cone uses the simulator camera mount and field of view; on hardware it
is illustrative until the physical camera mount/intrinsics are calibrated. Cone length is
only a drawing length, not depth, detection range, or obstacle clearance.

MOVEMENT
Call observe first. move accepts RELATIVE centimetres from the current gripper position.
up_cm=2 raises 2 cm; forward_cm=3 advances 3 cm; left_cm=-2 moves 2 cm right.
Omitted axes stay unchanged as planner goals; motion follows joint-space waypoints, not a
guaranteed straight Cartesian line. From folded rest the arm first unparks, then applies offsets.
tilt_down_deg is a relative finger tilt (positive points more down); roll_deg twists the jaws.
Per-call limits: each translation ±20 cm, tilt ±45 degrees, roll ±90 degrees. A large tilt
change needs multiple bounded calls. Animations can leave fingers pointed up; after a wave,
go_rest gives a known starting configuration before a manipulation task. Observe that pose first.
set_gripper openness is 0 closed, 100 open; it does not move the arm.
Motion calls block and already return a fresh observation. Do not redundantly poll observe.
Motion replies include physical cameras plus one side preview to keep context small. Request
observe(previews=true) for all three previews, or previews=false for physical cameras only.
room_left estimates directional room from reach/joint/table constraints, not object clearance.
up_cm in pose is height above the table; table_clearance_cm is the LOWEST arm-link height.
Travel above the table, align, then descend in small steps. Do not descend with under 1 cm
reported down room. Open before approach; close only around the object, lift and verify.
Command completion alone never proves grasp success. Stop freezes and holds; go_rest folds away.
refused means no motion was started; failed or stopped can mean partial motion: inspect achieved pose.

TARGET SELECTION AND TRACKING
Use detect_target before track: specify camera plus target=color and a supported color
(red/yellow/green/blue), or target=face. It returns an annotated photo and a normalized
bounding box [left, top, right, bottom], with (0,0) image top-left and (1,1) bottom-right.
Selection is the LARGEST matching blob or face each frame. It can switch between similar
objects or people. It is not arbitrary-object recognition, instance locking, or click-to-track.
The box is image geometry, not robot-space centimetres or a grasp pose.
For CALIBRATED cameras, detect_target optionally accepts plane_height_cm: an assumed height of
the visible centre above the table. It then returns projected robot-space position and offsets.
Sim cameras are calibrated; sim cubes are 3 cm tall, so 1.5 cm is a useful centre-plane assumption.
Physical cameras are NOT calibrated yet, so no world coordinates are inferred for them.
Projection is not measured depth or a guaranteed grasp pose. Hover above its X/Y position first,
then inspect the wrist image and descend in small steps. Do not blindly apply the full downward offset.
Color score=1 is a threshold match, not a calibrated probability. Face score is detector output.
track uses the SAME selection rule and returns execution state and metrics, then fresh images.
Wrist mode nudges pan and wrist about 5 degrees to learn image axes before centering the target.
Aim the wrist camera at the target FIRST; the probe also moves the arm. No blind searching.
Fixed front mode is only approximate horizontal pointing using an assumed camera arrangement;
it is not calibrated 3D following. invert reverses its pan direction when needed.
Tracking does not approach, grasp or infer depth. Occlusion beyond 3 seconds stops the job.
Inspect target visibility, final image error and reached_target; completed does not imply centered.
Use stop from another call to cancel a running track. Do not start a second motion concurrently.

LIVE CAMERA VIEW
show_wrist_camera opens a read-only MCP App in compatible hosts, with the local demo web page
as fallback. The user can watch while motion/tracking runs. This viewer does not start motion
or continuously send images to the model. Continue to use observe/detect_target for decisions.
"""
