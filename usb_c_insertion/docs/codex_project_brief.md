# Project brief: USB-C cable insertion with UR5e

## Goal
Automate insertion of a USB-C cable into a USB-C PCIe expansion card mounted in a PC mainboard.

## Stack
- ROS1 Noetic
- Python
- UR5e robot
- Cartesian velocity control via `/twist_controller/command`
- Force/torque input via `/wrench`
- Tool pose from TF: `base_link -> tool0`
- English comments only

## Available ROS interfaces
- Motion command: `/twist_controller/command`
- Force/torque: `/wrench`
- Optional FT zeroing service: `/ur_hardware_interface/zero_ftsensor`
- Pose must be obtained via TF between `base_link` and `tool0`

## Current assumptions
- Initial coarse port pose is provided in `base_link`
- Camera localization is not accurate enough for direct open-loop insertion
- The insertion pipeline is force-guided and multi-stage

## Planned pipeline
1. Move to a pre-pose based on the estimated port pose
2. Probe the housing wall at point 1
3. Retract
4. Probe the wall at point 2 with a y-offset
5. Estimate wall orientation
6. Correct tool yaw around z
7. Move near estimated port position
8. Approach wall until contact
9. Execute a small wall-aligned search pattern
10. Detect insertion from force and motion cues
11. Stop safely on success or failure

## Coordinate convention
For the PC case frame:
- x: along the long side of the case / along the port row
- z: upward against gravity when the case is upright
- y: right-handed completion

## Coding preferences
- Keep code modular
- One responsibility per file
- Conservative robot safety
- Explicit state transitions
- No giant monolithic scripts
- Add clear logging
- Add small unit tests where possible

## Safety requirements
- Low probing speed
- Hard timeout per phase
- Stop on excessive force
- Stop on stale wrench or missing TF
- Always provide a stop_motion() path
