# usb_c_insertion

ROS1 Noetic package for force-guided USB-C cable insertion with a UR5e.

## Main interfaces
- Motion command: `/twist_controller/command`
- Force/torque: `/wrench`
- Tool pose: TF from `base_link` to `tool0_controller`

## Architecture
- `robot_interface.py`: motion command abstraction
- `ft_interface.py`: wrench subscription, filtering, zeroing
- `tf_interface.py`: TF lookup utilities
- `contact_detector.py`: contact detection logic
- `wall_probe.py`: move-until-contact probing
- `wall_frame_estimator.py`: estimate wall orientation from contact points
- `search_pattern.py`: wall-aligned search pattern generation
- `insertion_state_machine.py`: orchestration logic

## Development principles
- Python only
- English comments only
- Conservative robot safety
- Keep files small and focused
