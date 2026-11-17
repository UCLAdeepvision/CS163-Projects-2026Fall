"""The closed-loop simulator.

scene.py    a benchmark scene: collision mesh, walkable map, camera placement
splat.py    renders a scene from the robot's camera (Gaussian splatting in plain PyTorch)
episode.py  the robot, its physics, the rules of one episode, and the score
runner.py   the loop that connects them to a policy: render, act, step the physics
"""
