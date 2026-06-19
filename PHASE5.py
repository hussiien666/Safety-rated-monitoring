"""
Robotic Arm Path-Tracing Simulation
------------------------------------
Interactive PyBullet demo: the arm follows a user-selected geometric path
(square / rectangle / triangle / circle) while the simulation visualises the
planned path, the real end-effector trail, and the live tracking error.

Controls (panel on the right of the window):
    Shape          0=Square  1=Rectangle  2=Triangle  3=Circle
    Speed          how fast the arm moves along the path
    Size           overall size of the shape
    Pause / Run    freeze / resume the motion
    Clear Trail    erase the drawn trail
    Reset View     re-centre the camera
"""

import pybullet as p
import pybullet_data
import time
import math


p.connect(p.GUI, options="--width=1280 --height=720 --background_color_red=0.12 "
                         "--background_color_green=0.13 --background_color_blue=0.18")
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

plane = p.loadURDF("plane.urdf")

robot = p.loadURDF(
    r"C:\Users\original\OneDrive\Desktop\python codes grad\assembly trial\urdf\assembly trial.urdf",
    [0, 0, 0],
    useFixedBase=True
)
num_joints = p.getNumJoints(robot)
end_effector = num_joints - 1

movable_joints = [j for j in range(num_joints)
                  if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]

print("Number of joints:", num_joints)
for i in range(num_joints):
    print(i, p.getJointInfo(robot, i)[1].decode("utf-8"))
print("Movable joints:", movable_joints)



def get_ee_pos():
    return list(p.getLinkState(robot, end_effector)[4])


def move_joints_to(target_pos):
    joint_positions = p.calculateInverseKinematics(robot, end_effector, target_pos)
    for k, j in enumerate(movable_joints):
        if k >= len(joint_positions):
            break
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,
                                targetPosition=joint_positions[k], force=500)


def distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def reset_camera():
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=50,
                                 cameraPitch=-35, cameraTargetPosition=[cx, cy, cz])


def build_shape(shape_id, center, size):
    """Return a list of waypoints for the chosen shape, centred on `center`."""
    cx, cy, cz = center
    s = size
    if shape_id == 0:
        h = s / 2
        return [[cx - h, cy - h, cz], [cx + h, cy - h, cz],
                [cx + h, cy + h, cz], [cx - h, cy + h, cz]]
    if shape_id == 1:
        w, h = s * 0.75, s * 0.45
        return [[cx - w, cy - h, cz], [cx + w, cy - h, cz],
                [cx + w, cy + h, cz], [cx - w, cy + h, cz]]
    if shape_id == 2:
        h = s / 2
        return [[cx, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]]
    n = 36
    r = s / 2
    return [[cx + r * math.cos(2 * math.pi * k / n),
             cy + r * math.sin(2 * math.pi * k / n), cz] for k in range(n)]


SHAPE_NAMES = ["Square", "Rectangle", "Triangle", "Circle"]



for _ in range(300):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

ee = get_ee_pos()
cx, cy, cz = ee
print(f"End-effector starts at: x={cx:.3f} y={cy:.3f} z={cz:.3f}")
reset_camera()


slider_shape = p.addUserDebugParameter("Shape  0=Sq 1=Rec 2=Tri 3=Cir", 0, 3, 0)
slider_speed = p.addUserDebugParameter("Speed", 0.001, 0.03, 0.008)
slider_size  = p.addUserDebugParameter("Size", 0.08, 0.30, 0.16)
btn_pause    = p.addUserDebugParameter("Pause / Run", 1, 0, 1)
btn_clear    = p.addUserDebugParameter("Clear Trail", 1, 0, 1)
btn_camera   = p.addUserDebugParameter("Reset View", 1, 0, 1)

target_marker = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
                                             rgbaColor=[1, 1, 0, 1]),
    basePosition=[cx, cy, cz])

p.addUserDebugText("ROBOTIC ARM  -  PATH TRACING DEMO",
                   [cx, cy, cz + 0.30], textColorRGB=[0.4, 0.8, 1.0], textSize=1.6)



def regenerate(shape_id, size):
    """(Re)build the path, redraw the planned (green) outline, reset progress."""
    pts = build_shape(shape_id, (cx, cy, cz), size)
    for lid in planned_lines:
        p.removeUserDebugItem(lid)
    planned_lines.clear()
    for i in range(len(pts)):
        planned_lines.append(p.addUserDebugLine(
            pts[i], pts[(i + 1) % len(pts)],
            lineColorRGB=[0.2, 0.9, 0.4], lineWidth=3, lifeTime=0))
    return pts


planned_lines = []
trail_lines   = []

cur_shape = 0
cur_size  = 0.16
points    = regenerate(cur_shape, cur_size)

current   = 0
alpha     = 0.0
prev_pos  = get_ee_pos()
err_text  = None
link_line = None
info_text = None

paused        = False
prev_pause    = 1
prev_clear    = 1
prev_camera   = 1


for _ in range(800):
    move_joints_to(points[0])
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    if distance(get_ee_pos(), points[0]) < 0.02:
        break
prev_pos = get_ee_pos()

max_err = 0.0
sum_err = 0.0
n_err   = 0

print("Simulation running. Use the panel on the right to control it.")


while True:
    shape_id = int(round(p.readUserDebugParameter(slider_shape)))
    speed    = p.readUserDebugParameter(slider_speed)
    size     = p.readUserDebugParameter(slider_size)
    pause_c  = p.readUserDebugParameter(btn_pause)
    clear_c  = p.readUserDebugParameter(btn_clear)
    cam_c    = p.readUserDebugParameter(btn_camera)

    if shape_id != cur_shape or abs(size - cur_size) > 1e-4:
        cur_shape, cur_size = shape_id, size
        points  = regenerate(cur_shape, cur_size)
        current = 0
        alpha   = 0.0
        max_err = sum_err = 0.0
        n_err   = 0

    if pause_c != prev_pause:
        paused = not paused
        prev_pause = pause_c

    if clear_c != prev_clear:
        for lid in trail_lines:
            p.removeUserDebugItem(lid)
        trail_lines.clear()
        prev_clear = clear_c

    if cam_c != prev_camera:
        reset_camera()
        prev_camera = cam_c

    start = points[current]
    end   = points[(current + 1) % len(points)]
    target = [start[i] + alpha * (end[i] - start[i]) for i in range(3)]

    move_joints_to(target)
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

    curr_pos = get_ee_pos()
    error    = distance(curr_pos, target)
    on_path  = error < 0.02

    max_err = max(max_err, error)
    sum_err += error
    n_err   += 1

    p.resetBasePositionAndOrientation(target_marker, target, [0, 0, 0, 1])

    color = [0.2, 0.9, 0.4] if on_path else [1, 0.25, 0.25]
    trail_lines.append(p.addUserDebugLine(prev_pos, curr_pos,
                                          lineColorRGB=color, lineWidth=4, lifeTime=0))
    prev_pos = curr_pos


    link_line = p.addUserDebugLine(
        target, curr_pos, lineColorRGB=[1, 1, 1], lineWidth=1,
        lifeTime=0, replaceItemUniqueId=link_line if link_line is not None else -1)


    status = "ON PATH" if on_path else "OFF PATH"
    info_text = p.addUserDebugText(
        f"{SHAPE_NAMES[cur_shape]}   |   {status}   "
        f"err {error*1000:5.1f} mm   max {max_err*1000:5.1f} mm   "
        f"avg {sum_err/n_err*1000:5.1f} mm   {'[PAUSED]' if paused else ''}",
        [cx, cy, cz + 0.22], textColorRGB=color, textSize=1.3,
        lifeTime=0, replaceItemUniqueId=info_text if info_text is not None else -1)


    if not paused:
        alpha += speed
        if alpha >= 1.0:
            alpha = 0.0
            current = (current + 1) % len(points)
