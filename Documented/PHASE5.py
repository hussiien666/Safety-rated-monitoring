"""
Robotic Arm Path-Tracing Simulation  (PHASE5.py)
------------------------------------
Interactive PyBullet demo: the arm follows a user-selected geometric path
(square / rectangle / triangle / circle) while the simulation visualises the
planned path, the real end-effector trail, and the live tracking error.

This phase swaps the simple 2-D OpenCV arm of PHASE4 for a real physics-based
3-D robot (loaded from a URDF) driven by inverse kinematics.

Controls (panel on the right of the window):
    Shape          0=Square  1=Rectangle  2=Triangle  3=Circle
    Speed          how fast the arm moves along the path
    Size           overall size of the shape
    Pause / Run    freeze / resume the motion
    Clear Trail    erase the drawn trail
    Reset View     re-centre the camera
"""

import pybullet as p                              # PyBullet: the physics engine + 3-D GUI.
import pybullet_data                              # Bundled assets (e.g. the ground 'plane.urdf').
import time                                        # time: pace the simulation at ~240 Hz.
import math                                        # math: trig for the circle waypoints.


p.connect(p.GUI, options="--width=1280 --height=720 --background_color_red=0.12 "  # Open the 3-D GUI window...
                         "--background_color_green=0.13 --background_color_blue=0.18")  # ...with a dark bg.
p.setGravity(0, 0, -9.81)                          # Earth gravity along -Z.
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Let loadURDF find the bundled assets.

p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)          # Hide the RGB preview pane (cleaner UI).
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)        # Hide the depth preview pane.
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)   # Hide the segmentation preview pane.
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)                     # Enable shadows (nicer look).
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)                         # Keep the parameter side-panel visible.

plane = p.loadURDF("plane.urdf")                   # Load a flat ground plane.

robot = p.loadURDF(                                # Load the robot arm model from its URDF file...
    r"C:\Users\original\OneDrive\Desktop\python codes grad\assembly trial\urdf\assembly trial.urdf",  # path
    [0, 0, 0],                                     # ...placed at the world origin...
    useFixedBase=True                              # ...with its base bolted down (won't fall over).
)
num_joints = p.getNumJoints(robot)                 # Total number of joints in the model.
end_effector = num_joints - 1                      # Assume the last joint/link is the tool tip.

movable_joints = [j for j in range(num_joints)     # Build the list of joints we can actually drive...
                  if p.getJointInfo(robot, j)[2] != p.JOINT_FIXED]  # ...excluding fixed joints.

print("Number of joints:", num_joints)             # Diagnostics: print joint count...
for i in range(num_joints):                        # ...and each joint's name...
    print(i, p.getJointInfo(robot, i)[1].decode("utf-8"))  # (index 1 is the name, as bytes → decode).
print("Movable joints:", movable_joints)           # ...and which ones are movable.



def get_ee_pos():                                  # Current world XYZ of the end-effector.
    return list(p.getLinkState(robot, end_effector)[4])  # Index 4 = world position of the link frame.


def move_joints_to(target_pos):                    # Drive the arm so the tool reaches target_pos.
    joint_positions = p.calculateInverseKinematics(robot, end_effector, target_pos)  # Solve IK → joint angles.
    # joint_positions[k] corresponds to movable_joints[k], NOT to joint index k.
    for k, j in enumerate(movable_joints):         # For each movable joint...
        if k >= len(joint_positions):              # (guard: never index past the IK solution)
            break
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL,  # ...command it toward the solved angle...
                                targetPosition=joint_positions[k], force=500)  # ...with up to 500 N·m.


def distance(a, b):                                # Euclidean distance between two 3-D points.
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5  # sqrt(Σ (a-b)²) over x,y,z.


def reset_camera():                                # Re-centre the debug camera on the workspace.
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=50,
                                 cameraPitch=-35, cameraTargetPosition=[cx, cy, cz])


def build_shape(shape_id, center, size):
    """Return a list of waypoints for the chosen shape, centred on `center`."""
    cx, cy, cz = center                            # Unpack the centre point.
    s = size                                       # Overall size of the shape.
    if shape_id == 0:                      # Square
        h = s / 2                                  # Half-side.
        return [[cx - h, cy - h, cz], [cx + h, cy - h, cz],   # Four corners of the square...
                [cx + h, cy + h, cz], [cx - h, cy + h, cz]]   # ...(in order).
    if shape_id == 1:                      # Rectangle
        w, h = s * 0.75, s * 0.45                  # Half-width and half-height (non-square).
        return [[cx - w, cy - h, cz], [cx + w, cy - h, cz],
                [cx + w, cy + h, cz], [cx - w, cy + h, cz]]
    if shape_id == 2:                      # Triangle
        h = s / 2                                  # Half-size.
        return [[cx, cy + h, cz], [cx + h, cy - h, cz], [cx - h, cy - h, cz]]  # Apex + two base corners.
    # Circle (approximated by N points)
    n = 36                                         # Number of segments approximating the circle.
    r = s / 2                                      # Radius.
    return [[cx + r * math.cos(2 * math.pi * k / n),   # X around the circle...
             cy + r * math.sin(2 * math.pi * k / n), cz] for k in range(n)]  # ...Y around, Z fixed.


SHAPE_NAMES = ["Square", "Rectangle", "Triangle", "Circle"]  # Display names indexed by shape_id.



for _ in range(300):                               # Let the robot settle under gravity for 300 steps...
    p.stepSimulation()                             # ...advance the physics one tick...
    time.sleep(1.0 / 240.0)                        # ...at ~240 Hz real-time pace.

ee = get_ee_pos()                                  # Read where the tool ended up.
cx, cy, cz = ee                                    # Use that as the centre of all drawn shapes.
print(f"End-effector starts at: x={cx:.3f} y={cy:.3f} z={cz:.3f}")  # Report the start pose.
reset_camera()                                     # Aim the camera at it.


slider_shape = p.addUserDebugParameter("Shape  0=Sq 1=Rec 2=Tri 3=Cir", 0, 3, 0)  # Shape selector slider.
slider_speed = p.addUserDebugParameter("Speed", 0.001, 0.03, 0.008)               # Path speed slider.
slider_size  = p.addUserDebugParameter("Size", 0.08, 0.30, 0.16)                  # Shape size slider.
btn_pause    = p.addUserDebugParameter("Pause / Run", 1, 0, 1)                    # Pause/run "button".
btn_clear    = p.addUserDebugParameter("Clear Trail", 1, 0, 1)                    # Clear-trail "button".
btn_camera   = p.addUserDebugParameter("Reset View", 1, 0, 1)                     # Reset-camera "button".

target_marker = p.createMultiBody(                 # A small yellow sphere that marks the current target...
    baseMass=0,                                    # massless (visual only)...
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.012,
                                             rgbaColor=[1, 1, 0, 1]),
    basePosition=[cx, cy, cz])                     # ...starting at the workspace centre.

p.addUserDebugText("ROBOTIC ARM  -  PATH TRACING DEMO",  # Floating title text in the 3-D scene.
                   [cx, cy, cz + 0.30], textColorRGB=[0.4, 0.8, 1.0], textSize=1.6)



def regenerate(shape_id, size):
    """(Re)build the path, redraw the planned (green) outline, reset progress."""
    pts = build_shape(shape_id, (cx, cy, cz), size)  # Compute the new waypoints.
    for lid in planned_lines:                      # Remove the previously drawn planned outline...
        p.removeUserDebugItem(lid)
    planned_lines.clear()                          # ...and forget those line ids.
    for i in range(len(pts)):                      # Draw the new outline edge by edge...
        planned_lines.append(p.addUserDebugLine(
            pts[i], pts[(i + 1) % len(pts)],       # ...connecting each point to the next (wrap around).
            lineColorRGB=[0.2, 0.9, 0.4], lineWidth=3, lifeTime=0))  # Green, persistent.
    return pts                                     # Return the waypoint list.


planned_lines = []                                 # Debug-line ids of the planned (green) outline.
trail_lines   = []                                 # Debug-line ids of the actual end-effector trail.

cur_shape = 0                                      # Currently selected shape.
cur_size  = 0.16                                   # Currently selected size.
points    = regenerate(cur_shape, cur_size)        # Build the initial path + draw it.

current   = 0                                      # Index of the current path segment.
alpha     = 0.0                                    # Progress 0..1 along the current segment.
prev_pos  = get_ee_pos()                           # Previous tool position (for drawing the trail).
err_text  = None                                   # (Reserved) error-text item id.
link_line = None                                   # Id of the white "target↔tool" link line.
info_text = None                                   # Id of the on-screen status text.

paused        = False                              # Whether motion is paused.
prev_pause    = 1                                  # Last raw value of the pause button (edge detection).
prev_clear    = 1                                  # Last raw value of the clear button.
prev_camera   = 1                                  # Last raw value of the camera button.


for _ in range(800):                               # Pre-move the arm onto the first waypoint...
    move_joints_to(points[0])                      # ...command IK toward point 0...
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
    if distance(get_ee_pos(), points[0]) < 0.02:   # ...stop early once we're within 2 cm.
        break
prev_pos = get_ee_pos()                            # Reset the trail's starting point.

max_err = 0.0                                      # Largest tracking error seen.
sum_err = 0.0                                      # Running sum of errors (for the average).
n_err   = 0                                        # Count of samples (for the average).

print("Simulation running. Use the panel on the right to control it.")  # Tell the user it's live.


while True:                                         # Main simulation loop (runs until the window closes).
    # ---- read the control panel ----
    shape_id = int(round(p.readUserDebugParameter(slider_shape)))  # Read & round the shape slider.
    speed    = p.readUserDebugParameter(slider_speed)              # Read the speed slider.
    size     = p.readUserDebugParameter(slider_size)               # Read the size slider.
    pause_c  = p.readUserDebugParameter(btn_pause)                 # Read the pause button counter.
    clear_c  = p.readUserDebugParameter(btn_clear)                 # Read the clear button counter.
    cam_c    = p.readUserDebugParameter(btn_camera)                # Read the camera button counter.

    if shape_id != cur_shape or abs(size - cur_size) > 1e-4:       # If shape or size changed...
        cur_shape, cur_size = shape_id, size                       # ...remember the new selection...
        points  = regenerate(cur_shape, cur_size)                  # ...rebuild & redraw the path...
        current = 0                                                # ...restart at segment 0...
        alpha   = 0.0                                              # ...at the start of that segment...
        max_err = sum_err = 0.0                                    # ...and reset the error stats.
        n_err   = 0

    if pause_c != prev_pause:                       # PyBullet buttons increment a counter on each click...
        paused = not paused                         # ...so a change means "clicked" → toggle pause.
        prev_pause = pause_c

    if clear_c != prev_clear:                       # Clear-trail clicked...
        for lid in trail_lines:                     # ...remove every trail segment...
            p.removeUserDebugItem(lid)
        trail_lines.clear()                         # ...and forget them.
        prev_clear = clear_c

    if cam_c != prev_camera:                        # Reset-view clicked...
        reset_camera()                              # ...re-centre the camera.
        prev_camera = cam_c

    start = points[current]                         # Start point of the current segment.
    end   = points[(current + 1) % len(points)]     # End point (wraps to the first point at the end).
    target = [start[i] + alpha * (end[i] - start[i]) for i in range(3)]  # Linear interpolation start→end.

    move_joints_to(target)                          # Command the arm toward the interpolated target.
    p.stepSimulation()                              # Advance physics one step.
    time.sleep(1.0 / 240.0)                         # Pace the loop.

    curr_pos = get_ee_pos()                         # Where the tool actually is now.
    error    = distance(curr_pos, target)           # How far it is from the commanded target.
    on_path  = error < 0.02                         # "On path" if within 2 cm.

    max_err = max(max_err, error)                   # Update worst-case error.
    sum_err += error                                # Accumulate for the average.
    n_err   += 1

    p.resetBasePositionAndOrientation(target_marker, target, [0, 0, 0, 1])  # Move the yellow marker to target.

    # actual trail (green on-path / red off-path)
    color = [0.2, 0.9, 0.4] if on_path else [1, 0.25, 0.25]  # Green when accurate, red when drifting.
    trail_lines.append(p.addUserDebugLine(prev_pos, curr_pos,  # Draw a trail segment from last to current...
                                          lineColorRGB=color, lineWidth=4, lifeTime=0))  # ...persistent.
    prev_pos = curr_pos                             # Advance the trail's anchor.


    link_line = p.addUserDebugLine(                 # Thin white line showing the target↔tool gap (the error)...
        target, curr_pos, lineColorRGB=[1, 1, 1], lineWidth=1,
        lifeTime=0, replaceItemUniqueId=link_line if link_line is not None else -1)  # ...reuse one item id.


    status = "ON PATH" if on_path else "OFF PATH"   # Status word.
    info_text = p.addUserDebugText(                 # Update the on-screen HUD text (reusing one item id)...
        f"{SHAPE_NAMES[cur_shape]}   |   {status}   "                    # shape + status...
        f"err {error*1000:5.1f} mm   max {max_err*1000:5.1f} mm   "      # ...current & max error in mm...
        f"avg {sum_err/n_err*1000:5.1f} mm   {'[PAUSED]' if paused else ''}",  # ...average + pause flag.
        [cx, cy, cz + 0.22], textColorRGB=color, textSize=1.3,
        lifeTime=0, replaceItemUniqueId=info_text if info_text is not None else -1)


    if not paused:                                  # Only progress along the path when running...
        alpha += speed                              # ...advance interpolation by the chosen speed...
        if alpha >= 1.0:                            # ...and when we reach the segment end...
            alpha = 0.0                             # ...reset progress...
            current = (current + 1) % len(points)   # ...and move to the next segment (wrap around).
