import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import pdb
from math import ceil

def move_robot(robot, desired_joint_positions):
    for i in range(3):
        p.resetJointState(robot, movable_joints[i], desired_joint_positions[i])

def generate_trajectory_single_dimension(x0, x1, v0, v1):
    M = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 2, 3],
        [0, 1, 0, 0]
    ])
    b = np.array([x0, x1, v0, v1])
    A, B, C, D = np.linalg.solve(M, b)
    return A, B, C, D
    

def generate_trajectory_2d(p0, p1, v0, v1, n_samples):
    t = np.linspace(0, 1, n_samples)
    if(p0[0] == p1[0]):
        x = p0[0]*np.ones(n_samples)
        A,B,C,D = generate_trajectory_single_dimension(p0[1], p1[1], v0[1], v1[1])
        y = A + B*t + C*t**2 + D*t**3
    else:
        A,B,C,D = generate_trajectory_single_dimension(p0[0], p1[0], v0[0], v1[0])
        x = A + B*t + C*t**2 + D*t**3
        y = p0[1]*np.ones(n_samples)+ (p1[1] - p0[1])/(p1[0] - p0[0])*(x-p0[0])
    return (x, y)

def plot_trajectory(x, y):
    plt.plot(y)
    plt.show()

def get_trajectory(robot,cartesian_task_trajectory,n_samples):
    x = []
    y = []
    for i in range(len(cartesian_task_trajectory) - 1):
        p0 = cartesian_task_trajectory[i]
        p1 = cartesian_task_trajectory[i+1]
        print(f"\n=== TRAJECTORY SEGMENT {i} ===")
        print(f"From: {p0} to {p1}")
        print(f"Delta: ({p1[0]-p0[0]:.3f}, {p1[1]-p0[1]:.3f})")
        
        current_trajectory = generate_trajectory_2d(p0, p1, [0,0], [0,0], n_samples)
        
        x = np.concatenate((x, current_trajectory[0]))
        y = np.concatenate((y, current_trajectory[1]))
    return x, y

def interpolate_waypoint(current_waypoint, prev_waypoint, time_since_prev_waypoint, time_to_next_waypoint):
    if(time_since_prev_waypoint > time_to_next_waypoint):
        return current_waypoint[0], current_waypoint[1]
    x_des, y_des = current_waypoint
    x_prev, y_prev = prev_waypoint
    x_des = x_prev + (x_des - x_prev)*(time_since_prev_waypoint)/time_to_next_waypoint
    y_des = y_prev + (y_des - y_prev)*(time_since_prev_waypoint)/time_to_next_waypoint
    return x_des, y_des

def inverse_kinematics_3R(x, y, theta, l1, l2, l3):
    # Compute wrist position
    x_w = x - l3 * np.cos(theta)
    y_w = y - l3 * np.sin(theta)
    
    # Compute theta2 using law of cosines
    D = (x_w**2 + y_w**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if np.abs(D) > 1.0:
        raise ValueError("Target is out of reach")
    theta2 = np.arctan2(np.sqrt(1 - D**2), D)  # elbow up
    # theta2 = np.arctan2(-np.sqrt(1 - D**2), D)  # elbow down (alternative)
    
    # Compute theta1
    phi = np.arctan2(y_w, x_w)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    k1 = l1 + l2 * c2
    k2 = l2 * s2
    theta1 = phi - np.arctan2(k2, k1)
    
    # Compute theta3
    theta3 = theta - (theta1 + theta2)
    
    return np.array([theta1, theta2, theta3])


def damped_least_squares(J, lambd=0.01):
    # J: Jacobian (m x n)
    m, n = J.shape
    JJt = J @ J.T
    damped = JJt + (lambd ** 2) * np.eye(m)
    damped_inv = np.linalg.inv(damped)
    J_pinv_DLS = J.T @ damped_inv
    return J_pinv_DLS

# def grad_h(q):
#     return np.array([
#         -2*(q[1] - q[0]),
#         2*(q[1] - q[0]) - 2*(q[2] - q[1]),
#         2*(q[2] - q[1])
#     ])

def grad_h(q):
    d1 = angle_diff(q[0], q[1])
    d2 = angle_diff(q[1], q[2])
    return np.array([
        0,
        2*(q[1]),
        2*(q[2])
    ])


def angle_diff(a, b):
    """Returns the signed shortest difference between angles a and b (in radians)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def custom_grad_h(q):
    d1 = q[1] - q[0]
    d2 = q[2] - q[1]
    grad = np.zeros(3)
    grad[0] = -np.sin(d1) / 2
    grad[1] = np.sin(d1) / 2 - np.sin(d2) / 2
    grad[2] = np.sin(d2) / 2
    return grad.reshape(-1, 1)


def sin2(x):
    """Returns sin^2(x/2) for smooth periodic cost."""
    return np.sin(x / 2) ** 2

def custom_h(q):
    # q: array-like, [theta1, theta2, theta3]
    cost = sin2(q[1] - q[0]) + sin2(q[2] - q[1])
    return cost

def dq_xy(robot, end_effector_link_idx, joint_positions, zero_vel, zero_acc, dx):
    jac_t, jac_r = p.calculateJacobian(robot, end_effector_link_idx, [0,0,0], joint_positions, zero_vel, zero_acc)
    J = np.vstack([np.array(jac_t)[:2, :], np.array(jac_r)[2, :][None, :]])  # x, y, theta
    
    J_pinv_DLS = damped_least_squares(J[:2, :])
    dx = dx[0:2]
    dq = J_pinv_DLS @ dx.reshape(-1,1)
    P = np.eye(3) - J_pinv_DLS @ J[:2, :]
    return dq, P

def dq_xyt(robot, end_effector_link_idx, joint_positions, zero_vel, zero_acc, dx):
    jac_t, jac_r = p.calculateJacobian(robot, end_effector_link_idx, [0,0,0], joint_positions, zero_vel, zero_acc)
    J = np.vstack([np.array(jac_t)[:2, :], np.array(jac_r)[2, :][None, :]])  # x, y, theta
    J_pinv_DLS = damped_least_squares(J)
    dq = J_pinv_DLS @ dx.reshape(-1,1)
    P = np.eye(3) - J_pinv_DLS @ J
    return dq, P

def run_all_configurations(robot,desired_joint_positions):
    for joint_positions in desired_joint_positions:
        move_robot(robot, joint_positions)
        time.sleep(5)
    exit()

def save_plots(end_effector_position, command_position, theta):
    plt.figure()
    plt.plot(end_effector_position[:,0],end_effector_position[:,1], label='End Effector Position')
    plt.plot(command_position[:,0], command_position[:,1], label='Command Position')
    plt.title('trajectory')
    plt.legend(loc='right')
    plt.savefig('trajectory.png')
    plt.close()
def save_individual_control_dim(end_effector_position, command_position, theta):
    plt.figure()
    plt.plot(end_effector_position[:,0],label = 'x')
    plt.plot(command_position[:,0],label = 'x_des')
    plt.plot(end_effector_position[:,1],label = 'y')
    plt.plot(theta,label = 'theta')
    plt.plot(command_position[:,1],label = 'y_des')
    plt.legend()
    plt.savefig('individual_control_dim.png')
    plt.close()

def draw_trajectory_in_pybullet(x, y, z_height=0.02, line_color=(0, 1, 0), line_width=2, sphere_radius=0.01):
    """Draws the planned trajectory in the PyBullet GUI using lines and small spheres."""
    # Draw lines between successive trajectory points
    for i in range(len(x) - 1):
        p.addUserDebugLine([float(x[i]), float(y[i]), z_height],
                           [float(x[i+1]), float(y[i+1]), z_height],
                           lineColorRGB=line_color, lineWidth=line_width, lifeTime=0)

    # Draw small spheres at each trajectory point
    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius,
                                       rgbaColor=[line_color[0], line_color[1], line_color[2], 1.0])
    for i in range(len(x)):
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=-1,
                          baseVisualShapeIndex=visual_shape,
                          basePosition=[float(x[i]), float(y[i]), z_height])

if __name__ == '__main__':
    run_part_no = 3 # 1 for open loop, 2 for closed loop, 3 for null space control   
    if run_part_no is None:
        raise Exception('please set run_part_no: 1 for open loop, 2 for closed loop, 3 for null space control')
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    plane = p.loadURDF('plane.urdf')
    robot = p.loadURDF('three-link-robot.urdf', useFixedBase = True)
    
    # Start recording for GIF
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot_animation.mp4")

    # get three movable joints and the end-effector link's index
    num_joints = p.getNumJoints(robot)
    movable_joints = []
    end_effector_link_idx = None
    for idx in range(num_joints): 
        info = p.getJointInfo(robot, idx)
        print('Joint {}: {}'.format(idx, info))

        joint_type = info[2]
        if joint_type != p.JOINT_FIXED: 
            movable_joints.append(idx)

        link_name = info[12].decode('utf-8')
        if link_name == 'end_effector': 
            end_effector_link_idx = idx

    # TODO: Your code here
    desired_joint_positions = np.array([[0,0,np.pi/2],[0,np.pi,0],[np.pi/2,np.pi/2,np.pi/4],[np.pi/3,np.pi/2,0]])
    # run_all_configurations(robot,desired_joint_positions)
    initial_position = np.array([1.2,0])
    total_task_time = 10
    cartesian_task_trajectory = np.array([initial_position, [0.5, 0.5],[0.5, -0.5], [0.4, 0], initial_position])
    n_samples = 10
    total_samples = n_samples*(len(cartesian_task_trajectory)-1)
    x,y = get_trajectory(robot,cartesian_task_trajectory,n_samples=n_samples)
    # Draw planned trajectory points/lines in the PyBullet GUI
    draw_trajectory_in_pybullet(x, y, z_height=0.02, line_color=(0,0,0), line_width=2, sphere_radius=0.008)
    # Add some points at the end to hold the final position
    dummy_x, dummy_y = np.ones(10)*x[-1], np.ones(10)*y[-1]
    x = np.concatenate((x, dummy_x))
    y = np.concatenate((y, dummy_y))        
    
    print(f"Trajectory starts at: ({x[0]:.3f}, {y[0]:.3f})")
    print(f"First waypoint should be: ({cartesian_task_trajectory[0][0]:.3f}, {cartesian_task_trajectory[0][1]:.3f})")
    print(f"Total trajectory points: {len(x)}")
    # plt.plot(x, y)
    # plt.show()
    end_effector_position = []
    command_position = []
    timer = []
    theta = []
    
    # Path plotting variables
    path_line_id = None
    path_points = []
    path_spheres = []

    start_time = time.time()
    prev_waypoint_idx = 0
    prev_waypoint = (x[0], y[0])
    waypoint_update_time = time.time()

    zero_vel = [0.0] * len(movable_joints)
    zero_acc = [0.0] * len(movable_joints)
    # move_robot(robot, [0,-np.pi + np.pi/15,np.pi/15])
    # time.sleep(2)

    separation = []
    # hold Ctrl and use the mouse to rotate, pan, or zoom
    time_to_next_waypoint = total_task_time/total_samples
    time_since_prev_waypoint = time_to_next_waypoint
    for _ in range(2400): 
        p.stepSimulation()
        time.sleep(1./240.)

        curr_waypoint_idx = min(int(ceil((time.time() - start_time)/total_task_time*total_samples)), total_samples-1)
        if curr_waypoint_idx != prev_waypoint_idx:
            waypoint_update_time = time.time()
            prev_waypoint = (x[prev_waypoint_idx], y[prev_waypoint_idx])
            prev_waypoint_idx = curr_waypoint_idx
            time_since_prev_waypoint = 0
        else:
            time_since_prev_waypoint = time.time() - waypoint_update_time

        current_waypoint = (x[curr_waypoint_idx], y[curr_waypoint_idx])

        current_ee_pos = p.getLinkState(robot, end_effector_link_idx)[0]
        
        x_des, y_des = interpolate_waypoint(current_waypoint, prev_waypoint, time_since_prev_waypoint, time_to_next_waypoint)
        # print('x_des, y_des', x_des, y_des)

        joint_positions = [joint_state[0] for joint_state in p.getJointStates(robot, movable_joints)]
        
        # # # ## OPEN LOOP CONTROL
        if run_part_no == 1:
            q_des = inverse_kinematics_3R(x_des, y_des, 0, 0.5, 0.5, 0.2)
            p.setJointMotorControlArray(robot, movable_joints, p.POSITION_CONTROL, targetPositions=q_des)
        ## CLOSED LOOP CONTROL
        if run_part_no == 2 or run_part_no == 3:
            kp = np.array([0.2,0.2,0.05])*5
            err = np.array([x_des-current_ee_pos[0], y_des-current_ee_pos[1]])
            time_left = time_to_next_waypoint - time_since_prev_waypoint
            current_waypoint_dot = np.array([(current_waypoint[0]-current_ee_pos[0])/time_left, (current_waypoint[1]-current_ee_pos[1])/time_left])
            dx = np.array([kp[0]*err[0] + current_waypoint_dot[0], kp[1]*err[1] + current_waypoint_dot[1],-(np.sum(joint_positions))])
            

            if run_part_no == 2:
                dq, P = dq_xyt(robot, end_effector_link_idx, joint_positions, zero_vel, zero_acc, dx)
            elif run_part_no == 3:
                knull = -10
                dq, P = dq_xy(robot, end_effector_link_idx, joint_positions, zero_vel, zero_acc, dx)
                dq_null = knull*P@grad_h(joint_positions)
                print('dq_null', dq_null)
                dq = dq.reshape(-1) + dq_null.reshape(-1)
            p.setJointMotorControlArray(robot, movable_joints, p.VELOCITY_CONTROL, targetVelocities=dq.reshape(-1))



        # Logging for easy plotting 
        command_position.append([x_des, y_des, time.time()-start_time])
        end_effector_position.append(current_ee_pos)
        timer.append(time.time()-start_time)
        theta.append(np.sum(joint_positions))
        

    print('shape of end_effector_position', np.array(end_effector_position).shape)
    end_effector_position = np.array(end_effector_position)
    command_position = np.array(command_position)
    save_plots(end_effector_position, command_position, theta)
    save_individual_control_dim(end_effector_position, command_position, theta)
    # print('separation looks like', separation)
    # plt.plot(separation)
    # plt.show()
    
    # Stop recording
    p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
    print("Animation saved as robot_animation.mp4")
    print("To convert to GIF, use: ffmpeg -i robot_animation.mp4 -vf 'fps=10,scale=640:-1' robot_animation.gif")
    

    p.disconnect()
