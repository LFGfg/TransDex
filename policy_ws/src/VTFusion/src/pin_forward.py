import pinocchio as pin
import numpy as np

def get_left_arm_end_effector_pose(joint_values_list):
    """
    Calculate the end-effector poses corresponding to multiple sets of left arm joint angles
    
    Parameters:
    - joint_values_list: List of multiple sets of joint angle values, shape is (N, num_joints)
    
    Returns:
    - poses: Corresponding pose array, shape is (N, 7), format is [x, y, z, qw, qx, qy, qz]
    """
    # Load URDF model (load only once for efficiency)
    urdf_path = "urdf/gen3_dualarm.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    data = pin.Data(model)
    
    # Define left arm joint name list
    joint_names = [
        "l_shoulder_1_joint",
        "l_shoulder_2_joint",
        "l_shoulder_3_joint",
        "l_elbow_1_joint",
        "l_elbow_2_joint",
        "l_wrist_1_joint",
        "l_wrist_2_joint",
        "l_wrist_3_joint"
    ]
    
    # Ensure input is in "multiple sets of joint angles" format (even if only one set)
    # If input is a single set of joint angles (1D), convert to 2D list [[joint1, joint2, ...]]
    if isinstance(joint_values_list[0], (int, float)):
        joint_values_list = [joint_values_list]
    
    # Preallocate result array
    num_samples = len(joint_values_list)
    poses = np.zeros((num_samples, 6))
    prev_euler = None  # Track Euler angles of previous frame (for continuity)
    # Process each set of joint angles
    for i, joint_values in enumerate(joint_values_list):
        # Process joint values: divide the 4th value by 2 and copy insert to 5th position
        # Convert to list to support insert operation
        processed_joint_values = list(joint_values)
        if len(processed_joint_values) >= 4:
            processed_joint_values[3] /= 2  # Divide 4th value by 2
            processed_joint_values.insert(4, processed_joint_values[3])  # Insert to 5th position
        else:
            raise ValueError(f"Insufficient number of joint values in set {i}, at least 4 values required")
        
        # Create joint configuration vector
        q = np.zeros(model.nq)
        
        # Fill joint values
        for name, value in zip(joint_names, processed_joint_values):
            joint_id = model.getJointId(name)
            if joint_id >= model.njoints:
                raise ValueError(f"Joint name {name} does not exist in the model")
            idx_q = model.joints[joint_id].idx_q
            q[idx_q] = value
        
        # Calculate forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get end-effector pose
        end_effector_name = "l_wrist_3_joint"
        frame_id = model.getFrameId(end_effector_name)
        frame_pose = data.oMf[frame_id]
        
        # Extract position and quaternion
        position = frame_pose.translation
        rotation_matrix = frame_pose.rotation
        quaternion = pin.Quaternion(rotation_matrix)
        # Convert rotation matrix to Euler angles (XYZ order by default)
        # euler_angles = pin.rpy.matrixToRpy(rotation_matrix)  # Returns [rx, ry, rz]
        euler_angles = pin.rpy.matrixToRpy(quaternion.toRotationMatrix())


        # 6.1 First normalize to [-π, π] (handle abnormal values from calculation overflow)
        euler_angles = np.mod(euler_angles + np.pi, 2 * np.pi) - np.pi
        # 6.2 Then adjust continuity between adjacent frames (avoid jump from π to -π)
        if prev_euler is not None:
            for j in range(3):
                diff = euler_angles[j] - prev_euler[j]
                if diff > np.pi:
                    euler_angles[j] -= 2 * np.pi
                elif diff < -np.pi:
                    euler_angles[j] += 2 * np.pi
        prev_euler = euler_angles.copy()

        # Merge and store results (3 position values + 3 Euler angle values)
        poses[i] = np.concatenate([position, euler_angles])
        
        # quaternion = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
        
        # # Merge and store results
        # poses[i] = np.concatenate([position, quaternion])
    
    return poses

# Usage example
def main():
    # Define a single set of joint angle values (unit: radians)
    left_arm_joint_values = [
        8.16123212e-03, 3.53569502e-01, -1.68401812e-01,
        1.55646856e+00, 3.51136712e-01, 3.21226237e-03,
        -6.45490075e-04
    ]
    
    # Directly pass a single set of joint angles (automatically handled as multiple sets inside the function)
    poses = get_left_arm_end_effector_pose(left_arm_joint_values)
    
    # Parse results (poses is an array of shape (N, 7), N=1)
    position = poses[0, :3]  # Position [x, y, z]
    quaternion = poses[0, 3:]  # Quaternion [qw, qx, qy, qz]
    
    # Print results
    print("End-effector position:")
    print(f"x: {position[0]:.4f} m")
    print(f"y: {position[1]:.4f} m")
    print(f"z: {position[2]:.4f} m")
    
    print("\nEnd-effector quaternion:")
    print(f"qw: {quaternion[0]:.4f}")
    print(f"qx: {quaternion[1]:.4f}")
    print(f"qy: {quaternion[2]:.4f}")
    print(f"qz: {quaternion[3]:.4f}")

if __name__ == "__main__":
    main()