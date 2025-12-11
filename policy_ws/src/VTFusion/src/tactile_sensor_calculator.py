import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

def normalize_force(value, min_val, max_val):
    """Normalize force values to the range 0-255"""
    return (value - min_val) / (max_val - min_val) * 255.0

def denormalize_force(pixel_value, min_val, max_val):
    """Denormalize pixel values (0-255) back to the original force value range"""
    return pixel_value / 255.0 * (max_val - min_val) + min_val

def compute_tactile_sensor_data(joint_angles, urdf_path, tactile_image):
    """
    Calculate tactile sensor related data based on input joint angles and tactile image
    
    Parameters:
    joint_angles: Array of joint angles, corresponding to the joint order in the URDF model
    urdf_path: Path to the URDF model file
    tactile_image: 22×30 tactile image in RGB format, used to infer force data
    
    Returns:
    A dictionary containing:
    - contact_points: 3D positions of all contact points relative to the tactile sensor origin (N×3 array)
    - sensor_poses: 6D poses of the tactile sensor origin relative to the hand base (N×6 array, including x,y,z,rx,ry,rz)
    - forces: 3D forces at each contact point (N×3 array, inferred from tactile image)
    """
    joint_angles[6+8]=-joint_angles[6+8]
    
    joint_angles[7+8]=-joint_angles[7+8]
    # Thumb
    joint_angles[2+7+1+8]=(-joint_angles[2+7+1+8])
    joint_angles[3+7+1+8]=(-joint_angles[3+7+1+8])
    
    joint_angles[4+7+1+8]=-joint_angles[4+7+1+8]
    joint_angles[5+7+1+8]=-joint_angles[5+7+1+8]
    joint_angles[6+7+1+8]=-joint_angles[6+7+1+8]
    joint_angles[7+7+1+8]=-joint_angles[7+7+1+8]

    # Abduction
    joint_angles[8+7+1+8]=-joint_angles[8+7+1+8]
    joint_angles[9+7+1+8]=-joint_angles[9+7+1+8]
    joint_angles[10+7+1+8]=joint_angles[10+7+1+8]
    joint_angles[11+7+1+8]=joint_angles[11+7+1+8]
    joint_angles[12+7+1+8]=joint_angles[12+7+1+8]
    joint_angles[13+7+1+8]=joint_angles[13+7+1+8]

    joint_angles[16+7+1+8]=-joint_angles[16+7+1+8]
    joint_angles[17+7+1+8]=-joint_angles[17+7+1+8]       
    joint_angles[18+7+1+8]=joint_angles[18+7+1+8]
    joint_angles[19+7+1+8]=joint_angles[19+7+1+8]
    joint_angles[20+7+1+8]=joint_angles[20+7+1+8]
    joint_angles[21+7+1+8]=joint_angles[21+7+1+8]


    joint_angles[24+7+1+8]=-joint_angles[24+7+1+8]
    joint_angles[25+7+1+8]=-joint_angles[25+7+1+8]  
    joint_angles[26+7+1+8]=joint_angles[26+7+1+8]
    joint_angles[27+7+1+8]=joint_angles[27+7+1+8]
    joint_angles[28+7+1+8]=joint_angles[28+7+1+8]
    joint_angles[29+7+1+8]=joint_angles[29+7+1+8]

    joint_angles[32+7+1+8]=-joint_angles[32+7+1+8]
    joint_angles[33+7+1+8]=-joint_angles[33+7+1+8] 
    joint_angles[34+7+1+8]=joint_angles[34+7+1+8]
    joint_angles[35+7+1+8]=joint_angles[35+7+1+8]
    joint_angles[36+7+1+8]=joint_angles[36+7+1+8]
    joint_angles[37+7+1+8]=joint_angles[37+7+1+8]    
    
    joint_angles=joint_angles[8:]

    target_order_indices = [
        # Arm joints (1-8)
        0, 1, 2, 3, 4, 5, 6, 7,
        # Index finger joints (9-16)
        16, 17, 18, 19, 20, 21, 22, 23,
        # Little finger joints (17-24)
        32, 33, 34, 35, 36, 37, 38, 39,
        # Middle finger joints (25-32)
        24, 25, 26, 27, 28, 29, 30, 31,
        # Ring finger joints (33-40)
        40, 41, 42, 43, 44, 45, 46, 47,
        # Thumb joints (41-47) - Note: There are 7 joints in your list (41-47)
        8, 9, 10, 11, 12, 13, 14,15  # Original thumb joint indices
    ]
    joint_angles = np.array([joint_angles[i] for i in target_order_indices])

    # Check if tactile image dimensions are correct
    if tactile_image.shape[:2] != (22, 30):
        raise ValueError(f"Invalid tactile image dimensions, expected 22×30, got {tactile_image.shape[:2]}")
    
    # Define normalization parameters for three axes (corresponding to C++ code)
    norm_params = [
        {-15.0, 15.0},   # X-axis range
        {-15.0, 15.0},   # Y-axis range
        {0.0, 25.0}      # Z-axis range
    ]
    
    # Define initial coordinates of contact points (converted from C++ code)
    contact_points_DP_initial = np.array([
        [-7.33, 1.00, 2.08], [-7.30, 3.26, 2.09], [-7.28, 5.52, 2.08], [-7.25, 7.77, 2.03], 
        [-7.22, 10.03, 1.92], [-7.18, 12.27, 1.72], [-7.12, 14.50, 1.36], [-7.05, 16.67, 0.77], 
        [-6.94, 18.74, -0.13], [-6.79, 20.62, -1.36], [-6.58, 22.26, -2.89], [-6.32, 23.66, -4.64], 
        [-3.86, 1.62, 3.33], [-3.85, 4.01, 3.31], [-3.85, 6.39, 3.25], [-3.85, 8.78, 3.15], 
        [-3.85, 11.16, 2.97], [-3.86, 13.52, 2.65], [-3.88, 15.85, 2.14], [-3.89, 18.10, 1.34], 
        [-3.88, 20.19, 0.19], [-3.78, 22.05, -1.29], [-3.55, 23.64, -3.06], [-3.24, 24.97, -5.01], 
        [-1.28, 1.66, 3.40], [-1.28, 4.01, 3.38], [-1.27, 6.40, 3.33], [-1.27, 8.78, 3.23], 
        [-1.26, 11.17, 3.06], [-1.26, 13.54, 2.76], [-1.26, 15.88, 2.26], [-1.26, 18.16, 1.48], 
        [-1.24, 20.29, 0.35], [-1.21, 22.19, -1.15], [-1.13, 23.79, -2.94], [-1.02, 25.10, -4.94], 
        [1.28, 1.66, 3.40], [1.28, 4.01, 3.38], [1.27, 6.40, 3.33], [1.27, 8.78, 3.23], 
        [1.26, 11.17, 3.06], [1.26, 13.54, 2.76], [1.26, 15.88, 2.26], [1.26, 18.16, 1.48], 
        [1.24, 20.29, 0.35], [1.21, 22.19, -1.15], [1.13, 23.79, -2.94], [1.02, 25.10, -4.94], 
        [3.86, 1.62, 3.33], [3.85, 4.01, 3.31], [3.85, 6.39, 3.25], [3.85, 8.78, 3.15], 
        [3.85, 11.16, 2.97], [3.86, 13.52, 2.65], [3.88, 15.85, 2.14], [3.89, 18.10, 1.34], 
        [3.88, 20.19, 0.19], [3.78, 22.05, -1.29], [3.55, 23.64, -3.06], [3.24, 24.97, -5.01], 
        [7.33, 1.00, 2.08], [7.30, 3.26, 2.09], [7.28, 5.52, 2.08], [7.25, 7.77, 2.03], 
        [7.22, 10.03, 1.92], [7.18, 12.27, 1.72], [7.12, 14.50, 1.36], [7.05, 16.67, 0.77], 
        [6.94, 18.74, -0.13], [6.79, 20.62, -1.36], [6.58, 22.26, -2.89], [6.32, 23.66, -4.64]
    ])
    
    contact_points_IP_initial = np.array([
        [ -7.27, 0.97, 1.85 ], [ -7.26, 3.49, 1.85 ], [ -7.27, 6.00, 1.85 ], [ -7.27, 8.52, 1.85 ],
        [ -7.26, 11.04, 1.85 ], [ -7.26, 13.56, 1.85 ], [ -7.27, 16.08, 1.85 ], [ -7.27, 18.59, 1.85 ],
        [ -7.26, 21.11, 1.85 ], [ -7.27, 23.63, 1.85 ], [ -4.56, 1.52, 2.97 ], [ -4.56, 3.91, 2.97 ],
        [ -4.56, 6.31, 2.97 ], [ -4.56, 8.71, 2.97 ], [ -4.56, 11.10, 2.97 ], [ -4.56, 13.50, 2.97 ],
        [ -4.56, 15.89, 2.97 ], [ -4.56, 18.29, 2.97 ], [ -4.56, 20.69, 2.97 ], [ -4.56, 23.08, 2.97 ],
        [ -1.52, 1.59, 3.10 ], [ -1.52, 3.91, 3.10 ], [ -1.52, 6.31, 3.10 ], [ -1.52, 8.71, 3.10 ],
        [ -1.52, 11.10, 3.10 ], [ -1.52, 13.50, 3.10 ], [ -1.52, 15.89, 3.10 ], [ -1.52, 18.29, 3.10 ],
        [ -1.52, 20.69, 3.10 ], [ -1.52, 23.01, 3.10 ], [ 1.52, 1.59, 3.10 ], [ 1.52, 3.91, 3.10 ],
        [ 1.52, 6.31, 3.10 ], [ 1.52, 8.71, 3.10 ], [ 1.52, 11.10, 3.10 ], [ 1.52, 13.50, 3.10 ],
        [ 1.52, 15.89, 3.10 ], [ 1.52, 18.29, 3.10 ], [ 1.52, 20.69, 3.10 ], [ 1.52, 23.01, 3.10 ],
        [ 4.56, 1.52, 2.97 ], [ 4.56, 3.91, 2.97 ], [ 4.56, 6.31, 2.97 ], [ 4.56, 8.71, 2.97 ],
        [ 4.56, 11.10, 2.97 ], [ 4.56, 13.50, 2.97 ], [ 4.56, 15.89, 2.97 ], [ 4.56, 18.29, 2.97 ],
        [ 4.56, 20.69, 2.97 ], [ 4.56, 23.08, 2.97 ], [ 7.27, 0.97, 1.85 ], [ 7.26, 3.49, 1.85 ],
        [ 7.27, 6.00, 1.85 ], [ 7.27, 8.52, 1.85 ], [ 7.26, 11.04, 1.85 ], [ 7.26, 13.56, 1.85 ],
        [ 7.27, 16.08, 1.85 ], [ 7.27, 18.59, 1.85 ], [ 7.26, 21.11, 1.85 ], [ 7.27, 23.63, 1.85 ]
    ])
    
    # Convert coordinate order: from x,y,z to y,z,x and add offset
    converted_points_dp = np.zeros_like(contact_points_DP_initial, dtype=np.float64)
    for i, point in enumerate(contact_points_DP_initial):
        # Convert to meters and adjust coordinates
        converted_points_dp[i] = [
            (point[1] + 3.11) / 1000.0,  # y coordinate
            (point[2] + 6.82) / 1000.0,  # z coordinate
            point[0] / 1000.0             # x coordinate
        ]
    
    converted_points_ip = np.zeros_like(contact_points_IP_initial, dtype=np.float64)
    for i, point in enumerate(contact_points_IP_initial):
        # Convert to meters and adjust coordinates
        converted_points_ip[i] = [
            (point[1] + 4.6) / 1000.0,   # y coordinate
            (point[2] + 8.4) / 1000.0,   # z coordinate
            point[0] / 1000.0            # x coordinate
        ]
    
    # Load URDF model
    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()

        # # Print joint angle names
        # print("List of joint angle names:")
        # for i in range(model.nq):
        #     # Get joint name corresponding to joint ID
        #     joint_name = model.names[i]
        #     print(f"Joint {i}: {joint_name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load URDF model: {str(e)}")
    
    # Ensure number of joint angles matches
    if len(joint_angles) != model.nq:
        raise ValueError(f"Mismatched number of joint angles: input {len(joint_angles)}, required {model.nq}")
    
    # Convert joint angles to Pinocchio vector
    q = np.array(joint_angles, dtype=np.float64)
    
    # Calculate forward kinematics
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # Define tactile sensor groups
    group_data = [
        {"link_name": "l_thumb8", "contact_points": converted_points_dp, "num_points": 72, "type": "dp"},
        {"link_name": "l_index8", "contact_points": converted_points_dp, "num_points": 72, "type": "dp"},
        {"link_name": "l_mid8", "contact_points": converted_points_dp, "num_points": 72, "type": "dp"},
        {"link_name": "l_ring8", "contact_points": converted_points_dp, "num_points": 72, "type": "dp"},
        {"link_name": "l_little8", "contact_points": converted_points_dp, "num_points": 72, "type": "dp"},
        {"link_name": "l_thumb4", "contact_points": converted_points_ip, "num_points": 60, "type": "ip"},
        {"link_name": "l_index4", "contact_points": converted_points_ip, "num_points": 60, "type": "ip"},
        {"link_name": "l_mid4", "contact_points": converted_points_ip, "num_points": 60, "type": "ip"},
        {"link_name": "l_ring4", "contact_points": converted_points_ip, "num_points": 60, "type": "ip"},
        {"link_name": "l_little4", "contact_points": converted_points_ip, "num_points": 60, "type": "ip"},
    ]
    
    # Get hand base coordinate system
    try:
        hand_base_id = model.getFrameId("l_handbase")
        base_link_id = model.getFrameId("base_link")   # Robot base coordinate system
    except:
        raise ValueError("Cannot find hand base coordinate system: l_handbase")
    
    # Store results
    all_contact_points = []  # 3D positions of all contact points relative to tactile sensor origin
    all_sensor_poses = []    # 6D poses of tactile sensor origin relative to hand base (x,y,z,rx,ry,rz)
    all_forces = []          # 3D forces at each contact point
    all_contact_handbase = []     # 12-14: 3D positions of contact points relative to hand base [Added]
    all_contact_baselink = []     # 15-17: 3D positions of contact points relative to base link [Added]
    # Iterate over all tactile sensor groups
    for finger_idx, group in enumerate(group_data):
        try:
            # Get sensor coordinate system ID
            sensor_id = model.getFrameId(group["link_name"])
        except:
            raise ValueError(f"Cannot find sensor coordinate system: {group['link_name']}")
        
        # Get sensor pose relative to base
        sensor_pose = data.oMf[sensor_id]
        
        # Get hand base pose relative to base
        hand_base_pose = data.oMf[hand_base_id]
        
        # Calculate sensor pose relative to hand base (hand base -> sensor)
        sensor_rel_hand = hand_base_pose.inverse() * sensor_pose
        
        # Convert to 6D pose (x,y,z,rx,ry,rz)
        translation = sensor_rel_hand.translation
        translation[1]=translation[1]+ 3.11 / 1000.0
        translation[2]=translation[2]+ 6.82 / 1000.0
        rotation = pin.rpy.matrixToRpy(sensor_rel_hand.rotation)
        sensor_pose_6d = np.concatenate([translation, rotation])
        
        # Calculate column range of this finger in the image
        col_start = (finger_idx % 5) * 6
        col_end = col_start + 6
        
        # Row range depends on type (dp = distal phalanx, ip = intermediate phalanx)
        if group["type"] == "dp":  # Distal phalanx (finger tip)
            row_start, row_end = 0, 12
        else:  # Intermediate phalanx (finger pad)
            row_start, row_end = 12, 22
        
        # Add all contact point data for this group
        for point_idx in range(min(group["num_points"], len(group["contact_points"]))):
            # Calculate row and column position of this contact point in the image
            # Row index: 11 - (point_idx // 6) because original data is reversed
            # Column index: point_idx % 6
            img_row = row_start + (11 - (point_idx // 6)) if group["type"] == "dp" else row_start + (9 - (point_idx // 6))
            img_col = col_start + (point_idx % 6)
            
            # Check if image indices are valid
            if img_row < 0 or img_row >= 22 or img_col < 0 or img_col >= 30:
                force = np.zeros(3)
            else:
                # Get RGB values from image (note OpenCV uses BGR format, need conversion)
                if len(tactile_image.shape) == 3 and tactile_image.shape[2] == 3:
                    b, g, r = tactile_image[img_row, img_col]
                else:
                    # If grayscale image, fill three channels with the same value
                    gray = tactile_image[img_row, img_col]
                    r, g, b = gray, gray, gray
                
                # Denormalize to get force values
                force_x = denormalize_force(r, -15.0, 15.0)  # X-axis corresponds to red channel
                force_y = denormalize_force(g, -15.0, 15.0)  # Y-axis corresponds to green channel
                force_z = denormalize_force(b, 0.0, 25.0)    # Z-axis corresponds to blue channel
                force = np.array([force_x, force_y, force_z])
            
            # Contact point position relative to sensor origin
            contact_point = group["contact_points"][point_idx]

            # 2. Homogeneous coordinates of contact point relative to sensor (for pose transformation: position + no rotation)
            # SE3 transformation in Pinocchio requires homogeneous coordinates, no rotation of contact point relative to sensor, only position offset
            contact_pose_sensor = pin.SE3.Identity()
            contact_pose_sensor.translation = contact_point  # Contact point position relative to sensor
             # 3. Pose of contact point relative to robot base (base_link)
            # Transformation logic: base_link → sensor → contact point → extract position
            contact_pose_baselink = sensor_pose * contact_pose_sensor
            contact_rel_baselink = contact_pose_baselink.translation  # Dimensions 15-17
            # 4. Pose of contact point relative to hand base (l_handbase)
            # Transformation logic: handbase → sensor → contact point → extract position
            contact_pose_handbase = sensor_rel_hand * contact_pose_sensor
            contact_rel_handbase = contact_pose_handbase.translation  # Dimensions 12-14

            all_contact_points.append(contact_point)
            all_sensor_poses.append(sensor_pose_6d)
            all_forces.append(force)
            all_contact_handbase.append(contact_rel_handbase)  # Added
            all_contact_baselink.append(contact_rel_baselink)  # Added
    # Convert to numpy array and return
    # Convert to 18-dimensional numpy array and return
    # Each row contains: [contact_x, contact_y, contact_z, sensor_x, sensor_y, sensor_z, sensor_rx, sensor_ry, sensor_rz, force_x, force_y, force_z, contact position relative to hand base, contact position relative to base]
    contact_points = np.array(all_contact_points)
    sensor_poses = np.array(all_sensor_poses)
    forces = np.array(all_forces)
    contact_handbase = np.array(all_contact_handbase)   # (N, 3) [Added]
    contact_baselink = np.array(all_contact_baselink)   # (N, 3) [Added]
    # Concatenate three arrays column-wise to form 12-dimensional array
    result = np.hstack((sensor_poses,contact_points,  forces, contact_handbase, contact_baselink))

    return result

# Usage example
if __name__ == "__main__":
    # Sample joint angles (provide correct joint angles according to robot model in practical applications)
    sample_joint_angles = [0.0] * 56  # Assume 48 joints

    # Create sample tactile image (22×30×3)
    # Create a gradient image for testing here
    sample_image = np.zeros((22, 30, 3), dtype=np.uint8)
    for i in range(22):
        for j in range(30):
            # Create gradient effect
            sample_image[i, j] = [
                int(255 * i / 22),  # B channel (Z-axis)
                int(255 * j / 30),  # G channel (Y-axis)
                128                 # R channel (X-axis) - fixed value
            ]
    
    # URDF file path (replace with actual path)
    urdf_path = "policy_ws/src/VTFusion/src/urdf/LeftArmHandURDFV3.urdf"
    
    try:
        # Calculate tactile sensor data
        tactile_data = compute_tactile_sensor_data(sample_joint_angles, urdf_path, sample_image)
        
        # Print result information
        print(f"Calculation completed:")
        print(f"Number of contact points: {tactile_data.shape[0]}")  # Number of rows in array is number of contact points
        print(f"Data dimensions: {tactile_data.shape[1]} (expected 18 dimensions)")
        print(f"Total data dimensions: {tactile_data.shape}")
        # Decompose 12-dimensional data of first contact point
        if tactile_data.shape[0] > 0:
            first_point = tactile_data[0]
            print(f"Complete data of first contact point: {first_point}")
            print(f"  Contact point position (x,y,z): {first_point[0:3]}")
            print(f"  Sensor pose (x,y,z,rx,ry,rz): {first_point[3:9]}")
            print(f"  Force data (fx,fy,fz): {first_point[9:12]}")
    except Exception as e:
        print(f"Calculation failed: {str(e)}")