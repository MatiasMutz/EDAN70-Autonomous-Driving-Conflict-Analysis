import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pandas as pd
import zarr
import os
from dataset.visual_utils import *

folder_av = 'av'
folder_hv = 'hv'

root_av = './dataset/data_3m/'+folder_av+'/'
root_hv = './dataset/data_3m/'+folder_hv+'/'

log_ids_av = [name for name in os.listdir(root_av) if name.endswith('.zarr')]
log_ids_hv = [name for name in os.listdir(root_hv) if name.endswith('.zarr')]

print('Number of scenarios for Autonomous Vehicles: ', len(log_ids_av))
print('Number of scenarios for Human-driven Vehicles: ', len(log_ids_hv))

'''
slices: len = nb_objects + 1, slices[n] and slices[n+1] gives the start/end indices of the n-th object
maps: lanes as NumPy array
type: len = nb_objects, contains 7 numbers with the following meanings:
        -1: Static background
        0: human-driven vehicles
        1: pedestrians
        2: motorcyclists
        3: cyclists
        4: buses
        10: autonomous vehicles
timestep: timestamps in second, timestep[slices[n]: slices[n+1]] give the timestamps for the n-th object
motion: motion state, with 7 dimensions
    motion[slices[n]: slices[n+1]] gives the motion of the n-th object, the 7 features are the following variables in order:
        [x, y, vx, vy, ax, ay, yaw]
        yaw is to the x-axis, between [-pi, pi]
'''
# Use the first scenario as an example
slices, timestep, motion, type, maps = read_scenario(log_ids_av[0], root_av)

fig, ax = visualize(log_ids_av[0], root_av, other_road_users=True, direction=True)

# Load metafile for autonomous vehicles
metafile_av = pd.read_csv('./dataset/metafile_av.csv')

# Filter for intersection scenarios (crossing trajectories)
intersection_cases = metafile_av[
    ((metafile_av['start'] == 'cross') | (metafile_av['end'] == 'cross')) &
    (metafile_av['typej'] != 'Pedestrian')
]

print(f"Total number of intersection scenarios: {len(intersection_cases)}")


def get_scenario_filename(scenario_id, root_path):
    """
    Maps a scenario ID to its corresponding zarr file in the dataset.
    
    Args:
        scenario_id (str): The ID of the scenario to find
        root_path (str): Root directory containing the scenario files
        
    Returns:
        str or None: Filename if found, None if no matching file exists
    """
    # List all files in the directory
    all_files = os.listdir(root_path)
    
    # Try different possible filename formats
    possible_formats = [
        f"{int(scenario_id):02d}.zarr",  # 01.zarr
        f"{scenario_id}.zarr",           # 1.zarr
        f"scenario_{scenario_id}.zarr",  # scenario_1.zarr
        f"scenario_{int(scenario_id):02d}.zarr"  # scenario_01.zarr
    ]
    
    # Try each format
    for format in possible_formats:
        if format in all_files:
            return format
    
    # If no exact match, try to find any file containing the scenario number
    matching_files = [f for f in all_files if str(int(scenario_id)) in f and f.endswith('.zarr')]
    
    if matching_files:
        return matching_files[0]
    
    return None

def analyze_intersection_scenario(scenario_id, root_path):
    """
    Analyzes a single intersection scenario and extracts relevant features for conflict detection.
    
    Args:
        scenario_id (str): The ID of the scenario to analyze
        root_path (str): Root directory containing the scenario files
        
    Returns:
        dict or None: Dictionary containing scenario features if successful:
            - scenario_id: ID of the analyzed scenario
            - trajectory_length: Number of timesteps in the scenario
            - vehicle_count: Number of vehicles in the scenario
            - motion_data: Vehicle motion states (position, velocity, acceleration, yaw)
            - timestep: Timestamps for each motion state
            - slices: Indices marking different vehicles' data
            - type_data: Vehicle type information
        Returns None if analysis fails
    """
    filename = get_scenario_filename(scenario_id, root_path)
    if filename is None:
        print(f"Error: No matching file found for scenario {scenario_id}")
        return None
    
    try:
        slices, timestep, motion, type_data, maps = read_scenario(filename, root_path)
        
        features = {
            'scenario_id': scenario_id,
            'trajectory_length': len(timestep),
            'vehicle_count': len(type_data),
            'motion_data': motion,
            'timestep': timestep,
            'slices': slices,
            'type_data': type_data
        }
        return features
    except Exception as e:
        print(f"Error analyzing scenario {scenario_id}: {str(e)}")
        return None

def visualize_scenario(scenario_id, root_path):
    """Visualizes a single scenario with trajectory information."""
    try:
        filename = get_scenario_filename(scenario_id, root_path)
        if filename:
            fig, ax = visualize(filename, root_path, 
                              other_road_users=True, 
                              direction=True)
            plt.title(f'Scenario {scenario_id}')
            return fig, ax
        return None, None
    except Exception as e:
        print(f"Error visualizing scenario {scenario_id}: {str(e)}")
        return None, None

    """Visualizes a single scenario with trajectory information."""
    try:
        filename = get_scenario_filename(scenario_id, root_path)
        if filename:
            fig, ax = visualize(filename, root_path, 
                              other_road_users=True, 
                              direction=True)
            plt.title(f'Scenario {scenario_id}')
            return fig, ax
        return None, None
    except Exception as e:
        print(f"Error visualizing scenario {scenario_id}: {str(e)}")
        return None, None


sample_size = 5
sample_scenarios = intersection_cases['log_id'].iloc[:sample_size]

print("\nAnalyzing sample scenarios:")
for scenario_id in sample_scenarios:
    print(f"\nProcessing scenario {scenario_id}")
    features = analyze_intersection_scenario(scenario_id, root_av)
    
    if features:
        print(f"Successfully analyzed scenario:")
        print(f"- Trajectory length: {features['trajectory_length']}")
        print(f"- Number of vehicles: {features['vehicle_count']}")
    else:
        print("Analysis failed")

fig, axs = plt.subplots(1, min(5, len(sample_scenarios)), figsize=(20, 4))
if not isinstance(axs, np.ndarray):
    axs = [axs]

for i, scenario_id in enumerate(sample_scenarios[:5]):
    #_, _ = visualize_scenario(scenario_id, root_av)
    if i < len(axs):
        axs[i].set_title(f'Scenario {i+1}')

#plt.tight_layout()
#plt.show()


# Constants for conflict analysis
CONFLICT_THRESHOLDS = {
    'TTC_CRITICAL': 2.0,     # Critical Time-to-Collision (seconds)
    'PET_CRITICAL': 1.0,     # Critical Post-Encroachment Time (seconds)
    'ANGLE_THRESHOLD': {
        'CROSSING': 45,      # Minimum angle for crossing conflict (degrees)
        'HEAD_ON': 150       # Minimum angle for head-on conflict (degrees)
    },
    'DISTANCE_CRITICAL': 5.0 # Critical distance (meters)
}

class ConflictType:
    """Possible conflict types in autonomous driving scenarios"""
    CROSSING = "crossing"           # Trajectories intersect at an angle
    REAR_END = "rear_end"          # Following vehicle conflicts with leading vehicle
    HEAD_ON = "head_on"            # Vehicles approaching from opposite directions
    MERGING = "merging"            # Vehicle merging into traffic
    NO_CONFLICT = "no_conflict"    # No conflict detected

def calculate_time_to_collision(ego_motion, other_motion):
    """
    Calculates Time-to-Collision (TTC) between two vehicles
    
    Args:
        ego_motion: Motion data for ego vehicle [x, y, vx, vy, ...]
        other_motion: Motion data for other vehicle [x, y, vx, vy, ...]
    
    Returns:
        float: Minimum TTC value or infinity if no collision course
    """
    # Interpolate trajectories to common length
    target_length = 100
    
    # Create normalized time arrays for interpolation
    t_ego = np.linspace(0, 1, len(ego_motion))
    t_other = np.linspace(0, 1, len(other_motion))
    t_common = np.linspace(0, 1, target_length)
    
    # Initialize interpolated arrays
    ego_interp = np.zeros((target_length, ego_motion.shape[1]))
    other_interp = np.zeros((target_length, other_motion.shape[1]))
    
    # Interpolate each component
    for i in range(ego_motion.shape[1]):
        ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
        other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])
    
    # Extract positions and velocities from interpolated data
    ego_pos = ego_interp[:, :2]    # [x, y]
    ego_vel = ego_interp[:, 2:4]   # [vx, vy]
    other_pos = other_interp[:, :2]
    other_vel = other_interp[:, 2:4]
    
    # Calculate relative velocity and distance
    rel_pos = ego_pos - other_pos
    rel_vel = ego_vel - other_vel
    
    # Calculate TTC
    distance = np.linalg.norm(rel_pos, axis=1)
    rel_speed = np.linalg.norm(rel_vel, axis=1)
    
    # Avoid division by zero and negative relative speeds
    valid_idx = (rel_speed > 0.1)
    if not np.any(valid_idx):
        return float('inf')
    
    ttc = distance[valid_idx] / rel_speed[valid_idx]
    return np.min(ttc) if len(ttc) > 0 else float('inf')

def calculate_post_encroachment_time(ego_motion, other_motion, conflict_point=None):
    """
    Calculates Post-Encroachment Time (PET) at the conflict point
    
    Args:
        ego_motion: Motion data for ego vehicle
        other_motion: Motion data for other vehicle
        conflict_point: Optional pre-defined conflict point
    
    Returns:
        float: PET value in seconds
    """
    # Interpolate trajectories to common length
    target_length = 100
    
    # Create normalized time arrays for interpolation
    t_ego = np.linspace(0, 1, len(ego_motion))
    t_other = np.linspace(0, 1, len(other_motion))
    t_common = np.linspace(0, 1, target_length)
    
    # Initialize interpolated arrays
    ego_interp = np.zeros((target_length, ego_motion.shape[1]))
    other_interp = np.zeros((target_length, other_motion.shape[1]))
    
    # Interpolate each component
    for i in range(ego_motion.shape[1]):
        ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
        other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])
    
    if conflict_point is None:
        # Estimate conflict point as the closest point between trajectories
        ego_pos = ego_interp[:, :2]
        other_pos = other_interp[:, :2]
        distances = np.linalg.norm(ego_pos[:, np.newaxis] - other_pos, axis=2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        conflict_point = (ego_pos[min_idx[0]] + other_pos[min_idx[1]]) / 2
    
    # Calculate arrival times at conflict point using interpolated data
    ego_times = calculate_arrival_time(ego_interp, conflict_point)
    other_times = calculate_arrival_time(other_interp, conflict_point)
    
    # PET is the absolute difference between arrival times
    return abs(ego_times - other_times)

def calculate_arrival_time(motion, point):
    """
    Calculates time of arrival to a specific point
    
    Args:
        motion: Vehicle motion data
        point: Target point coordinates
    
    Returns:
        float: Estimated arrival time in seconds
    """
    positions = motion[:, :2]
    velocities = motion[:, 2:4]
    
    # Find closest point to conflict point
    distances = np.linalg.norm(positions - point, axis=1)
    closest_idx = np.argmin(distances)
    
    # Calculate time based on distance and speed
    speed = np.linalg.norm(velocities[closest_idx])
    if speed < 0.1:  # Almost stopped
        return float('inf')
    
    return distances[closest_idx] / speed

def classify_conflict_type(ego_motion, other_motion):
    """
    Classifies the type of conflict based on vehicle trajectories
    
    Args:
        ego_motion: Motion data for ego vehicle
        other_motion: Motion data for other vehicle
    
    Returns:
        str: Type of conflict (CROSSING, REAR_END, HEAD_ON, MERGING)
    """
    # Calculate angle between trajectories
    ego_direction = ego_motion[-1, 2:4] - ego_motion[0, 2:4]
    other_direction = other_motion[-1, 2:4] - other_motion[0, 2:4]
    
    angle = np.arccos(np.dot(ego_direction, other_direction) / 
                     (np.linalg.norm(ego_direction) * np.linalg.norm(other_direction)))
    angle_deg = np.degrees(angle)
    
    # Classify based on angle
    if angle_deg > CONFLICT_THRESHOLDS['ANGLE_THRESHOLD']['HEAD_ON']:
        return ConflictType.HEAD_ON
    elif angle_deg > CONFLICT_THRESHOLDS['ANGLE_THRESHOLD']['CROSSING']:
        return ConflictType.CROSSING
    else:
        # Determine if rear-end or merging
        relative_position = other_motion[0, :2] - ego_motion[0, :2]
        heading_difference = abs(ego_motion[0, 6] - other_motion[0, 6])
        
        if heading_difference < np.pi/4:  # Similar directions
            return ConflictType.REAR_END
        else:
            return ConflictType.MERGING

def assess_risk_level(ttc, pet, distance):
    """
    Evaluates risk level based on multiple metrics
    
    Args:
        ttc: Time-to-Collision value
        pet: Post-Encroachment Time value
        distance: Minimum distance between vehicles
    
    Returns:
        str: Risk level (HIGH, MEDIUM, LOW)
    """
    if ttc < CONFLICT_THRESHOLDS['TTC_CRITICAL'] or \
       pet < CONFLICT_THRESHOLDS['PET_CRITICAL'] or \
       distance < CONFLICT_THRESHOLDS['DISTANCE_CRITICAL']:
        return "HIGH"
    elif ttc < CONFLICT_THRESHOLDS['TTC_CRITICAL'] * 2 or \
         pet < CONFLICT_THRESHOLDS['PET_CRITICAL'] * 2 or \
         distance < CONFLICT_THRESHOLDS['DISTANCE_CRITICAL'] * 2:
        return "MEDIUM"
    else:
        return "LOW"

def analyze_scenario_conflicts(scenario_data):
    """
    Complete conflict analysis for a scenario
    
    Args:
        scenario_data: Dictionary containing scenario information
    
    Returns:
        dict: Analysis results including conflict type, metrics, and risk level
    """
    # Extract motion data for ego and other vehicles
    ego_motion = scenario_data['motion_data'][scenario_data['slices'][0]:scenario_data['slices'][1]]
    other_motion = scenario_data['motion_data'][scenario_data['slices'][1]:scenario_data['slices'][2]]
    
    # Interpolate trajectories to common length for minimum distance calculation
    target_length = 100
    t_ego = np.linspace(0, 1, len(ego_motion))
    t_other = np.linspace(0, 1, len(other_motion))
    t_common = np.linspace(0, 1, target_length)
    
    ego_interp = np.zeros((target_length, ego_motion.shape[1]))
    other_interp = np.zeros((target_length, other_motion.shape[1]))
    
    for i in range(ego_motion.shape[1]):
        ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
        other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])
    
    # Calculate main metrics using interpolated data
    ttc = calculate_time_to_collision(ego_motion, other_motion)
    pet = calculate_post_encroachment_time(ego_motion, other_motion)
    min_distance = np.min(np.linalg.norm(ego_interp[:, :2] - other_interp[:, :2], axis=1))
    
    # Classify conflict type (using interpolated data)
    conflict_type = classify_conflict_type(ego_interp, other_interp)
    
    # Evaluate risk level
    risk_level = assess_risk_level(ttc, pet, min_distance)
    
    return {
        'scenario_id': scenario_data['scenario_id'],
        'conflict_type': conflict_type,
        'metrics': {
            'TTC': ttc,
            'PET': pet,
            'min_distance': min_distance,
            'risk_level': risk_level
        },
        'timestamp': scenario_data['timestep']
    }

def analyze_all_scenarios(intersection_cases, root_path):
    """
    Analyzes all scenarios and generates a report
    
    Args:
        intersection_cases: DataFrame containing scenario metadata
        root_path: Root directory containing scenario files
    
    Returns:
        list: Analysis results for all scenarios
    """
    conflict_analyses = []
    
    for _, case in intersection_cases.iterrows():
        scenario_id = case['log_id']
        features = analyze_intersection_scenario(scenario_id, root_path)
        
        if features:
            conflict_analysis = analyze_scenario_conflicts(features)
            conflict_analyses.append(conflict_analysis)
            
            print(f"\nAnalysis for Scenario {scenario_id}:")
            print(f"Conflict Type: {conflict_analysis['conflict_type']}")
            print(f"Risk Level: {conflict_analysis['metrics']['risk_level']}")
            print(f"TTC: {conflict_analysis['metrics']['TTC']:.2f} seconds")
            print(f"PET: {conflict_analysis['metrics']['PET']:.2f} seconds")
            print(f"Minimum Distance: {conflict_analysis['metrics']['min_distance']:.2f} meters")
    
    return conflict_analyses


import pandas as pd
from IPython.display import display, FileLink

print("\nAnalyzing conflicts in scenarios...")
conflict_analyses = analyze_all_scenarios(intersection_cases[:10], root_av)

#save the conflict analyses to a csv file
flat_conflict_analyses = []
for entry in conflict_analyses:
    flat_entry = {
        'Scenario ID': entry['scenario_id'],
        'Conflict Type': entry['conflict_type'],
        'Risk Level': entry['metrics']['risk_level'],
        'TTC (s)': entry['metrics']['TTC'],
        'PET (s)': entry['metrics']['PET'],
        'Minimum Distance (m)': entry['metrics']['min_distance'],
        'Timestamp': entry.get('timestamp', None)
    }
    flat_conflict_analyses.append(flat_entry)
    
conflict_analyses_df = pd.DataFrame(flat_conflict_analyses)
conflict_analyses_df.to_csv('conflict_analyses.csv', index=False)
print("Conflict analyses saved to")
display(FileLink('conflict_analyses.csv'))

conflict_analyses_df



def prepare_logistic_regression_data(scenario_data):
    """
    Prepares scenario data for logistic regression by extracting relevant features.
    
    This function processes trajectory data to create a feature vector for collision prediction.
    It uses multiple metrics to determine if a scenario represents a collision:
    - Minimum distance between vehicles
    - Time to Collision (TTC)
    - Post-Encroachment Time (PET)
    - Relative velocity
    
    Thresholds for collision classification:
    - Distance: < 0.5m (critical distance)
    - TTC: < 0.3s AND distance < 2.0m (critical time to collision)
    - PET: < 0.1s AND distance < 2.0m (critical post-encroachment time)
    - Velocity: > 10.0 m/s AND distance < 1.0m (high speed and critical distance)
    
    Args:
        scenario_data (dict): Dictionary containing scenario features including:
            - motion_data: Vehicle motion states
            - slices: Indices marking different vehicles' data
            - timestep: Timestamps for motion data
            
    Returns:
        tuple: (features, collision_label) where:
            - features: numpy array of extracted features
            - collision_label: 1 for collision, 0 for no collision
    """
    def extract_features(motion_data, slices, timestep):
        """
        Extracts relevant features for collision prediction.
        
        Features extracted:
        1. Minimum distance between vehicles
        2. Average relative velocity
        3. Minimum time to intersection
        4. Average yaw angle difference
        5. Average speed difference
        6. Time to closest approach
        7. Average relative acceleration
        
        Args:
            motion_data: Raw motion data for all vehicles
            slices: Indices marking different vehicles' data
            timestep: Timestamps for motion data
            
        Returns:
            numpy.array: Feature vector containing the 7 features listed above
        """
        # Get ego vehicle (AV) and other vehicle trajectories
        ego_motion = motion_data[slices[0]:slices[1]]
        other_motion = motion_data[slices[1]:slices[2]]
        
        # Interpolate trajectories to common length for consistent analysis
        target_length = 100  # Fixed number of points for interpolation
        t_ego = np.linspace(0, 1, len(ego_motion))
        t_other = np.linspace(0, 1, len(other_motion))
        t_common = np.linspace(0, 1, target_length)
        
        # Initialize interpolated arrays
        ego_interp = np.zeros((target_length, ego_motion.shape[1]))
        other_interp = np.zeros((target_length, other_motion.shape[1]))
        
        # Interpolate each component of the trajectories
        for i in range(ego_motion.shape[1]):
            ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
            other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])
        
        # Calculate features using interpolated trajectories
        relative_distance = np.linalg.norm(ego_interp[:, :2] - other_interp[:, :2], axis=1)
        relative_velocity = np.linalg.norm(ego_interp[:, 2:4] - other_interp[:, 2:4], axis=1)
        time_to_intersection = relative_distance / (relative_velocity + 1e-6)  # Avoid division by zero
        
        # Calculate yaw angle difference (normalized to [0, pi])
        yaw_diff = np.abs(ego_interp[:, 6] - other_interp[:, 6])
        yaw_diff = np.minimum(yaw_diff, 2*np.pi - yaw_diff)
        
        # Calculate speed difference
        ego_speed = np.linalg.norm(ego_interp[:, 2:4], axis=1)
        other_speed = np.linalg.norm(other_interp[:, 2:4], axis=1)
        speed_diff = np.abs(ego_speed - other_speed)
        
        # Calculate time to closest approach
        min_dist_idx = np.argmin(relative_distance)
        time_to_closest = t_common[min_dist_idx]
        
        # Calculate relative acceleration
        ego_acc = np.linalg.norm(ego_interp[:, 4:6], axis=1)
        other_acc = np.linalg.norm(other_interp[:, 4:6], axis=1)
        rel_acc = np.abs(ego_acc - other_acc)
        
        # Combine features into feature vector
        features = np.array([
            relative_distance.min(),          # Minimum distance
            relative_velocity.mean(),         # Average relative velocity
            time_to_intersection.min(),       # Minimum time to intersection
            yaw_diff.mean(),                  # Average yaw angle difference
            speed_diff.mean(),                # Average speed difference
            time_to_closest,                  # Time to closest approach
            rel_acc.mean()                    # Average relative acceleration
        ])
        
        return features

    # Extract features for all timesteps
    features = extract_features(
        scenario_data['motion_data'],
        scenario_data['slices'],
        scenario_data['timestep']
    )
    
    # Interpolate trajectories for collision detection
    ego_motion = scenario_data['motion_data'][scenario_data['slices'][0]:scenario_data['slices'][1]]
    other_motion = scenario_data['motion_data'][scenario_data['slices'][1]:scenario_data['slices'][2]]
    
    target_length = 100
    t_ego = np.linspace(0, 1, len(ego_motion))
    t_other = np.linspace(0, 1, len(other_motion))
    t_common = np.linspace(0, 1, target_length)
    
    ego_interp = np.zeros((target_length, ego_motion.shape[1]))
    other_interp = np.zeros((target_length, other_motion.shape[1]))
    
    for i in range(ego_motion.shape[1]):
        ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
        other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])
    
    # Calculate collision metrics
    min_distance = np.min(np.linalg.norm(ego_interp[:, :2] - other_interp[:, :2], axis=1))
    ttc = calculate_time_to_collision(ego_motion, other_motion)
    pet = calculate_post_encroachment_time(ego_motion, other_motion)
    relative_velocity = np.linalg.norm(ego_interp[:, 2:4] - other_interp[:, 2:4], axis=1).mean()
    
    # Determine collision label based on multiple criteria
    # A scenario is considered a collision if ANY of these conditions are met:
    collision_label = 1 if (
        min_distance < 0.5 or  # Critical distance threshold
        (ttc < 0.3 and min_distance < 2.0) or  # Critical TTC with small distance
        (pet < 0.1 and min_distance < 2.0) or  # Critical PET with small distance
        (relative_velocity > 10.0 and min_distance < 1.0)  # High speed with critical distance
    ) else 0
    
    return features, collision_label


def get_binary_features_and_labels(conflict_analyses):
    X_bin = []
    y_bin = []
    for entry in conflict_analyses:
        features, label = prepare_logistic_regression_data(entry)
        X_bin.append(features)
        y_bin.append(label)
    return np.array(X_bin), np.array(y_bin)


def train_logistic_regression_binary(intersection_cases, root_path):
    """
    Trains a binary logistic regression model (collision vs. no collision) using SMOTE for class balancing.
    Evaluates with ROC and precision-recall curves.

    Returns:
        model, scaler, X_test, y_test, y_pred, y_prob, test_data
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
    )
    from imblearn.over_sampling import SMOTE
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---------------------------
    # Step 1: Feature Extraction
    # ---------------------------
    X = []
    y = []
    ids = []

    for _, case in intersection_cases.iterrows():
        scenario_id = case['log_id']
        features = analyze_intersection_scenario(scenario_id, root_path)
        if features:
            scenario_features, collision_label = prepare_logistic_regression_data(features)
            X.append(scenario_features)
            y.append(collision_label)  # 1: collision, 0: no collision
            ids.append(scenario_id)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("No valid data extracted. Check feature extraction functions.")

    print("\nClass distribution in dataset:")
    print(f"No Collision (0): {np.sum(y == 0)}")
    print(f"Collision     (1): {np.sum(y == 1)}")

    if np.sum(y == 1) < 10 or np.sum(y == 0) < 10:
        print("\n⚠️ Warning: Very imbalanced dataset. Proceeding anyway.")

    # ---------------------------
    # Step 2: Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(y_train)}")
    print(f"Test samples:     {len(y_test)}")

    # ---------------------------
    # Step 3: SMOTE Balancing
    # ---------------------------
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(f"No Collision (0): {np.sum(y_train_res == 0)}")
    print(f"Collision     (1): {np.sum(y_train_res == 1)}")

    # ---------------------------
    # Step 4: Scale Features
    # ---------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # Step 5: Train Model
    # ---------------------------
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train_res)

    # ---------------------------
    # Step 6: Evaluation
    # ---------------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['No Collision', 'Collision']))

    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = ['No Collision', 'Collision']
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print("\nConfusion Matrix:")
    print(conf_df)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    ap = average_precision_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # ---------------------------
    # Step 7: Return Results
    # ---------------------------
    test_data = pd.DataFrame({'log_id': ids_test})

    return model, scaler, X_test, y_test, y_pred, y_prob, test_data



def predict_collision(model, scaler, scenario_data):
    """
    Predicts collision (1) or no collision (0) for a given scenario.
    Uses a trained logistic regression model, or falls back to rule-based logic if model/scaler is unavailable.

    Returns:
        prediction (int): 1 for collision, 0 for no collision
        probability (float): predicted probability of collision
    """
    import numpy as np

    # ---------- Fallback Rule-Based Prediction ----------
    def rule_based_prediction(data):
        ego_motion = data['motion_data'][data['slices'][0]:data['slices'][1]]
        other_motion = data['motion_data'][data['slices'][1]:data['slices'][2]]

        # Interpolate trajectories
        target_length = 100
        t_ego = np.linspace(0, 1, len(ego_motion))
        t_other = np.linspace(0, 1, len(other_motion))
        t_common = np.linspace(0, 1, target_length)

        ego_interp = np.zeros((target_length, ego_motion.shape[1]))
        other_interp = np.zeros((target_length, other_motion.shape[1]))
        for i in range(ego_motion.shape[1]):
            ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
            other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])

        # Extract key indicators
        min_distance = np.min(np.linalg.norm(ego_interp[:, :2] - other_interp[:, :2], axis=1))
        ttc = calculate_time_to_collision(ego_motion, other_motion)
        pet = calculate_post_encroachment_time(ego_motion, other_motion)
        relative_velocity = np.linalg.norm(ego_interp[:, 2:4] - other_interp[:, 2:4], axis=1).mean()

        # Apply binary collision rules
        is_collision = (
            min_distance < 0.5 or
            (ttc < 0.3 and min_distance < 2.0) or
            (pet < 0.1 and min_distance < 2.0) or
            (relative_velocity > 10.0 and min_distance < 1.0)
        )
        return int(is_collision), 1.0  # Fallback always returns certainty

    # ---------- Model-Based Prediction ----------
    if model is not None and scaler is not None:
        try:
            features, _ = prepare_logistic_regression_data(scenario_data)
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]  # Probability of class '1' (collision)
            return prediction, probability
        except Exception as e:
            print(f"\nError using model: {str(e)}")
            print("Falling back to rule-based logic...\n")

    # Fallback route if model is None or failed
    return rule_based_prediction(scenario_data)


from IPython.display import FileLink
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np

# Step 1: Collect scenarios with available data
print("\nSplitting data into training and testing sets...")
available_scenarios = []
for _, case in intersection_cases.iterrows():
    scenario_id = case['log_id']
    filename = get_scenario_filename(scenario_id, root_av)
    if filename is not None:
        available_scenarios.append(case)

print(f"\nTotal available scenarios: {len(available_scenarios)}")
all_data = pd.DataFrame(available_scenarios)

# Step 2: Train logistic regression model
print("\nTraining logistic regression model...")
model, scaler, X_test, y_test, y_pred, y_prob, test_data = train_logistic_regression_binary(all_data, root_av)
print("Model training complete.")

# Step 3: Predict test scenarios
print("\nPredicting on test scenarios...")

scenario_ids = []
risk_labels = []
confidences = []

for _, case in test_data.iterrows():
    scenario_id = case['log_id']
    test_scenario = analyze_intersection_scenario(scenario_id, root_av)

    if test_scenario:
        try:
            features, _ = prepare_logistic_regression_data(test_scenario)
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]  # Probability of collision
        except Exception as e:
            print(f"⚠️ Model error on scenario {scenario_id}: {str(e)}")
            prediction, probability = predict_collision(None, None, test_scenario)

        risk_label = 'collision' if prediction == 1 else 'no collision'
        scenario_ids.append(scenario_id)
        risk_labels.append(risk_label)
        confidences.append(probability)

# Step 4: Create results table
results_df = pd.DataFrame({
    'Scenario ID': scenario_ids,
    'Risk Label': risk_labels,
    'Confidence': confidences
})

print("\nCounts of each risk label in test predictions:")
print(results_df['Risk Label'].value_counts())

# Save to CSV
results_df.to_csv('all_test_predictions.csv', index=False)
print("\nPrediction results saved as: all_test_predictions.csv")
display(FileLink('all_test_predictions.csv'))

results_df['Confidence'].hist(bins=50)
plt.title("Distribution of Predicted Collision Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.show()

# Display the DataFrame
results_df


def visualize_scenario_with_prediction(scenario_id, root_av, model, scaler, conflict_map):
    import matplotlib.pyplot as plt
    import numpy as np

    scenario_data = analyze_intersection_scenario(scenario_id, root_av)
    if not scenario_data or 'motion_data' not in scenario_data:
        print(f"Scenario {scenario_id}: No motion data available, skipping.")
        return

    features, label = prepare_logistic_regression_data(scenario_data)
    prediction, prob = predict_collision(model, scaler, scenario_data)

    # Extract ego and other vehicle trajectories
    ego_motion = scenario_data['motion_data'][scenario_data['slices'][0]:scenario_data['slices'][1]]
    other_motion = scenario_data['motion_data'][scenario_data['slices'][1]:scenario_data['slices'][2]]

    # Interpolate to common length for visualization
    target_length = 100
    t_ego = np.linspace(0, 1, len(ego_motion))
    t_other = np.linspace(0, 1, len(other_motion))
    t_common = np.linspace(0, 1, target_length)
    ego_interp = np.zeros((target_length, ego_motion.shape[1]))
    other_interp = np.zeros((target_length, other_motion.shape[1]))
    for i in range(ego_motion.shape[1]):
        ego_interp[:, i] = np.interp(t_common, t_ego, ego_motion[:, i])
        other_interp[:, i] = np.interp(t_common, t_other, other_motion[:, i])

    # Calculate the conflict point (closest approach)
    distances = np.linalg.norm(ego_interp[:, :2] - other_interp[:, :2], axis=1)
    min_idx = np.argmin(distances)
    conflict_point = (ego_interp[min_idx, :2] + other_interp[min_idx, :2]) / 2

    # Plot the trajectories and conflict point
    plt.figure(figsize=(6, 6))
    plt.plot(ego_interp[:, 0], ego_interp[:, 1], label='Ego Vehicle', color='blue')
    plt.plot(other_interp[:, 0], other_interp[:, 1], label='Other Vehicle', color='orange')
    plt.scatter(conflict_point[0], conflict_point[1], s=200, facecolors='none', edgecolors='red', linewidths=2, label='Conflict Point')
    plt.title(
        f"Scenario {scenario_id}\n"
        f"Actual: {conflict_map[label]} | Predicted: {conflict_map[prediction]} (Prob: {prob:.2f})"
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.axis('equal')
    plt.show()

    print(f"Scenario {scenario_id}:")
    print(f"  Actual: {conflict_map[label]}")
    print(f"  Predicted: {conflict_map[prediction]} (Probability: {prob:.3f})")


import matplotlib.pyplot as plt
import numpy as np

# --- Select a scenario to visualize ---
scenario_idx = np.random.randint(0, len(test_data))
scenario_id = test_data.iloc[scenario_idx]['log_id']
scenario_data = analyze_intersection_scenario(scenario_id, root_av)

if scenario_data and 'motion_data' in scenario_data:
    features, label = prepare_logistic_regression_data(scenario_data)
    prediction, prob = predict_collision(model, scaler, scenario_data)
    conflict_map = {0: 'No Conflict', 1: 'Conflict'}
    feature_names = [
        "Min Distance", "Avg Rel Velocity", "Min Time to Intersection",
        "Avg Yaw Angle Diff", "Avg Speed Diff", "Time to Closest Approach", "Avg Rel Acc"
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Model Input Features
    axs[0].bar(feature_names, features)
    axs[0].set_title("Model Input Features")
    axs[0].set_ylabel("Feature Value")
    axs[0].tick_params(axis='x', rotation=45)

    # 2. Actual vs. Predicted
    axs[1].bar(['Actual', 'Predicted'], [label, prediction], color=['blue', 'orange'])
    axs[1].set_title("Actual vs. Predicted")
    axs[1].set_ylabel("Class (0=No Conflict, 1=Conflict)")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['Actual', 'Predicted'])

    plt.tight_layout()
    plt.show()

    # Now show the scenario visualization as a separate plot
    if 'visualize_scenario' in globals():
        visualize_scenario(scenario_id, root_av)
    else:
        print("No scenario visualization function found.")

    print(f"Scenario {scenario_id}:")
    print(f"  Actual: {conflict_map[label]}")
    print(f"  Predicted: {conflict_map[prediction]} (Probability: {prob:.3f})")
else:
    print(f"Scenario {scenario_id}: No motion data available, skipping.")

conflict_map = {0: 'No Conflict', 1: 'Conflict'}
visualize_scenario_with_prediction(scenario_id, root_av, model, scaler, conflict_map)
