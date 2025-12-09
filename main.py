import cv2
import sys
# Compatibility shim: ensure google.protobuf.message_factory provides
# a GetMessageClass function expected by MediaPipe across protobuf versions.
try:
    from google.protobuf import message_factory as _mf
    if not hasattr(_mf, 'GetMessageClass'):
        def GetMessageClass(descriptor):
            return _mf.MessageFactory().GetPrototype(descriptor)
        _mf.GetMessageClass = GetMessageClass
except Exception:
    # If anything fails here, we'll let the normal import error surface later.
    pass
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time

# --- KONFIGURASI UTAMA ---
OSC_IP = "127.0.0.1"
OSC_PORT = 39539
WEBCAM_ID = 0
TARGET_FPS = 30

# --- KONFIGURASI SMOOTHING ---
HEAD_MIN_CUTOFF = 0.03  # Slightly higher for more stable face (less responsive to noise)
HEAD_BETA       = 1.5   # Stronger dampening for less exaggerated movement
FINGER_MIN_CUTOFF = 0.8 # Higher = smoother, less twitchy fingers
FINGER_BETA       = 1.2 # More dampening
EYE_MIN_CUTOFF = 0.03   # Smoother eye tracking
EYE_BETA = 1.0

# --- DATA KALIBRASI ---
ARM_INVERT_X, ARM_INVERT_Y, ARM_INVERT_Z = 1.0, 1.0, 1.0
ARM_GAIN_XY, ARM_GAIN_Z = 0.8, 0.35  # Reduced from 1.2, 0.5 = smaller arm movements
FINGER_SENSITIVITY = 0.9  # Reduced from 1.3 = less dramatic finger curls
NECK_RATIO = 0.4  # Reduced from 0.5 = less exaggerated neck tilt
PITCH_CORRECTION_FACTOR = 0.01  # Reduced from 0.015 = subtler eye pitch adjustment
EYE_Y_OFFSET = 0.015  # Reduced from 0.02
EAR_THRESH_CLOSE, EAR_THRESH_OPEN = 0.15, 0.25

# --- HELPER FUNCTIONS ---
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

def get_finger_quat(angle, axis_idx):
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    if axis_idx == 0:   return [s, 0, 0, c]
    elif axis_idx == 1: return [0, s, 0, c]
    elif axis_idx == 2: return [0, 0, s, c]
    return [0, 0, 0, 1]

def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1]
    v_curr = v_curr / norm
    v_rest = np.array(rest_vector)
    v_rest = v_rest / np.linalg.norm(v_rest)
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    axis = axis / axis_len
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def get_relative_iris(face_landmarks, iris_idx, inner_idx, outer_idx, img_w, img_h):
    iris = np.array([face_landmarks.landmark[iris_idx].x * img_w, face_landmarks.landmark[iris_idx].y * img_h])
    inner = np.array([face_landmarks.landmark[inner_idx].x * img_w, face_landmarks.landmark[inner_idx].y * img_h])
    outer = np.array([face_landmarks.landmark[outer_idx].x * img_w, face_landmarks.landmark[outer_idx].y * img_h])
    eye_width = np.linalg.norm(outer - inner)
    eye_vec = outer - inner
    eye_vec_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-6)
    iris_vec = iris - inner
    proj_x = np.dot(iris_vec, eye_vec_norm)
    norm_x = (proj_x / eye_width) * 2.0 - 1.0
    cross_prod = (eye_vec[0] * (iris[1] - inner[1])) - (eye_vec[1] * (iris[0] - inner[0]))
    dist_y = cross_prod / eye_width
    norm_y = dist_y / (eye_width * 0.3)
    return norm_x, norm_y

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)
    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    curl = (ratio - 1.9) / (0.8 - 1.9)
    return max(0.0, min(1.0, curl)) * FINGER_SENSITIVITY

# --- ONE EURO FILTER (SMOOTHING) ---
class OneEuroFilter:
    # PERBAIKAN: Gunakan double underscore (__init__)
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    # PERBAIKAN: Gunakan double underscore (__call__)
    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

# --- INIT ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.3, 
    refine_face_landmarks=True,
    model_complexity=1
)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# --- SETUP FILTERS ---
t_start = time.time()
filter_pitch = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_yaw   = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_roll  = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_spine_roll = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_spine_yaw  = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_eye_x = OneEuroFilter(t_start, 0, min_cutoff=EYE_MIN_CUTOFF, beta=EYE_BETA)
filter_eye_y = OneEuroFilter(t_start, 0, min_cutoff=EYE_MIN_CUTOFF, beta=EYE_BETA)

filters_fingers_L = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]
filters_fingers_R = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]

# Per-bone quaternion filters to smooth bone rotations (qx,qy,qz,qw)
bone_names = [
    'LeftUpperArm','LeftLowerArm','RightUpperArm','RightLowerArm',
    'LeftUpperLeg','LeftLowerLeg','RightUpperLeg','RightLowerLeg'
]
bone_quat_filters = {}
for name in bone_names:
    bone_quat_filters[name] = [OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA) for _ in range(4)]

# Quaternion filters for head and neck bones (extra smoothing to reduce face jitter)
bone_quat_filters['Head'] = [OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA) for _ in range(4)]
bone_quat_filters['Neck'] = [OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA) for _ in range(4)]

# CONSTANTS
model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
L_IRIS_C, L_IN, L_OUT = 468, 133, 33  
R_IRIS_C, R_IN, R_OUT = 473, 362, 263 
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)] 
BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]
THUMB_AXIS_L, THUMB_AXIS_R = 1, 1
THUMB_SIGN_L, THUMB_SIGN_R = -1.0, 1.0
FINGER_AXIS_L, FINGER_AXIS_R = 2, 2
FINGER_SIGN_L, FINGER_SIGN_R = 1.0, -1.0

blink_l_state, blink_r_state = 0.0, 0.0
prev_time = 0

# --- CAMERA ---
cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

print("=== VTuber TESSELATION ONLY MODE: ON ===")

# Quick runtime check: ensure the OpenCV build being used has GUI support.
# If the user runs the system Python (which may have a headless OpenCV),
# show a clear message and exit instead of raising the low-level cv2 error.
try:
    build_info = cv2.getBuildInformation()
    if ('Win32 UI' not in build_info) and ('WIN32UI' not in build_info) and ('Cocoa' not in build_info and 'GTK' not in build_info):
        print('\nERROR: The OpenCV build in use does not include GUI support (cv2.imshow).')
        print('Run this script with the project virtualenv Python:')
        print(r'"C:\Users\Lenovo\Downloads\PCV NEW\.venv\Scripts\python.exe" main.py')
        sys.exit(1)
except Exception:
    pass

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 and (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    
    image.flags.writeable = False
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True

    # ==================================================
    # === 1. VISUALISASI (MODIFIED) ===
    # ==================================================
    if results.face_landmarks:
        # A. Gambar Jaring-Jaring (Tesselation)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # C. Gambar Iris Mata
        if results.face_landmarks.landmark:
             lm = results.face_landmarks.landmark
             cx_l, cy_l = int(lm[468].x * img_w), int(lm[468].y * img_h)
             cx_r, cy_r = int(lm[473].x * img_w), int(lm[473].y * img_h)
             cv2.circle(image, (cx_l, cy_l), 2, (0, 0, 255), -1)
             cv2.circle(image, (cx_r, cy_r), 2, (0, 0, 255), -1)

    # ==================================================
    # === LOGIKA TRACKING ===
    # ==================================================
    if results.face_landmarks:
        fl = results.face_landmarks
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        
        raw_pitch, raw_yaw, raw_roll = angles[0], angles[1], math.degrees(math.atan2(dY, dX))

        s_pitch = filter_pitch(curr_time, raw_pitch)
        s_yaw   = filter_yaw(curr_time, raw_yaw)
        s_roll  = filter_roll(curr_time, raw_roll)

        neck_pitch, neck_yaw, neck_roll = s_pitch * NECK_RATIO, s_yaw * NECK_RATIO, s_roll * NECK_RATIO
        head_pitch, head_yaw, head_roll = s_pitch - neck_pitch, s_yaw - neck_yaw, s_roll - neck_roll
        
        raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
        raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)
        if raw_ear_l < EAR_THRESH_CLOSE: blink_l_state = 1.0
        elif raw_ear_l > EAR_THRESH_OPEN: blink_l_state = 0.0
        if raw_ear_r < EAR_THRESH_CLOSE: blink_r_state = 1.0
        elif raw_ear_r > EAR_THRESH_OPEN: blink_r_state = 0.0
        if s_yaw > 20.0: blink_r_state = blink_l_state 
        elif s_yaw < -20.0: blink_l_state = blink_r_state

        lx, ly = get_relative_iris(fl, L_IRIS_C, L_IN, L_OUT, img_w, img_h)
        rx, ry = get_relative_iris(fl, R_IRIS_C, R_IN, R_OUT, img_w, img_h)
        avg_x, avg_y = (lx + rx)/2.0, ((ly + ry)/2.0) - (s_pitch * PITCH_CORRECTION_FACTOR) + EYE_Y_OFFSET
        
        if not (blink_l_state > 0.5 or blink_r_state > 0.5):
            smooth_eye_x = filter_eye_x(curr_time, avg_x)
            smooth_eye_y = filter_eye_y(curr_time, avg_y)
        else:
            smooth_eye_x = filter_eye_x.x_prev
            smooth_eye_y = filter_eye_y.x_prev
        
        mouth_dist = np.linalg.norm(np.array([fl.landmark[13].x*img_w, fl.landmark[13].y*img_h]) - np.array([fl.landmark[14].x*img_w, fl.landmark[14].y*img_h]))
        mouth_open = max(0.0, min(1.0, (mouth_dist - 5.0) * (1.0/(35.0))))

        # Compute raw quaternions, then smooth per-component to reduce jitter
        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(neck_pitch), math.radians(neck_yaw), math.radians(neck_roll))
        bf_neck = bone_quat_filters['Neck']
        nq_smooth = [bf_neck[i](curr_time, [nqx, nqy, nqz, nqw][i]) for i in range(4)]
        norm = np.linalg.norm(nq_smooth)
        if norm > 1e-6: nq_smooth = [x / norm for x in nq_smooth]
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nq_smooth[0]), float(nq_smooth[1]), float(nq_smooth[2]), float(nq_smooth[3])])
        
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(head_pitch), math.radians(head_yaw), math.radians(head_roll))
        bf_head = bone_quat_filters['Head']
        hq_smooth = [bf_head[i](curr_time, [hqx, hqy, hqz, hqw][i]) for i in range(4)]
        norm = np.linalg.norm(hq_smooth)
        if norm > 1e-6: hq_smooth = [x / norm for x in hq_smooth]
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hq_smooth[0]), float(hq_smooth[1]), float(hq_smooth[2]), float(hq_smooth[3])])
        
        client.send_message("/VMC/Ext/Root/Pos", ["Root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])
        eqx, eqy, eqz, eqw = euler_to_quaternion(math.radians(smooth_eye_y*70.0), math.radians(smooth_eye_x*70.0), 0)
        client.send_message("/VMC/Ext/Bone/Pos", ["LeftEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["RightEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])

    # 2. BODY & ARM TRACKING
    if results.pose_landmarks:
        # Gambar Pose dengan style default
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        def get_vec(idx): return [lm[idx].x, lm[idx].y, lm[idx].z]
        def to_unity_vec(mp_vec): 
            return np.array([ mp_vec[0] * ARM_INVERT_X * ARM_GAIN_XY, mp_vec[1] * ARM_INVERT_Y * ARM_GAIN_XY, mp_vec[2] * ARM_INVERT_Z * ARM_GAIN_Z ])

        l_sh, r_sh = get_vec(11), get_vec(12)
        raw_spine_roll = (l_sh[1] - r_sh[1]) * -120.0
        raw_spine_yaw  = (l_sh[2] - r_sh[2]) * -80.0
        spine_roll = filter_spine_roll(curr_time, raw_spine_roll)
        spine_yaw  = filter_spine_yaw(curr_time, raw_spine_yaw)
        sqx, sqy, sqz, sqw = euler_to_quaternion(0, math.radians(spine_yaw), math.radians(spine_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Spine", 0.0, 0.0, 0.0, float(sqx), float(sqy), float(sqz), float(sqw)])

        if lm[11].visibility > 0.5 and lm[13].visibility > 0.5:
                start, end = to_unity_vec(get_vec(11)), to_unity_vec(get_vec(13))
                q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                # Smooth quaternion per-component
                bf = bone_quat_filters['LeftUpperArm']
                q_lu_s = [bf[i](curr_time, q_lu[i]) for i in range(4)]
                # normalize
                norm = np.linalg.norm(q_lu_s)
                if norm > 1e-6: q_lu_s = [x / norm for x in q_lu_s]
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu_s[0]), float(q_lu_s[1]), float(q_lu_s[2]), float(q_lu_s[3])])
                if lm[15].visibility > 0.5:
                    start, end = to_unity_vec(get_vec(13)), to_unity_vec(get_vec(15))
                    q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                    bf = bone_quat_filters['LeftLowerArm']
                    q_ll_s = [bf[i](curr_time, q_ll[i]) for i in range(4)]
                    norm = np.linalg.norm(q_ll_s)
                    if norm > 1e-6: q_ll_s = [x / norm for x in q_ll_s]
                    client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll_s[0]), float(q_ll_s[1]), float(q_ll_s[2]), float(q_ll_s[3])])

        if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
            start, end = to_unity_vec(get_vec(12)), to_unity_vec(get_vec(14))
            q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            bf = bone_quat_filters['RightUpperArm']
            q_ru_s = [bf[i](curr_time, q_ru[i]) for i in range(4)]
            norm = np.linalg.norm(q_ru_s)
            if norm > 1e-6: q_ru_s = [x / norm for x in q_ru_s]
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru_s[0]), float(q_ru_s[1]), float(q_ru_s[2]), float(q_ru_s[3])])
            if lm[16].visibility > 0.5:
                start, end = to_unity_vec(get_vec(14)), to_unity_vec(get_vec(16))
                q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                bf = bone_quat_filters['RightLowerArm']
                q_rl_s = [bf[i](curr_time, q_rl[i]) for i in range(4)]
                norm = np.linalg.norm(q_rl_s)
                if norm > 1e-6: q_rl_s = [x / norm for x in q_rl_s]
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl_s[0]), float(q_rl_s[1]), float(q_rl_s[2]), float(q_rl_s[3])])

        # --- LEG TRACKING (Added): Upper/Lower leg bones with smoothing ---
        if lm[23].visibility > 0.5 and lm[25].visibility > 0.5:
            start = to_unity_vec(get_vec(23))
            end = to_unity_vec(get_vec(25))
            q_lul = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
            bf = bone_quat_filters['LeftUpperLeg']
            q_lul_s = [bf[i](curr_time, q_lul[i]) for i in range(4)]
            norm = np.linalg.norm(q_lul_s)
            if norm > 1e-6: q_lul_s = [x / norm for x in q_lul_s]
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperLeg", 0.0, 0.0, 0.0, float(q_lul_s[0]), float(q_lul_s[1]), float(q_lul_s[2]), float(q_lul_s[3])])
            if lm[27].visibility > 0.5:
                start = to_unity_vec(get_vec(25))
                end = to_unity_vec(get_vec(27))
                q_lll = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                bf = bone_quat_filters['LeftLowerLeg']
                q_lll_s = [bf[i](curr_time, q_lll[i]) for i in range(4)]
                norm = np.linalg.norm(q_lll_s)
                if norm > 1e-6: q_lll_s = [x / norm for x in q_lll_s]
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerLeg", 0.0, 0.0, 0.0, float(q_lll_s[0]), float(q_lll_s[1]), float(q_lll_s[2]), float(q_lll_s[3])])

        if lm[24].visibility > 0.5 and lm[26].visibility > 0.5:
            start = to_unity_vec(get_vec(24))
            end = to_unity_vec(get_vec(26))
            q_rul = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
            bf = bone_quat_filters['RightUpperLeg']
            q_rul_s = [bf[i](curr_time, q_rul[i]) for i in range(4)]
            norm = np.linalg.norm(q_rul_s)
            if norm > 1e-6: q_rul_s = [x / norm for x in q_rul_s]
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperLeg", 0.0, 0.0, 0.0, float(q_rul_s[0]), float(q_rul_s[1]), float(q_rul_s[2]), float(q_rul_s[3])])
            if lm[28].visibility > 0.5:
                start = to_unity_vec(get_vec(26))
                end = to_unity_vec(get_vec(28))
                q_rll = get_limb_rotation(start, end, [0.0, -1.0, 0.0])
                bf = bone_quat_filters['RightLowerLeg']
                q_rll_s = [bf[i](curr_time, q_rll[i]) for i in range(4)]
                norm = np.linalg.norm(q_rll_s)
                if norm > 1e-6: q_rll_s = [x / norm for x in q_rll_s]
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerLeg", 0.0, 0.0, 0.0, float(q_rll_s[0]), float(q_rll_s[1]), float(q_rll_s[2]), float(q_rll_s[3])])

    # 3. FINGER TRACKING
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0)
            curl = filters_fingers_L[i](curr_time, raw_curl)
            angle = curl * (math.pi / 2.0) * THUMB_SIGN_L if name == "Thumb" else curl * (math.pi / 1.5) * FINGER_SIGN_L
            axis = THUMB_AXIS_L if name == "Thumb" else FINGER_AXIS_L
            fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
            for suffix in BONE_SUFFIXES: 
                client.send_message("/VMC/Ext/Bone/Pos", [f"Left{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0)
            curl = filters_fingers_R[i](curr_time, raw_curl)
            angle = curl * (math.pi / 2.0) * THUMB_SIGN_R if name == "Thumb" else curl * (math.pi / 1.5) * FINGER_SIGN_R
            axis = THUMB_AXIS_R if name == "Thumb" else FINGER_AXIS_R
            fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
            for suffix in BONE_SUFFIXES: 
                client.send_message("/VMC/Ext/Bone/Pos", [f"Right{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VTuber TESSELATION ONLY', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
