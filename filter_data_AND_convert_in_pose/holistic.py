import numpy as np
from tqdm import tqdm

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format import Pose
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
from pose_format.utils.openpose import hand_colors
from person_detection import detection_person_frame


try:
    import mediapipe as mp
except ImportError:
    raise ImportError("Please install mediapipe with: pip install mediapipe")

mp_holistic = mp.solutions.holistic

FACEMESH_CONTOURS_POINTS = [
    str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
]

BODY_POINTS = mp_holistic.PoseLandmark._member_names_

BODY_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.POSE_CONNECTIONS]


HAND_POINTS = mp_holistic.HandLandmark._member_names_
HAND_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.HAND_CONNECTIONS]

FACE_POINTS_NUM = lambda additional_points=0: additional_points + 468

FACE_POINTS_NUM.__doc__ = """
Gets total number of face points and additional points.

Parameters
----------
additional_points : int, optional
    number of additional points to be added. The defaults is 0.

Returns
-------
int
    total number of face points.
"""
FACE_POINTS = lambda additional_points=0: [str(i) for i in range(FACE_POINTS_NUM(additional_points))]
FACE_POINTS.__doc__ = """
Makes a list of string representations of face points indexes up to total face points number

Parameters
----------
additional_points : int, optional
    number of additional points to be considered. Defaults to 0

Returns
-------
list[str]
    List of strings of face point indices.
"""

FACE_LIMBS = [(int(a), int(b)) for a, b in mp_holistic.FACEMESH_TESSELATION]


FLIPPED_BODY_POINTS = [
    'NOSE',
    'RIGHT_EYE_INNER',
    'RIGHT_EYE',
    'RIGHT_EYE_OUTER',
    'LEFT_EYE_INNER',
    'LEFT_EYE',
    'LEFT_EYE_OUTER',
    'RIGHT_EAR',
    'LEFT_EAR',
    'MOUTH_RIGHT',
    'MOUTH_LEFT',
    'RIGHT_SHOULDER',
    'LEFT_SHOULDER',
    'RIGHT_ELBOW',
    'LEFT_ELBOW',
    'RIGHT_WRIST',
    'LEFT_WRIST',
    'RIGHT_PINKY',
    'LEFT_PINKY',
    'RIGHT_INDEX',
    'LEFT_INDEX',
    'RIGHT_THUMB',
    'LEFT_THUMB',
    'RIGHT_HIP',
    'LEFT_HIP',
    'RIGHT_KNEE',
    'LEFT_KNEE',
    'RIGHT_ANKLE',
    'LEFT_ANKLE',
    'RIGHT_HEEL',
    'LEFT_HEEL',
    'RIGHT_FOOT_INDEX',
    'LEFT_FOOT_INDEX',
]


def component_points(component, width: int, height: int, num: int):
    """
    Gets component points

    Parameters
    ----------
    component : object
        Component with landmarks
    width : int
        Width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and confidence for each landmark
    """
    if component is not None:
        lm = component.landmark
        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)

    return np.zeros((num, 3)), np.zeros(num)

def component_points_face(component, width: int, height: int, num: int):
    """
    Gets component points

    Parameters
    ----------
    component : object
        Component with landmarks
    width : int
        Width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and confidence for each landmark
    """
    FACE_LANDMARKS = ['0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55',
                      '58',
                      '61', '63', '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93',
                      '95',
                      '103', '105', '107', '109', '127', '132', '133', '136', '144', '145', '146', '148', '149',
                      '150',
                      '152', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163', '172', '173',
                      '176',
                      '178', '181', '185', '191', '234', '246', '249', '251', '263', '267', '269', '270', '276',
                      '282',
                      '283', '284', '285', '288', '291', '293', '295', '296', '297', '300', '308', '310', '311',
                      '312',
                      '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362',
                      '365',
                      '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387',
                      '388',
                      '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466']
    FACE_LANDMARKS = list(map(int, FACE_LANDMARKS))
    size = len(FACE_LANDMARKS)
    if component is not None:
        lm = component.landmark
        fLm = [(lm[i].x * width, lm[i].y* height, lm[i].z) for i in FACE_LANDMARKS]
        #return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.ones(num)
        return np.array(fLm), np.ones(size)

    return np.zeros((size, 3)), np.zeros(size)


def body_points(component, width: int, height: int, num: int):
    """
    gets body points

    Parameters
    ----------
    component : object
        component containing landmarks
    width : int
        width
    height : int
        Height
    num : int
        number of landmarks

    Returns
    -------
    tuple of np.array
        coordinates and visibility for each landmark.
    """
    #BODY_POINTS = [11, 12, 14, 16, 13, 15, 24, 23]

    #size = len(BODY_POINTS)
    if component is not None:
        lm = component.landmark
        #fLm = [(lm[i].x * width, lm[i].y * height, lm[i].z) for i in BODY_POINTS]

        return np.array([[p.x * width, p.y * height, p.z] for p in lm]), np.array([p.visibility for p in lm])
        #return np.array(fLm), np.ones(size)

    return np.zeros((num, 3)), np.zeros(num)
    #return np.zeros((size, 3)), np.zeros(size)



def process_holistic(frames: list,
                     fps: float,
                     w: int,
                     h: int,
                     kinect=None,
                     progress=False,
                     additional_face_points=0,
                     additional_holistic_config={}, first_frame_to_stop=50) -> NumPyPoseBody:
    """
    process frames using holistic model from mediapipe

    Parameters
    ----------
    frames : list
        List of frames to be processed
    fps : float
        Frames per second
    w : int
        Frame width
    h : int
        Frame height.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_face_points : int, optional
        Additional face landmarks (points)
    additional_holistic_config : dict, optional
        Additional configurations for holistic model

    Returns
    -------
    NumPyPoseBody
        Processed pose data
    """
    holistic = mp_holistic.Holistic(static_image_mode=False, **additional_holistic_config)

    try:
        datas = []
        confs = []

        #from ultralytics import YOLO

        #model = YOLO('yolov8s.pt')


        for i, frame in enumerate(tqdm(frames, disable=not progress)):
            #frame = detection_person_frame(model, frame) #some yolov person detection
            results = holistic.process(frame)
            # Vérifie si aucun landmark n'est détecté
            if (results.pose_landmarks is None or
                    results.face_landmarks is None or
                    results.left_hand_landmarks is None or
                    results.right_hand_landmarks is None) and i > first_frame_to_stop:
                print("Aucun landmark détecté, arrêt du traitement.")
                break  # Arrête le traitement des frames


            body_data, body_confidence = body_points(results.pose_landmarks, w, h, 33)
            face_data, face_confidence = component_points_face(results.face_landmarks, w, h,
                                                          FACE_POINTS_NUM(additional_face_points))

            lh_data, lh_confidence = component_points(results.left_hand_landmarks, w, h, 21)
            rh_data, rh_confidence = component_points(results.right_hand_landmarks, w, h, 21)
            data = np.concatenate([body_data, face_data, lh_data, rh_data])
            conf = np.concatenate(
                [body_confidence, face_confidence, lh_confidence, rh_confidence])

            if kinect is not None:
                kinect_depth = []
                for x, y, z in np.array(data, dtype="int32"):
                    if 0 < x < w and 0 < y < h:
                        kinect_depth.append(kinect[i, y, x, 0])
                    else:
                        kinect_depth.append(0)

                kinect_vec = np.expand_dims(np.array(kinect_depth), axis=-1)
                data = np.concatenate([data, kinect_vec], axis=-1)

            datas.append(data)
            confs.append(conf)


        # Compléter les matrices plus petites avec des zéros
        """datas_padded = [
            np.pad(data, [(0, max_shape[i] - data.shape[i]) for i in range(len(data.shape))], mode='constant') for data
            in datas]"""

        pose_body_data = np.expand_dims(np.stack(datas), axis=1)
        pose_body_conf = np.expand_dims(np.stack(confs), axis=1)

        return NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=fps)
    finally:
        holistic.close()


def holistic_hand_component(name, pf="XYZC") -> PoseHeaderComponent:
    """
    Creates holistic hand component

    Parameters
    ----------
    name : str
        Component name
    pf : str, optional
        Point format

    Returns
    -------
    PoseHeaderComponent
        Hand component
    """
    return PoseHeaderComponent(name=name, points=HAND_POINTS, limbs=HAND_LIMBS, colors=hand_colors, point_format=pf)


def holistic_components(pf="XYZC", additional_face_points=0):
    """
    Creates list of holistic components

    Parameters
    ----------
    pf : str, optional
        Point format
    additional_face_points : int, optional
        Additional face points/landmarks

    Returns
    -------
    list of PoseHeaderComponent
        List of holistic components.
    """
    FACE_LANDMARKS = ['0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55',
                      '58',
                      '61', '63', '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93',
                      '95',
                      '103', '105', '107', '109', '127', '132', '133', '136', '144', '145', '146', '148', '149',
                      '150',
                      '152', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163', '172', '173',
                      '176',
                      '178', '181', '185', '191', '234', '246', '249', '251', '263', '267', '269', '270', '276',
                      '282',
                      '283', '284', '285', '288', '291', '293', '295', '296', '297', '300', '308', '310', '311',
                      '312',
                      '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362',
                      '365',
                      '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387',
                      '388',
                      '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466']
    #FACE_LANDMARKS = list(map(int, FACE_LANDMARKS))
    FACE_LIMBS = [(34, 18), (29, 32), (6, 56), (63, 10), (52, 51), (32, 23), (19, 15), (79, 105), (35, 19), (107, 108),
                  (127, 69), (77, 97), (80, 73), (2, 36), (0, 8), (35, 15), (82, 97), (122, 90), (18, 22), (37, 65),
                  (14, 6), (74, 80), (42, 41), (54, 53), (87, 86), (110, 111), (12, 18), (1, 7), (34, 12), (105, 92),
                  (89, 5), (43, 17), (123, 92), (124, 79), (23, 64), (66, 55), (81, 74), (40, 46), (8, 0), (97, 82),
                  (80, 96), (49, 48), (12, 34), (67, 118), (73, 84), (3, 88), (9, 10), (18, 34), (75, 80), (92, 123),
                  (31, 38), (59, 39), (11, 22), (36, 21), (20, 34), (62, 30), (63, 17), (48, 42), (100, 93), (13, 18),
                  (70, 0), (126, 99), (96, 82), (77, 81), (69, 67), (12, 19), (64, 24), (82, 74), (72, 124), (5, 89),
                  (18, 12), (24, 25), (25, 26), (112, 113), (89, 123), (15, 19), (124, 72), (97, 77), (20, 12),
                  (71, 70), (35, 20), (46, 45), (116, 127), (75, 74), (120, 112), (83, 98), (113, 114), (114, 115),
                  (15, 35), (85, 94), (106, 121), (13, 12), (20, 35), (27, 5), (104, 109), (5, 27), (34, 20), (57, 1),
                  (103, 104), (53, 52), (81, 82), (109, 110), (119, 78), (55, 54), (74, 75), (45, 60), (99, 117),
                  (95, 83), (74, 82), (28, 61), (58, 40), (93, 126), (125, 85), (80, 74), (60, 44), (12, 13), (76, 95),
                  (84, 73), (12, 20), (18, 13), (50, 49), (22, 11), (81, 77), (30, 43), (123, 89), (105, 79), (74, 81),
                  (26, 3), (73, 80), (56, 37), (90, 4), (33, 14), (92, 105), (11, 18), (111, 101), (82, 81), (4, 28),
                  (62, 27), (30, 62), (121, 107), (39, 50), (13, 11), (20, 19), (102, 119), (44, 47), (86, 125),
                  (71, 72), (0, 70), (115, 116), (47, 106), (17, 63), (80, 75), (19, 12), (118, 103), (41, 57),
                  (65, 31), (38, 16), (84, 80), (43, 30), (19, 35), (22, 18), (78, 100), (91, 122), (7, 66), (70, 71),
                  (51, 59), (9, 8), (101, 120), (16, 58), (61, 29), (72, 71), (8, 9), (21, 33), (18, 11), (10, 9),
                  (96, 80), (94, 91), (73, 75), (74, 96), (75, 73), (108, 102), (88, 87), (97, 81), (79, 124), (11, 13),
                  (19, 20), (82, 96), (80, 84), (117, 68), (27, 62), (81, 97), (98, 2), (17, 43), (96, 74), (68, 76),
                  (10, 63)]
    return [
        PoseHeaderComponent(name="POSE_LANDMARKS",
                            points=BODY_POINTS,
                            limbs=BODY_LIMBS,
                            colors=[(255, 0, 0)],
                            point_format=pf),

        PoseHeaderComponent(name="FACE_LANDMARKS",
                            #points=FACE_POINTS(additional_face_points),
                            points=FACE_LANDMARKS,
                            limbs=FACE_LIMBS,
                            colors=[(128, 0, 0)],
                            point_format=pf),
        holistic_hand_component("LEFT_HAND_LANDMARKS", pf),
        holistic_hand_component("RIGHT_HAND_LANDMARKS", pf),


    ]


def load_holistic(frames: list,
                  fps: float = 24,
                  width=1000,
                  height=1000,
                  depth=0,
                  kinect=None,
                  progress=False,
                  additional_holistic_config={}) -> Pose:
    """
    Loads holistic pose data

    Parameters
    ----------
    frames : list
        List of frames.
    fps : float, optional
        Frames per second.
    width : int, optional
        Frame width.
    height : int, optional
        Frame height.
    depth : int, optional
        Depth data.
    kinect : object, optional
        Kinect depth data.
    progress : bool, optional
        If True, show the progress bar.
    additional_holistic_config : dict, optional
        Additional configurations for the holistic model.

    Returns
    -------
    Pose
        Loaded pose data with header and body
    """
    pf = "XYZC" if kinect is None else "XYZKC"

    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    refine_face_landmarks = 'refine_face_landmarks' in additional_holistic_config and additional_holistic_config[
        'refine_face_landmarks']
    if refine_face_landmarks:
        print("refine_face_landmarks")
    additional_face_points = 10 if refine_face_landmarks else 0
    header: PoseHeader = PoseHeader(version=0.1,
                                    dimensions=dimensions,
                                    components=holistic_components(pf, additional_face_points))
    body: NumPyPoseBody = process_holistic(frames, fps, width, height, kinect, progress, additional_face_points,
                                           additional_holistic_config)

    return Pose(header, body)