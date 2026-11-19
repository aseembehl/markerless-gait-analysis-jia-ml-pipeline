import numpy as np

from jia_gait.kinematics import compute_joint_angles
from jia_gait.landmarks import landmark_index

from .helpers import random_landmarks
from .legacy_reference import legacy_angles


def test_matches_original_code_exactly():
    lm = random_landmarks(n=500, seed=1)
    rows = [[float(i), *frame.reshape(-1).tolist()] for i, frame in enumerate(lm)]
    expected = legacy_angles(rows)
    np.testing.assert_allclose(compute_joint_angles(lm).features(), expected, atol=1e-9)


def _pose(side, knee_flex_deg, foot_dir):
    """Standing pose for one side; knee flexed by moving the ankle behind the walker."""
    lm = np.full((1, 12, 2), 500.0)
    t = np.radians(knee_flex_deg)
    back = -foot_dir  # 'behind' the walker in image x
    pts = {
        "shoulder": (0, -300),
        "hip": (0, 0),
        "knee": (0, 200),
        "ankle": (back * 200 * np.sin(t), 200 + 200 * np.cos(t)),
    }
    ax, ay = pts["ankle"]
    pts["heel"] = (ax + back * 20, ay + 30)
    pts["toe"] = (ax - back * 60, ay + 30)
    for j, (x, y) in pts.items():
        lm[0, landmark_index(j, side)] = (500 + x, 300 + y)
    return lm


def test_sign_conventions():
    # Walking right-to-left (toward -x) shows the LEFT side; left-to-right shows the RIGHT side.
    for side, direction, foot_dir in (("left", "rtl", -1), ("right", "ltr", +1)):
        straight = compute_joint_angles(_pose(side, 0, foot_dir)).direction(direction)
        assert abs(straight["hip"][0]) < 1e-6
        assert abs(straight["knee"][0]) < 1e-6
        assert abs(straight["ankle"][0]) < 0.5  # ltr carries the original 1 px ankle shift

        flexed = compute_joint_angles(_pose(side, 40, foot_dir)).direction(direction)
        np.testing.assert_allclose(flexed["knee"][0], 40, atol=0.5)
