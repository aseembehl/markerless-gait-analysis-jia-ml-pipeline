"""Verbatim copies of the original research code, used as test oracles.

Sources (OneDrive research/gait_analysis/notebooks):
  - pointsToAngles-MacBook Air (5).ipynb       -> angle_between, legacy_angles
  - heal_on_event_prediction-MacBook Air (5).ipynb -> legacy_postprocess
  - plots_v7.ipynb (latest make_plots)         -> legacy_preprocess, legacy_cycle_to_grid,
                                                  legacy_lowess_wrap
Only I/O was replaced (CSV reading/writing -> in-memory arrays); the logic is unchanged.
"""

import math
import warnings

import numpy as np
import pandas as pd
import scipy.interpolate


def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)


def legacy_angles(rows):
    """rows: list of [imgid, 24 coords]. Returns columns in CSV order (hipLtoR, hipRtoL, ...)."""
    ankleRtoL, ankleLtoR, hipRtoL, hipLtoR, kneeRtoL, kneeLtoR = [], [], [], [], [], []
    for row in rows:
        knee = (row[9], row[10])
        ankle = (row[13], row[14])
        ankleVertical = (row[13], row[14] + 1)
        toe = (row[17], row[18])
        toeVertical = (row[17], row[18] + 1)
        heel = (row[21], row[22])
        ankleRtoL.append(
            (angle_between(heel, toe, toeVertical) - 90)
            - (angle_between(knee, ankle, ankleVertical) - 90)
            - 90
        )
    for row in rows:
        knee = (row[11], row[12])
        ankle = (row[15], row[16])
        ankleVertical = ankle = (row[15], row[16] + 1)
        toe = (row[19], row[20])
        toeVertical = (row[19], row[20] + 1)
        heel = (row[23], row[24])
        ankleLtoR.append(
            -angle_between(heel, toe, toeVertical)
            + (angle_between(knee, ankle, ankleVertical) - 180)
            + 90
        )
    for row in rows:
        trunk = (row[1], row[2])
        hip = (row[5], row[6])
        knee = (row[9], row[10])
        hipRtoL.append(-angle_between(trunk, hip, knee) + 180)
    for row in rows:
        trunk = (row[3], row[4])
        hip = (row[7], row[8])
        knee = (row[11], row[12])
        hipLtoR.append(angle_between(trunk, hip, knee) - 180)
    for row in rows:
        ankle = (row[13], row[14])
        hip = (row[5], row[6])
        knee = (row[9], row[10])
        kneeRtoL.append(angle_between(hip, knee, ankle) - 180)
    for row in rows:
        ankle = (row[15], row[16])
        hip = (row[7], row[8])
        knee = (row[11], row[12])
        kneeLtoR.append(180 - angle_between(hip, knee, ankle))
    return np.array([hipLtoR, hipRtoL, kneeLtoR, kneeRtoL, ankleLtoR, ankleRtoL]).T


def legacy_postprocess(y_pred_proba):
    y_pred = np.zeros(y_pred_proba.shape)
    y_pred[:, 0] = np.where(y_pred_proba[:, 0] > 0.5, 1, 0)
    y_pred[:, 1] = np.where(y_pred_proba[:, 1] > 0.5, 1, 0)
    y_pred[:, 2] = np.where(y_pred_proba[:, 2] > 0.5, 1, 0)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred_smooth = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        if i < 2 or i > y_pred.shape[0] - 3:
            y_pred_smooth[i] = y_pred[i]
        else:
            y_pred_smooth[i] = np.argmax(np.bincount(y_pred[i - 2 : i + 3]))
    middle_points = []
    for i in range(2):
        start = -1
        end = -1
        middle_points.append([])
        for j in range(y_pred_smooth.shape[0]):
            if y_pred_smooth[j] == i and j > end:
                start = j
                for k in range(start + 1, y_pred_smooth.shape[0] - 1):
                    if y_pred_smooth[k] != i:
                        end = k
                        if end - start > 3:
                            middle_points[i].append(int((start + end) / 2))
                        break
    y_pred_peak = 2 * np.ones(y_pred_smooth.shape)
    for i in range(2):
        for j in range(len(middle_points[i])):
            y_pred_peak[middle_points[i][j]] = i
    num_frames = []
    for i in range(2):
        start = -1
        end = -1
        for j in range(y_pred_peak.shape[0]):
            if y_pred_peak[j] == i and j > end:
                start = j
                for k in range(start + 1, y_pred_smooth.shape[0] - 1):
                    if y_pred_peak[k] == i:
                        end = k
                        num_frames.append(end - start)
                        break
                    elif y_pred_peak[k] == 1 or y_pred_peak[k] == 0:
                        end = k
                        break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med, sd = np.median(num_frames), np.ceil(np.std(num_frames))
    frame_numbers = []
    for i in range(2):
        start = -1
        end = -1
        for j in range(y_pred_peak.shape[0]):
            if y_pred_peak[j] == i and j > end:
                start = j
                for k in range(start + 1, y_pred_smooth.shape[0] - 1):
                    if y_pred_peak[k] == i:
                        end = k
                        if end - start > med + 2.0 * sd or end - start < med - 2.0 * sd:
                            frame_numbers.append(start)
                            frame_numbers.append(end)
                        break
                    elif y_pred_peak[k] == 1 or y_pred_peak[k] == 0:
                        end = k
                        break
    for i in range(len(frame_numbers)):
        y_pred_peak[frame_numbers[i]] = 2
    return y_pred_peak


def legacy_preprocess(df):
    df = df.copy()
    df["time"] = df.index
    df["f_split"] = np.nan
    df["f_split"] = df["f_split"].astype(object)
    lr_idx = df.index[df["Heel on left to right"].notna()]
    rl_idx = df.index[df["Heel on right to left"].notna()]
    df.loc[lr_idx, "f_split"] = [f"l{i}" for i in range(len(lr_idx))]
    df.loc[rl_idx, "f_split"] = [f"r{i}" for i in range(len(rl_idx))]
    df["f_split"] = df["f_split"].ffill()
    df = df.dropna(subset=["f_split"])
    splits_unique = pd.Index(df["f_split"].unique())
    l_splits = [s for s in splits_unique if isinstance(s, str) and s.startswith("l")]
    r_splits = [s for s in splits_unique if isinstance(s, str) and s.startswith("r")]
    lr_max = max([int(s[1:]) for s in l_splits], default=-1)
    rl_max = max([int(s[1:]) for s in r_splits], default=-1)
    inserts = []
    for i in range(1, lr_max + 1):
        g = df[df["f_split"] == f"l{i}"]
        if not g.empty:
            row = g.iloc[0].copy()
            row["f_split"] = f"l{i - 1}"
            inserts.append(row)
    for i in range(1, rl_max + 1):
        g = df[df["f_split"] == f"r{i}"]
        if not g.empty:
            row = g.iloc[0].copy()
            row["f_split"] = f"r{i - 1}"
            inserts.append(row)
    if inserts:
        df = pd.concat([df, pd.DataFrame(inserts)], ignore_index=True)
    split_order = df.groupby("f_split", as_index=True)["time"].min().sort_values().index.tolist()
    splits_remove = set()
    if lr_max >= 0:
        splits_remove.add(f"l{lr_max}")
    if rl_max >= 0:
        splits_remove.add(f"r{rl_max}")
    for i in range(len(split_order) - 1):
        if split_order[i][0] != split_order[i + 1][0]:
            splits_remove.add(split_order[i])
    df = df[~df["f_split"].isin(splits_remove)].copy()
    df.sort_values(by=["f_split", "time"], inplace=True)
    df["relative_time"] = df.groupby("f_split").cumcount()

    def _norm(s):
        m = s.max()
        return s / m if m and m > 0 else 0.0

    df["relative_time"] = df.groupby("f_split", group_keys=False)["relative_time"].apply(_norm)
    return df


XGRID = np.arange(0.0, 1.0 + 1e-9, 0.01)


def legacy_cycle_to_grid(x, y):
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    tmp = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
    x_u = tmp["x"].to_numpy()
    y_u = tmp["y"].to_numpy()
    f = scipy.interpolate.interp1d(x_u, y_u, bounds_error=False, fill_value=np.nan)
    ygrid = f(XGRID)
    return pd.Series(ygrid).interpolate(limit_direction="both").to_numpy()


def legacy_lowess_wrap(ygrid, frac=0.08, it=1, wrap_frac=0.04):
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    xgrid = XGRID
    n = len(ygrid)
    w = max(2, int(round(wrap_frac * n)))
    x_ext = np.concatenate([xgrid[-w:] - 1.0, xgrid, xgrid[:w] + 1.0])
    y_ext = np.concatenate([ygrid[-w:], ygrid, ygrid[:w]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = sm_lowess(y_ext, x_ext, frac=frac, it=it, return_sorted=True)
    f = scipy.interpolate.interp1d(sm[:, 0], sm[:, 1], bounds_error=False, fill_value=np.nan)
    y_sm = f(xgrid)
    return pd.Series(y_sm).interpolate(limit_direction="both").to_numpy()


# --- vicon_crop.ipynb: per-folder crop loop (model output given as the best box per frame) ---
def legacy_crop_decisions(best_boxes, width, cohort=True):
    """best_boxes: list of (x1, y1, x2, y2) or None per frame. Returns list of kept frame indices."""
    kept = []
    out_of_region_flag = 5
    for index, box in enumerate(best_boxes):
        if box is not None:
            if cohort:
                at_edge = box[0] < width / 6 or box[2] > (width - width / 6)
            else:
                at_edge = box[2] < width / 6 or box[0] > (width - width / 6)
            if at_edge:
                if out_of_region_flag >= 4:
                    continue
                else:
                    out_of_region_flag += 1
                    kept.append(index)
                    continue
            if box[3] - box[1] < (200 if cohort else 500):
                continue
            out_of_region_flag = 0
            kept.append(index)
    return kept


# --- vicon_yolo_pose.ipynb: keypoint selection and CSV row ---
def legacy_select_row(frame_keypoints0):
    fkpts = frame_keypoints0[5:23]
    fkpts = np.delete(fkpts, [2, 3, 4, 5], 0)
    fkpts = np.delete(fkpts, [-3, -6], 0)
    myPoints = fkpts
    return [
        myPoints[0][1],
        myPoints[0][0],
        myPoints[1][1],
        myPoints[1][0],
        myPoints[2][1],
        myPoints[2][0],
        myPoints[3][1],
        myPoints[3][0],
        myPoints[4][1],
        myPoints[4][0],
        myPoints[5][1],
        myPoints[5][0],
        myPoints[6][1],
        myPoints[6][0],
        myPoints[7][1],
        myPoints[7][0],
        myPoints[8][1],
        myPoints[8][0],
        myPoints[10][1],
        myPoints[10][0],
        myPoints[9][1],
        myPoints[9][0],
        myPoints[11][1],
        myPoints[11][0],
    ]
