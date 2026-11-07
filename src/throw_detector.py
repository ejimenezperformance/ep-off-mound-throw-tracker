"""EP Off-Mound Throw Tracker - prototype v0.2.1
Emerson Jimenez | Emerson Performance

Counts the throws a pitcher makes in a video (catch play, long toss, flat ground,
bullpen) using YOLO pose estimation, ByteTrack person tracking and a rule-based
arm-speed signal.

Usage:
    python src/throw_detector.py <video.mp4> <output_dir> [--model yolo11m-pose.pt]
"""
import sys, json, os
import numpy as np
import cv2
from ultralytics import YOLO
from scipy.signal import find_peaks, savgol_filter

# COCO keypoints
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12

MODEL = "yolo11m-pose.pt"   # downloaded automatically by Ultralytics on first run
SPEED_THRESHOLD = 0.35   # wrist speed (torso lengths per frame) to count as a throw
MIN_GAP_SEC = 2.5        # two throws cannot be closer than this (ball must come back)
CONF_MIN = 0.3


def extract_pose(video):
    """Track every person (ByteTrack) and keep keypoints per track id."""
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracks, t = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")[0]
        if r.boxes is not None and r.boxes.id is not None:
            for j, tid in enumerate(r.boxes.id.int().cpu().tolist()):
                tracks.setdefault(tid, {})[t] = (r.keypoints.xy[j].cpu().numpy(),
                                                r.keypoints.conf[j].cpu().numpy())
        t += 1
    cap.release()
    return tracks, t, fps


def as_rows(track, n):
    return [track.get(t) for t in range(n)]


def arm_signal(rows, wrist, shoulder):
    """Wrist speed relative to the hip center, normalized by torso length.
    Removing hip motion ignores running / crow-hop; normalizing by torso
    makes it work whether the player is near or far from the camera."""
    n = len(rows)
    rel = np.full((n, 2), np.nan)
    torso = np.full(n, np.nan)
    for t, r in enumerate(rows):
        if r is None:
            continue
        kp, cf = r
        need = [wrist, shoulder, L_HIP, R_HIP, L_SH, R_SH]
        if min(cf[need]) < CONF_MIN:
            continue
        hip = (kp[L_HIP] + kp[R_HIP]) / 2
        sh = (kp[L_SH] + kp[R_SH]) / 2
        torso[t] = np.linalg.norm(sh - hip)
        rel[t] = kp[wrist] - hip
    # fill short gaps
    for d in range(2):
        s = rel[:, d]
        idx = np.arange(n)
        good = ~np.isnan(s)
        if good.sum() > 2:
            rel[:, d] = np.interp(idx, idx[good], s[good])
    good = ~np.isnan(torso)
    if good.sum() < 3:
        return np.zeros(n)
    torso = np.interp(np.arange(n), np.arange(n)[good], torso[good])
    speed = np.linalg.norm(np.diff(rel, axis=0, prepend=rel[:1]), axis=1) / torso
    if n >= 7:
        speed = np.clip(savgol_filter(speed, 5, 2), 0, None)
    return speed


def throws_for(rows, fps):
    sig = {"R": arm_signal(rows, R_WR, R_SH), "L": arm_signal(rows, L_WR, L_SH)}
    arm = max(sig, key=lambda a: np.nanmax(sig[a]))
    speed = sig[arm]
    present = np.array([r is not None for r in rows])
    speed = np.where(present, speed, 0.0)
    peaks, _ = find_peaks(speed, height=SPEED_THRESHOLD, distance=max(1, int(MIN_GAP_SEC * fps)))
    return arm, speed, peaks


def scene_cuts(video, thresh=0.5):
    """Frames where the camera cuts (big change in color histogram)."""
    cap = cv2.VideoCapture(video)
    prev, cuts, t = None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = cv2.calcHist([cv2.cvtColor(cv2.resize(f, (160, 160)), cv2.COLOR_BGR2HSV)],
                         [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > thresh:
            cuts.append(t)
        prev, t = h, t + 1
    cap.release()
    return cuts


def detect(video, out_dir):
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "poses.pkl")
    if os.path.exists(cache):
        tracks, n, fps = pickle.load(open(cache, "rb"))
    else:
        tracks, n, fps = extract_pose(video)
        pickle.dump((tracks, n, fps), open(cache, "wb"))
    cuts = scene_cuts(video)
    blocked = np.zeros(n, bool)
    for c in cuts:
        blocked[max(0, c - 2):c + 3] = True   # ignore fake jumps at camera cuts

    cands = {}
    for tid, tr in tracks.items():
        if len(tr) < 1 * fps:
            continue
        rows = as_rows(tr, n)
        arm, speed, _ = throws_for(rows, fps)
        speed = np.where(blocked, 0.0, speed)
        peaks, _ = find_peaks(speed, height=SPEED_THRESHOLD, distance=max(1, int(MIN_GAP_SEC * fps)))
        cands[tid] = dict(rows=rows, arm=arm, speed=speed, peaks=peaks,
                          score=float(np.sort(speed)[-3:].mean()))
    best = max(v["score"] for v in cands.values())
    # thrower tracks: the main thrower may get a new id after each camera cut
    thrower = [k for k, v in cands.items() if v["score"] >= 0.5 * best and len(v["peaks"])]
    # only one thrower at a time: if two candidates share the same frames
    # (e.g. the pitcher and a coach in the background), keep the stronger arm
    def overlap(a, b):
        fa, fb = set(tracks[a]), set(tracks[b])
        return len(fa & fb) / min(len(fa), len(fb))
    thrower.sort(key=lambda k: -cands[k]["score"])
    kept = []
    for k in thrower:
        if all(overlap(k, j) < 0.5 for j in kept):
            kept.append(k)
    thrower = kept
    # merge peaks from all thrower tracks, keep strongest within MIN_GAP
    allp = sorted([(int(p), float(cands[k]["speed"][p]), k) for k in thrower for p in cands[k]["peaks"]])
    merged = []
    for p, v, k in allp:
        if merged and p - merged[-1][0] < MIN_GAP_SEC * fps:
            if v > merged[-1][1]:
                merged[-1] = (p, v, k)
        else:
            merged.append((p, v, k))
    # combined signal / rows for drawing: the thrower track present in each frame
    speed = np.zeros(n)
    rows = [None] * n
    for k in thrower:
        for t in range(n):
            if cands[k]["rows"][t] is not None and cands[k]["speed"][t] >= speed[t]:
                speed[t] = cands[k]["speed"][t]
                rows[t] = cands[k]["rows"][t]
    peaks = np.array([p for p, _, _ in merged], int)
    arm = cands[max(thrower, key=lambda k: cands[k]["score"])]["arm"]
    throws = [{"throw": i + 1, "frame": p, "time_sec": round(p / fps, 2),
               "peak_arm_speed": round(v, 3), "track_id": k} for i, (p, v, k) in enumerate(merged)]
    result = {"video": os.path.basename(video), "fps": fps, "frames": n,
              "camera_cuts_sec": [round(c / fps, 2) for c in cuts],
              "people_tracked": len(cands), "thrower_tracks": thrower,
              "signal_arm": arm, "throws_detected": len(throws), "throws": throws,
              "thrower_visible_pct": round(100 * sum(r is not None for r in rows) / n, 1)}
    with open(os.path.join(out_dir, "throws.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savetxt(os.path.join(out_dir, "arm_speed.csv"), speed, delimiter=",")
    annotate(video, rows, speed, peaks, fps, arm, os.path.join(out_dir, "annotated.mp4"))
    plot(speed, peaks, fps, os.path.join(out_dir, "arm_speed.png"))
    return result


def annotate(video, rows, speed, peaks, fps, arm, path):
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(3)), int(cap.get(4))
    tmp = path.replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    wr = R_WR if arm == "R" else L_WR
    count, t = 0, 0
    flash = set()
    for p in peaks:
        flash.update(range(p, p + int(fps)))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t in set(peaks):
            count += 1
        r = rows[t]
        if r is not None:
            kp, cf = r
            for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                         (11, 13), (13, 15), (12, 14), (14, 16)]:
                if cf[a] > CONF_MIN and cf[b] > CONF_MIN:
                    pa, pb = tuple(map(int, kp[a])), tuple(map(int, kp[b]))
                    cv2.line(frame, pa, pb, (51, 27, 11), 6)    # navy outline
                    cv2.line(frame, pa, pb, (58, 165, 212), 3)  # EP gold
            for k in range(5, 17):
                if cf[k] > CONF_MIN:
                    cv2.circle(frame, tuple(map(int, kp[k])), 4, (51, 27, 11), -1)
            if cf[wr] > CONF_MIN:
                cv2.circle(frame, tuple(map(int, kp[wr])), 9, (0, 0, 230), -1)
        cv2.rectangle(frame, (0, 0), (w, 70), (51, 27, 11), -1)
        cv2.putText(frame, f"THROWS: {count}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (58, 165, 212), 2)
        cv2.putText(frame, f"arm speed: {speed[t]:.2f}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 243, 236), 1)
        if t in flash:
            cv2.putText(frame, "THROW DETECTED", (w - 230, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        out.write(frame)
        t += 1
    cap.release()
    out.release()
    os.system(f'ffmpeg -v error -y -i "{tmp}" -c:v libx264 -preset veryfast -pix_fmt yuv420p "{path}" && rm "{tmp}"')


def plot(speed, peaks, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(speed)) / fps
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, speed, color="#0B1B33", lw=2, label="Arm speed (wrist vs. hips)")
    ax.axhline(SPEED_THRESHOLD, color="#999999", ls="--", lw=1, label="Throw threshold")
    ax.scatter(t[peaks], speed[peaks], color="#D4A53A", s=80, zorder=3, label="Throw detected")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Torso lengths / frame")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EP Off-Mound Throw Tracker")
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=MODEL, help="YOLO pose weights (default: yolo11m-pose.pt)")
    args = ap.parse_args()
    MODEL = args.model
    print(json.dumps(detect(args.video, args.out_dir), indent=2))
