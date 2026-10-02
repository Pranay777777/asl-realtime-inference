# predict.py
import argparse, json
from pathlib import Path
from collections import deque, Counter
import math

import cv2
import numpy as np
from tensorflow import keras

import mediapipe as mp


# --------------------------- helpers ---------------------------

def preprocess_frame(frame, roi, img_size, *, use_clahe=True, use_skin_mask=False):
    x, y, w, h = roi
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, img_size, interpolation=cv2.INTER_AREA)
    # model expects RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # (optional) background suppression via simple skin mask
    if use_skin_mask:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        # crude ranges; tune if needed for your lighting/skin tone
        mask1 = cv2.inRange(hsv,   (0,  40,  40),  (25, 255, 255))
        mask2 = cv2.inRange(ycrcb, (0, 133,  77),  (255, 173, 127))
        mask  = cv2.bitwise_and(mask1, mask2)
        mask  = cv2.medianBlur(mask, 5)
        rgb   = cv2.bitwise_and(rgb, rgb, mask=mask)

    # (optional) CLAHE on V channel for lighting robustness
    if use_clahe:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v])
        rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    arr = rgb.astype(np.float32)          # keep 0..255
    return np.expand_dims(arr, axis=0)


def load_calibration(path):
    """(temperature, abstain threshold) from inference/calibrate.py, or (1.0, None)."""
    p = Path(path)
    if not p.exists():
        return 1.0, None
    cal = json.loads(p.read_text())
    return float(cal.get("temperature", 1.0)), float(cal["threshold"])


def calibrated(probs, temperature):
    """Softmax scores as if the logits were divided by the fitted temperature."""
    if temperature == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def majority_vote(q):
    if not q:
        return None
    c = Counter(q)
    return c.most_common(1)[0][0]


def make_tta_batch(arr, angles, do_flip):
    """
    arr: (1,H,W,3) float32 in 0..255 (no /255).
    Returns batch (N,H,W,3) float32 in 0..255.
    """
    img = arr[0].astype(np.uint8)  # already 0..255
    h, w = img.shape[:2]
    imgs = []
    for a in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        imgs.append(rot)
        if do_flip:
            imgs.append(cv2.flip(rot, 1))
    batch = np.stack([x.astype(np.float32) for x in imgs], axis=0)
    return batch

def mp_aligned_hand(frame_bgr, target_size, hands, *, pad_scale=1.4):
    """
    Detect 1 hand with MediaPipe, rotate to upright, crop with margin, resize to target_size.
    Returns (arr[None,...] in RGB[0..1], bbox tuple) or (None, None) if not found.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    if not res.multi_hand_landmarks:
        return None, None

    # use first hand
    lm = res.multi_hand_landmarks[0]
    xs = [int(p.x * w) for p in lm.landmark]
    ys = [int(p.y * h) for p in lm.landmark]
    x0, x1 = max(0, min(xs)), min(w-1, max(xs))
    y0, y1 = max(0, min(ys)), min(h-1, max(ys))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bw, bh = (x1 - x0), (y1 - y0)
    side = int(max(bw, bh) * pad_scale)

    # estimate rotation: vector from wrist(0) to index_mcp(5)
    wrist = lm.landmark[0]; idx_mcp = lm.landmark[5]
    vx = (idx_mcp.x - wrist.x) * w
    vy = (idx_mcp.y - wrist.y) * h
    angle_deg = -math.degrees(math.atan2(vy, vx)) + 90  # make index roughly 'up'

    # crop square around center with padding
    half = side // 2
    xA, yA = max(0, cx - half), max(0, cy - half)
    xB, yB = min(w, cx + half), min(h, cy + half)
    crop = frame_bgr[yA:yB, xA:xB]
    if crop.size == 0:
        return None, None

    # rotate crop to canonical orientation
    ch, cw = crop.shape[:2]
    M = cv2.getRotationMatrix2D((cw/2, ch/2), angle_deg, 1.0)
    rot = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    # center-crop square after rotation (ensure square)
    s = min(ch, cw)
    y0r = (ch - s) // 2; x0r = (cw - s) // 2
    rot_sq = rot[y0r:y0r+s, x0r:x0r+s]

    # resize → RGB → [0..1]
    resized = cv2.resize(rot_sq, tuple(target_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(rgb, axis=0), (xA, yA, xB - xA, yB - yA)


# --------- dynamic gesture helpers (J / Z via fingertip path) ---------

def mp_tip_and_bbox(frame_bgr, hands):
    """Return ((tx,ty) fingertip normalized to bbox, bbox=(x,y,w,h)) or (None, None)."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None

    lm = res.multi_hand_landmarks[0]
    xs = [int(p.x * w) for p in lm.landmark]
    ys = [int(p.y * h) for p in lm.landmark]
    x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)

    tip = lm.landmark[8]  # index fingertip
    tx = (tip.x * w - x0) / bw
    ty = (tip.y * h - y0) / bh
    # clamp to [0,1]
    tx = 0.0 if tx < 0 else (1.0 if tx > 1 else tx)
    ty = 0.0 if ty < 0 else (1.0 if ty > 1 else ty)
    return (tx, ty), (x0, y0, bw, bh)


def _path_length(traj):
    if len(traj) < 2: return 0.0
    return sum(math.hypot(traj[i][0] - traj[i-1][0], traj[i][1] - traj[i-1][1]) for i in range(1, len(traj)))


def _angle(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))  # [-180, 180]


def _split_vectors(traj):
    """Return coarse 3 segments (v1,v2,v3) using quartiles of the trajectory."""
    n = len(traj)
    if n < 9: return None
    p0 = traj[0]
    p1 = traj[n//3]
    p2 = traj[(2*n)//3]
    p3 = traj[-1]
    v1 = (p1[0]-p0[0], p1[1]-p0[1])
    v2 = (p2[0]-p1[0], p2[1]-p1[1])
    v3 = (p3[0]-p2[0], p3[1]-p2[1])
    return v1, v2, v3


def _signed_area(traj):
    """Shoelace area (approx curvature direction). Negative often matches a 'J' hook."""
    area = 0.0
    for i in range(1, len(traj)):
        x1,y1 = traj[i-1]; x2,y2 = traj[i]
        area += (x1*y2 - x2*y1)
    return area * 0.5


def detect_dynamic_JZ(traj, min_move=0.35):
    """
    Heuristic: normalized trajectory in bbox space (0..1).
    Returns 'J', 'Z' or None.
    """
    if len(traj) < 12:
        return None
    path = _path_length(traj)
    if path < min_move:
        return None

    segs = _split_vectors(traj)
    if not segs: return None
    v1, v2, v3 = segs
    a1 = _angle(v1); a2 = _angle(v2); a3 = _angle(v3)

    # Z pattern: right (~0°), down-left (~-135°), right (~0°), or mirrored left/down-right/left.
    def near(angle, target, tol=40):  # degrees
        # handle wrap-around
        d = (angle - target + 180) % 360 - 180
        return abs(d) <= tol

    # try right, down-left, right
    if near(a1, 0) and near(a2, -135) and near(a3, 0):
        return 'Z'
    # mirrored Z: left, down-right, left
    if near(a1, 180) and near(a2, -45) and near(a3, 180):
        return 'Z'

    # J pattern: overall downward motion + negative signed area (clockwise hook)
    dy = traj[-1][1] - traj[0][1]
    area = _signed_area(traj)
    if dy > 0.20 and area < -0.01:
        return 'J'

    return None


# --------------------------- main ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--labels_path", required=True, type=str)
    ap.add_argument("--img_size", nargs=2, type=int, default=[96, 96])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--roi", nargs=4, type=int, default=[50, 50, 224, 224], help="x y w h")
    ap.add_argument("--smooth_k", type=int, default=9)
    ap.add_argument("--conf_thresh", type=float, default=None,
                    help="Abstain below this confidence (default: models/calibration.json, else 0.40)")
    ap.add_argument("--calibration", type=str, default="models/calibration.json",
                    help="Temperature + threshold written by inference/calibrate.py")
    ap.add_argument("--letters_only", type=lambda s: str(s).lower() != "false", default=True)
    ap.add_argument("--skin_mask", action="store_true", help="Apply HSV+YCrCb skin mask to ROI")
    ap.add_argument("--use_mediapipe", action="store_true",
                    help="Detect and align hand ROI with MediaPipe Hands")

    # UI niceties
    ap.add_argument("--show_spaces", action="store_true", help="Render spaces as ␣ in overlay")

    # Test-Time Augmentation (TTA) for angle invariance (compact set)
    ap.add_argument("--tta_angles", nargs="+", type=float, default=[-15, 0, 15],
                    help="Angles (degrees) to rotate ROI and average predictions")
    ap.add_argument("--tta_flip", action="store_true",
                    help="Also include a horizontally flipped version in TTA")
    ap.add_argument("--no_tta", action="store_true",
                    help="Disable TTA; use single ROI prediction")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE lighting normalization")
    ap.add_argument("--mirror", action="store_true",
                    help="Mirror webcam frames horizontally before processing")

    # Dynamic gestures (J, Z) via landmarks
    ap.add_argument("--dynamic", action="store_true",
                    help="Enable dynamic gesture detection (J, Z) using MediaPipe landmarks")
    ap.add_argument("--dynamic_len", type=int, default=24,
                    help="Frames kept for fingertip trajectory (20–32 recommended)")
    ap.add_argument("--dynamic_min_move", type=float, default=0.35,
                    help="Min normalized path length (0..~2) to consider a dynamic gesture")
    ap.add_argument("--dynamic_cooldown", type=int, default=18,
                    help="Frames to ignore after a dynamic commit to prevent repeats")

    # Single image mode (no webcam)
    ap.add_argument("--test_image", type=str, default=None, help="Run on a single image and exit")
    args = ap.parse_args()

    # Load model & labels
    model = keras.models.load_model(args.model_path)
    idx_to_label = {int(k): v for k, v in json.loads(Path(args.labels_path).read_text()).items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    temperature, cal_thresh = load_calibration(args.calibration)
    if args.conf_thresh is None:
        args.conf_thresh = cal_thresh if cal_thresh is not None else 0.40
    print(f"[INFO] temperature={temperature} abstain below {args.conf_thresh:.2f}")

    # Auto-align image size to the model's input (None, H, W, C)
    mh, mw = model.input_shape[1:3]
    img_size = (mw, mh) if (mh is not None and mw is not None) else tuple(args.img_size)
    print(f"[INFO] Using img_size={img_size} (model expects HxW={(mh, mw)})")

    # ---------- Single image test mode ----------
    if args.test_image:
        img = cv2.imread(args.test_image)
        if img is None:
            raise FileNotFoundError(f"Could not read: {args.test_image}")
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
        arr = cv2.resize(crop, img_size, interpolation=cv2.INTER_AREA)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
        probs = calibrated(model.predict(arr[None, ...], verbose=0)[0], temperature)
        idx = int(np.argmax(probs))
        if probs[idx] < args.conf_thresh:
            print(f"Not confident (best guess {idx_to_label[idx]} at {probs[idx] * 100:.1f}%)")
        else:
            print(f"Top-1: {idx_to_label[idx]} ({probs[idx] * 100:.1f}%)")
        raise SystemExit(0)

    # ---------- Webcam mode ----------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try --camera 1 or check permissions.")

    # init MediaPipe Hands if needed (for ROI alignment or dynamic tracking)
    mp_hands = mp.solutions.hands
    hands = None
    if args.use_mediapipe or args.dynamic:
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

    roi = tuple(args.roi)
    recent = deque(maxlen=args.smooth_k)
    running_text = ""

    # EMA over probabilities (temporal smoothing)
    probs_ema = None
    EMA_ALPHA = 0.5  # 0.3–0.7; lower=steadier, higher=faster

    # stability lock (require same label for K frames)
    stable_label = None
    stable_count = 0
    LOCK_FRAMES = 3  # 2–4 work well

    # Dynamic-gesture buffers (persist across frames)
    tip_traj = deque(maxlen=args.dynamic_len)
    dyn_cooldown = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        x, y, rw, rh = roi
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        rw = max(1, min(rw, W - x))
        rh = max(1, min(rh, H - y))
        roi = (x, y, rw, rh)

        # Draw ROI & hint
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        cv2.putText(frame, "Place hand inside box", (x, max(15, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Preprocess: try MediaPipe-aligned ROI first ONLY if enabled, else fallback to static ROI ---
        mp_bbox = None
        arr = None
        if args.use_mediapipe and (hands is not None):
            arr, mp_bbox = mp_aligned_hand(frame, img_size, hands)
        if arr is None:
            arr = preprocess_frame(frame, roi, img_size, use_clahe=args.clahe, use_skin_mask=args.skin_mask)

        display_label = "Detecting..."

        # Optional: draw MP bbox if present (green)
        if mp_bbox is not None:
            mx, my, mw2, mh2 = mp_bbox
            cv2.rectangle(frame, (mx, my), (mx + mw2, my + mh2), (0, 255, 0), 2)

        if arr is not None:
            # Predict with/without TTA
            if args.no_tta:
                probs = model.predict(arr, verbose=0)[0]
            else:
                tta_batch = make_tta_batch(arr, args.tta_angles, args.tta_flip)
                probs = model.predict(tta_batch, verbose=0).mean(axis=0)
            probs = calibrated(probs, temperature)

            # EMA smoothing
            if probs_ema is None:
                probs_ema = probs.copy()
            else:
                probs_ema = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * probs_ema
            probs = probs_ema

            # Margin gating (Top-1 vs Top-2)
            top2 = np.argpartition(probs, -2)[-2:]
            t1, t2 = top2[np.argsort(probs[top2])][::-1]  # best, second-best
            margin = float(probs[t1] - probs[t2])
            # class-specific margin: require a bigger gap for confusable letters
            BASE_MARGIN = 0.10
            HARD_MARGIN = 0.15
            hard_classes = {'X', 'U', 'S', 'V', 'E'}  # from your confusion analysis
            c1 = idx_to_label[int(t1)]
            c2 = idx_to_label[int(t2)]
            min_margin = HARD_MARGIN if (c1 in hard_classes or c2 in hard_classes) else BASE_MARGIN

            # Top-1 only
            idx_top = int(t1)
            best_label = idx_to_label[idx_top]
            best_conf = float(probs[idx_top])

            # letters-only filter + thresholds
            pred_label = None
            if not (args.letters_only and (not best_label.isalpha() or len(best_label) != 1)):
                if best_conf >= args.conf_thresh and margin >= min_margin:
                    pred_label = best_label

            # update history and stability only if confident
            if pred_label is not None:
                recent.append(pred_label)
                if pred_label == stable_label:
                    stable_count += 1
                else:
                    stable_label = pred_label
                    stable_count = 1

            # Choose display label: below the calibrated threshold, say so instead of guessing
            if pred_label is None and best_conf < args.conf_thresh:
                display_label = f"Not confident ({best_conf * 100:.0f}%)"
            elif stable_count >= LOCK_FRAMES and stable_label is not None:
                conf_disp = float(probs[label_to_idx.get(stable_label, idx_top)])
                display_label = f"{stable_label} ({conf_disp * 100:.0f}%)"
            elif len(recent) > 0:
                tmp = majority_vote(recent)
                conf_tmp = float(probs[label_to_idx.get(tmp, idx_top)]) if tmp else best_conf
                display_label = f"{tmp} ({conf_tmp * 100:.0f}%)" if tmp else "Detecting..."
            else:
                display_label = "Detecting..."

        # --- Dynamic gesture tracking (does not alter static ROI pipeline) ---
        dyn_detected = None
        if args.dynamic and (hands is not None):
            tip, _bbox = mp_tip_and_bbox(frame, hands)
            if tip:
                tip_traj.append(tip)
                if dyn_cooldown > 0:
                    dyn_cooldown -= 1
                else:
                    dyn = detect_dynamic_JZ(list(tip_traj), min_move=args.dynamic_min_move)
                    if dyn in ('J', 'Z'):
                        dyn_detected = dyn
                        tip_traj.clear()
                        dyn_cooldown = args.dynamic_cooldown
                        # commit dynamic letter immediately
                        running_text += dyn

        # Key handling (robust on Windows)
        key = cv2.waitKeyEx(1) & 0xFFFF
        ch = chr(key & 0xFF).lower() if 0 <= key <= 255 else ""

        if ch == 'a' and majority_vote(recent):
            running_text += majority_vote(recent)
        elif ch == ' ':
            running_text += ' '
        elif ch == 'd' or key in (8, 127):  # Backspace/Delete
            if running_text:
                running_text = running_text[:-1]
        elif ch == 'c':
            running_text = ""
        elif ch == '+':
            rw = int(rw * 1.1); rh = int(rh * 1.1)
            rw = min(rw, W - x); rh = min(rh, H - y)
            roi = (x, y, rw, rh)
        elif ch == '-':
            rw = int(rw / 1.1); rh = int(rh / 1.1)
            rw = max(10, rw);   rh = max(10, rh)
            roi = (x, y, rw, rh)
        elif ch == 's':  # save current ROI for debugging
            if arr is not None:
                dbg = arr[0].clip(0, 255).astype(np.uint8)  # RGB HxW, already 0..255
                bgr = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)
                cv2.imwrite("debug_roi.jpg", bgr)
                print("[DEBUG] Saved ROI to debug_roi.jpg")
        elif ch == 'q':
            break

        # Overlays
        cv2.putText(frame, f"Pred: {display_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        display_running = running_text.replace(' ', '␣') if args.show_spaces else running_text
        cv2.putText(frame, f"Text: {display_running}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if dyn_detected:
            cv2.putText(frame, f"Dyn: {dyn_detected}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("ASL Recognizer", frame)

    cap.release()
    if hands is not None:
        hands.close()
    cv2.destroyAllWindows()
