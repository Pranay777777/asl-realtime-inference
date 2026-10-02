# scripts/capture_letter.py
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp


def mp_aligned_hand(frame_bgr, target_size, hands, pad_scale=1.4):
    h, w = frame_bgr.shape[:2]
    res = hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None, None
    lm = res.multi_hand_landmarks[0]
    xs = [int(p.x * w) for p in lm.landmark]
    ys = [int(p.y * h) for p in lm.landmark]
    x0, x1 = max(0, min(xs)), min(w - 1, max(xs))
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bw, bh = (x1 - x0), (y1 - y0)
    side = int(max(bw, bh) * pad_scale)
    half = side // 2
    xA, yA = max(0, cx - half), max(0, cy - half)
    xB, yB = min(w, cx + half), min(h, cy + half)
    crop = frame_bgr[yA:yB, xA:xB]
    if crop.size == 0: return None, None
    ch, cw = crop.shape[:2]
    s = min(ch, cw)
    y0r = (ch - s) // 2; x0r = (cw - s) // 2
    sq = crop[y0r:y0r+s, x0r:x0r+s]
    resized = cv2.resize(sq, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb, (xA, yA, xB - xA, yB - yA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", required=True, help="A..Z (the label to save)")
    ap.add_argument("--out_dir", default="data/custom", type=str)
    ap.add_argument("--num", default=120, type=int, help="images to capture")
    ap.add_argument("--camera", default=0, type=int)
    ap.add_argument("--img_size", nargs=2, type=int, default=[64, 64])
    ap.add_argument("--mirror", action="store_true", help="mirror webcam view")
    ap.add_argument("--every", default=4, type=int, help="save 1 of every N frames while recording")
    args = ap.parse_args()

    letter = args.letter.upper()
    assert len(letter) == 1 and letter.isalpha(), "letter must be A..Z"

    save_dir = Path(args.out_dir) / letter
    save_dir.mkdir(parents=True, exist_ok=True)

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    target_size = (args.img_size[0], args.img_size[1])  # (W,H)
    recording = False
    saved = 0
    frame_idx = 0

    print("[INFO] Controls: SPACE = start/stop capture, Q = quit. Aim for varied angles/distances/lighting.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        rgb, bbox = mp_aligned_hand(frame, (target_size[0], target_size[1]), hands)
        if bbox is not None:
            x,y,w,h = bbox
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        if recording and rgb is not None:
            if frame_idx % args.every == 0 and saved < args.num:
                out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out_path = save_dir / f"{letter}_{int(time.time()*1000)}.jpg"
                cv2.imwrite(str(out_path), out)
                saved += 1

        frame_idx += 1
        cv2.putText(frame, f"Letter: {letter}  Saved: {saved}/{args.num}  Rec:{'ON' if recording else 'OFF'}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Capture Letter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
        elif key in (ord('q'), ord('Q')):
            break
        if saved >= args.num:
            cv2.putText(frame, "DONE!", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow("Capture Letter", frame)
            cv2.waitKey(500)
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print(f"[OK] Saved {saved} images to {save_dir}")

if __name__ == "__main__":
    main()
