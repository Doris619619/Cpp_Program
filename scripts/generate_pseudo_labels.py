import argparse
import os
import random
from pathlib import Path
from ultralytics import YOLO
import json
import cv2

# Generate pseudo labels for frames without manual person annotations.
# Scheme B: Use seat occupancy JSON to synthesize box if seat appears occupied but no detection overlaps.
# YOLO txt format: class x_center y_center width height (normalized)
# Assumptions:
#   - Manual labels are in <root>/labels_manual (filename.txt matching image stem)
#   - Manual images in <root>/images_manual
#   - Seat state JSON files are in --seat-json-dir with names like f_000123.json and contain seat polygons + states.
#   - Frames directory contains source images (jpg/png) used for pseudo labeling.
# Output:
#   - Pseudo images copied to <root>/images_pseudo
#   - Pseudo labels written to <root>/labels_pseudo
#   - A summary JSON written to <root>/pseudo_summary.json

def load_seat_states(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def polygon_to_rect(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by_y1) if False else (bx2-bx1)*(by2-by1)  # keep style simple
    return inter / (area_a + area_b - inter + 1e-9)

def rect_overlap_any(r, rects, thres=0.05):
    for rr in rects:
        if iou(r, rr) >= thres:
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='fine_tuning root path')
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--seat-json-dir', help='dir containing seat state JSON (optional)')
    ap.add_argument('--max-images', type=int, default=0, help='limit pseudo labeling for debugging')
    args = ap.parse_args()

    root = Path(args.root)
    frames_dir = Path(args.frames_dir)
    manual_labels = root / 'labels_manual'
    out_images = root / 'images_pseudo'
    out_labels = root / 'labels_pseudo'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    images = []
    for p in frames_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}:
            images.append(p)
    images.sort()

    manual_stems = {p.stem for p in manual_labels.glob('*.txt')}
    done = 0
    synth_boxes_total = 0
    det_boxes_total = 0
    for img_path in images:
        if args.max_images and done >= args.max_images:
            break
        if img_path.stem in manual_stems:
            continue  # skip manual annotated frames
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        res = model.predict(source=str(img_path), conf=args.conf, classes=[0], verbose=False)
        preds = res[0]  # ultralytics Result
        det_rects = []
        h, w = im.shape[:2]
        for box in preds.boxes:
            # box.xyxy in pixels
            xyxy = box.xyxy[0].tolist()
            x1,y1,x2,y2 = xyxy
            det_rects.append((x1,y1,x2,y2))
        synth_rects = []
        # Scheme B synthetic boxes using seat occupancy states
        if args.seat_json_dir:
            seat_json = Path(args.seat_json_dir) / f"{img_path.stem}.json"
            seat_data = load_seat_states(seat_json)
            if seat_data and isinstance(seat_data, dict) and 'seats' in seat_data:
                for seat in seat_data['seats']:
                    state = seat.get('state','')
                    poly = seat.get('seat_poly') or seat.get('poly') or []
                    if not poly or state.lower() not in {'occupied','object_only','object'}:
                        continue
                    x1,y1,x2,y2 = polygon_to_rect(poly)
                    # shrink slightly
                    shrink = 0.05
                    dx = (x2 - x1) * shrink; dy = (y2 - y1) * shrink
                    rx1 = max(0, x1+dx); ry1 = max(0, y1+dy)
                    rx2 = min(w-1, x2-dx); ry2 = min(h-1, y2-dy)
                    rect = (rx1, ry1, rx2, ry2)
                    if not rect_overlap_any(rect, det_rects, thres=0.05):
                        synth_rects.append(rect)
        # write label file
        if not det_rects and not synth_rects:
            continue  # skip empty
        # copy image
        dst_img = out_images / img_path.name
        cv2.imwrite(str(dst_img), im)
        yolo_lines = []
        for (x1,y1,x2,y2) in det_rects + synth_rects:
            cx = (x1 + x2)/2.0; cy = (y1 + y2)/2.0; bw = (x2 - x1); bh = (y2 - y1)
            yolo_lines.append(f"0 {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        dst_lbl = out_labels / f"{img_path.stem}.txt"
        with open(dst_lbl,'w',encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
        det_boxes_total += len(det_rects)
        synth_boxes_total += len(synth_rects)
        done += 1
    summary = {
        'images_written': done,
        'det_boxes_total': det_boxes_total,
        'synth_boxes_total': synth_boxes_total,
        'conf': args.conf,
        'model': args.model
    }
    with open(root/'pseudo_summary.json','w',encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Pseudo labeling summary:', summary)

if __name__ == '__main__':
    main()
