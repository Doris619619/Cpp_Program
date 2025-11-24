import argparse, os, random, shutil, json, time
from pathlib import Path
from ultralytics import YOLO

# Fine-tune YOLO for single 'person' class using manual + pseudo labels.
# Expected directory layout before running:
#   <root>/images_manual, <root>/labels_manual
#   <root>/images_pseudo, <root>/labels_pseudo (optional)
# Will create:
#   <root>/images/train, <root>/images/val, <root>/labels/train, <root>/labels/val
#   data_person.yaml (updated if necessary)
# Logs:
#   logs/fine-tuning/train_metrics.json
#   logs/fine-tuning/run_summary.json
# Exported ONNX placed at existing model directory path passed via --onnx-out

def prepare_split(root: Path, val_ratio: float):
    images_manual = list((root/'images_manual').glob('*.jpg')) + list((root/'images_manual').glob('*.png'))
    if not images_manual:
        raise RuntimeError('No manual images found.')
    random.shuffle(images_manual)
    val_count = max(1, int(len(images_manual)*val_ratio))
    val_set = set(p.stem for p in images_manual[:val_count])
    train_imgs = []
    val_imgs = []
    for p in images_manual:
        (val_imgs if p.stem in val_set else train_imgs).append(p)
    # Add pseudo images to train
    for p in (root/'images_pseudo').glob('*.jpg'): train_imgs.append(p)
    for p in (root/'images_pseudo').glob('*.png'): train_imgs.append(p)
    return train_imgs, val_imgs

def copy_dataset(imgs, dst_img_dir: Path, lbl_src_dirs, dst_lbl_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in imgs:
        shutil.copy2(img, dst_img_dir / img.name)
        stem = img.stem
        lbl_path = None
        for d in lbl_src_dirs:
            cand = d / f'{stem}.txt'
            if cand.exists():
                lbl_path = cand; break
        if lbl_path:
            shutil.copy2(lbl_path, dst_lbl_dir / f'{stem}.txt')
        copied += 1
    return copied

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='fine_tuning root directory')
    ap.add_argument('--model', default='yolov8n.pt', help='base model weights')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--onnx-out', required=True, help='output onnx file path (person01.onnx)')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    root = Path(args.root)
    logs_dir = Path('logs')/'fine-tuning'
    logs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_imgs, val_imgs = prepare_split(root, args.val_ratio)
    train_img_dir = root/'images'/'train'
    val_img_dir = root/'images'/'val'
    train_lbl_dir = root/'labels'/'train'
    val_lbl_dir = root/'labels'/'val'

    copied_train = copy_dataset(train_imgs, train_img_dir, [root/'labels_manual', root/'labels_pseudo'], train_lbl_dir)
    copied_val = copy_dataset(val_imgs, val_img_dir, [root/'labels_manual'], val_lbl_dir)

    # Update YAML
    data_yaml = root/'data_person.yaml'
    train_abs = train_img_dir.resolve()
    val_abs = val_img_dir.resolve()
    with open(data_yaml,'w',encoding='utf-8') as f:
        f.write('train: '+str(train_abs).replace('\\','/')+'\n')
        f.write('val: '+str(val_abs).replace('\\','/')+'\n')
        f.write("names: ['person']\n")

    model = YOLO(args.model)
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=True, optimizer='SGD')

    # metrics (latest results are in results dictionary)
    # Ultralytics stores metrics in results.metrics (for v8). We'll extract key ones.
    metrics = {
        'train_images': len(train_imgs),
        'val_images': len(val_imgs),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
    }
    try:
        rdict = results.__dict__
        for k in ['fitness','best_fitness']:
            if k in rdict: metrics[k] = rdict[k]
        if hasattr(results, 'metrics') and isinstance(results.metrics, dict):
            for mk in ['precision','recall','map50','map','f1']:
                if mk in results.metrics: metrics[mk] = results.metrics[mk]
    except Exception:
        pass

    with open(logs_dir/'train_metrics.json','w',encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Export ONNX
    exported = model.export(format='onnx', imgsz=args.imgsz, simplify=True)
    onnx_path = Path(exported)  # model directory exported
    # Move first .onnx found to target path
    target = Path(args.onnx_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    moved = None
    for p in onnx_path.parent.glob('*.onnx'):
        shutil.copy2(p, target)
        moved = p
        break

    summary = {
        'time_elapsed_s': round(time.time()-t0,2),
        'onnx_exported_from': str(moved) if moved else None,
        'onnx_target': str(target),
        'metrics': metrics
    }
    with open(logs_dir/'run_summary.json','w',encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('Run summary:', summary)

if __name__ == '__main__':
    main()
