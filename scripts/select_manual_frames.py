import argparse, random, os, shutil
from pathlib import Path

# Select a stratified subset of raw frames for manual person box annotation.
# Raw frames expected as <frames_dir>/f_XXXXXX.jpg (exclude *_annotated.jpg)
# Strategy: uniform sampling over index range using step or percentile buckets.

def parse_index(stem: str):
    # expects f_000123 pattern
    try:
        return int(stem.split('_')[1])
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--out-dir', required=True, help='destination subset directory')
    ap.add_argument('--count', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = []
    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != '.jpg':
            continue
        # Accept raw or annotated images; prefer raw but if only annotated exists we'll use them.
        if p.name.endswith('_annotated.jpg') or p.stem.startswith('f_'):
            # When annotated, still parse index from prefix f_XXXXXX
            imgs.append(p)
    if not imgs:
        raise SystemExit('No raw frame images found.')
    # Sort by numeric index
    imgs.sort(key=lambda p: parse_index(p.stem))
    n = len(imgs)
    k = min(args.count, n)

    # Stratified selection: pick k indices spaced across range
    if k == n:
        chosen = imgs
    else:
        step = n / k
        chosen = []
        for i in range(k):
            base_idx = int(i * step + step/2)
            # introduce small jitter
            jitter = random.randint(-2,2)
            idx = max(0, min(n-1, base_idx + jitter))
            chosen.append(imgs[idx])
    # Deduplicate possible collisions
    seen = set()
    final = []
    for p in chosen:
        if p not in seen:
            final.append(p); seen.add(p)
    # If shortage due to collisions, append random others
    while len(final) < k:
        cand = random.choice(imgs)
        if cand not in seen:
            final.append(cand); seen.add(cand)

    for p in final:
        shutil.copy2(p, out_dir / p.name)
    print(f'Selected {len(final)} frames out of {n}. Written to {out_dir}')

if __name__ == '__main__':
    main()
