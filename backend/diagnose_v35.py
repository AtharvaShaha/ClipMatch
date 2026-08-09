"""
Diagnostic: Compare Query_Video35 clip against all references.
Shows raw scoring metrics for each reference to identify what differentiates.
"""
import sys, os, time
sys.path.insert(0, '.')

import numpy as np
from services.video_processor import VideoProcessor
from services.feature_extractor import FeatureExtractor
from services.matcher import _vectorised_hamming, _build_int_arrays
from models.database import get_session, ReferenceVideo, VideoFrame

vp = VideoProcessor()
fe = FeatureExtractor()

clip_path = r"D:\Desktop\mismatch\Query video\Query_Video35(2.50-3.10).mp4"
print(f"Using: {clip_path}\n")

# Extract features from clip
print("Extracting frames...")
frames = vp.extract_all_frames_to_list(clip_path, max_frames=20, query_mode=True)
print(f"Got {len(frames)} frames")

features = []
for _, _, frame in frames:
    feat = fe.extract_features(frame)
    features.append(feat)

# Get all references
session = get_session()
refs = session.query(ReferenceVideo).filter_by(status='indexed').all()
print(f"Comparing against {len(refs)} references...\n")

results = []

for ref in refs:
    ref_frames = session.query(VideoFrame).filter_by(video_id=ref.id).order_by(VideoFrame.timestamp).all()
    if not ref_frames:
        continue
    
    hashes_raw = [(f.id, f.timestamp, f.phash, f.dhash, f.whash,
                   f.brightness, f.avg_color_r, f.avg_color_g, f.avg_color_b) for f in ref_frames]
    ph, dh, wh, ts, br, cr, cg, cb = _build_int_arrays(hashes_raw)
    
    step = 3
    ph_s, dh_s, wh_s = ph[::step], dh[::step], wh[::step]
    
    distances = []
    for feat in features:
        ph_int = int(feat['phash'], 16)
        dh_int = int(feat['dhash'], 16)
        wh_int = int(feat['whash'], 16)
        
        ph_dist = _vectorised_hamming(ph_int, ph_s)
        dh_dist = _vectorised_hamming(dh_int, dh_s)
        wh_dist = _vectorised_hamming(wh_int, wh_s)
        avg_dist = (ph_dist.astype(np.float64) + dh_dist + wh_dist) / 3.0
        distances.append(float(np.min(avg_dist)))
    
    avg_d = np.mean(distances)
    min_d = np.min(distances)
    median_d = np.median(distances)
    pct_20 = sum(1 for d in distances if d <= 20) / len(distances) * 100
    pct_24 = sum(1 for d in distances if d <= 24) / len(distances) * 100
    pct_32 = sum(1 for d in distances if d <= 32) / len(distances) * 100
    
    results.append({
        'name': ref.filename, 'avg': avg_d, 'min': min_d, 'med': median_d,
        'p20': pct_20, 'p24': pct_24, 'p32': pct_32,
    })

results.sort(key=lambda x: x['avg'])

print(f"{'#':<4} {'Video':<25} {'AvgDist':<9} {'MinDist':<9} {'MedDist':<9} {'<20%':<7} {'<24%':<7} {'<32%':<7}")
print("-" * 78)
for i, r in enumerate(results[:25], 1):
    m = " <<<" if "Video35" in r['name'] else ""
    print(f"{i:<4} {r['name']:<25} {r['avg']:<9.1f} {r['min']:<9.1f} {r['med']:<9.1f} {r['p20']:<7.0f} {r['p24']:<7.0f} {r['p32']:<7.0f}{m}")

for i, r in enumerate(results, 1):
    if "Video35" in r['name']:
        if i > 25:
            print(f"\n{i:<4} {r['name']:<25} {r['avg']:<9.1f} {r['min']:<9.1f} {r['med']:<9.1f} {r['p20']:<7.0f} {r['p24']:<7.0f} {r['p32']:<7.0f} <<<")
        print(f"\n=== Video35 rank: #{i}/{len(results)} ===")
        break

session.close()
