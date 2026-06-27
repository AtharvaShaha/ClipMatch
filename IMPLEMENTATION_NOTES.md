# ClipMatch: Architectural Decisions & Optimizations

This document outlines the engineering rationale behind the key improvements made to the system.

## 1. Coarse-to-Fine Search Optimization
**Problem:** Comparing every query frame against every reference frame (O(N*M)) resulted in high latency (~113s for 62 videos).
**Solution:** Implemented a **Step-Search** in the primary matcher. By sampling the reference database every 4 frames (stride=4), search complexity was reduced by 75% with negligible impact on timestamp accuracy.
**Impact:** Significant reduction in processing time while maintaining high recall.

## 2. Robust Metadata Handling
**Problem:** `OpenCV`'s `CAP_PROP_FRAME_COUNT` often returns 0 for stream-based containers (MKV/MP4), breaking duration reporting.
**Solution:** Implemented a multi-stage fallback:
1. Try `CAP_PROP_POS_FRAMES`.
2. Seek to the end (AVI Ratio) to force header parsing.
3. Fallback to file-size heuristics if metadata is missing.
**Impact:** Eliminated "Duration = 0" bugs in reporting.

## 3. Probabilistic Confidence Scoring
**Problem:** Initial scoring used arbitrary heuristics, leading to inflated "100%" scores.
**Solution:** Re-weighted the confidence engine:
- **Hash Match (50%):** Primary global similarity.
- **Color Consistency (20%):** Secondary verification to prevent black-screen false positives.
- **Distance Score (30%):** Global pixel-distribution similarity.
**Constraint:** Base confidence is capped at 90%. Scores > 90% require **Temporal Consistency** (sequential frame evidence) to prove it's a video match, not just a single frame match.

## 4. Logical Threshold Softening
**Problem:** High "early rejection" thresholds caused many valid matches to fail.
**Solution:** Lowered the "Gate" requirement (e.g., `MIN_WINDOW_MATCHES` from 7 to 4). This allows the **Verification Engine** (NCC/SSIM) to evaluate more candidates, trusting the precise pixel-level math to filter out the noise.

## 5. Scalability Consideration
While SQLite was originally identified as a potential bottleneck, analysis shows that for datasets < 10,000 videos, the bottleneck is **CPU-bound Python loops**, not database I/O. Therefore, the priority was shifted to loop optimization (Phase 3) rather than moving to a more complex vector database like Milvus or FAISS.
