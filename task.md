# ClipMatch Improvement — Task Tracker

## PHASE 1 — Critical Fixes
- [x] 1.1 Fix `min_distance` NameError in matcher.py
- [x] 1.2 Add `extract_frame_at_index` to video_processor.py
- [x] 1.3 Fix metadata reporting (duration=0, frames=0)
- [x] 1.4 Remove shadowed `batch_extract` in feature_extractor.py

## PHASE 2 — Stabilization & Logging
- [x] 2.1 Add pipeline timing/profiling logs to both matchers
- [x] 2.2 Improve logging config in app.py
- [x] 2.3 Return pipeline timing in API response
- [x] 2.4 SQLite optimizations (indexes, WAL pragma)

## PHASE 3 — Performance & Confidence
- [ ] 3.1 Fix confidence score normalization (both matchers)
- [ ] 3.2 Extract shared utils (format_timestamp_range, get_confidence_label)
- [ ] 3.3 Fix early rejection scope in advanced_matcher.py

## PHASE 4 — Frontend UX
- [ ] 4.1 Pipeline progress visualization (elapsed timer, stage messages)
- [ ] 4.2 Improve result/evidence display (technical details panel)
- [ ] 4.3 Long-processing-time UX (timeout messages, cancel)
- [ ] 4.4 Limitations section in frontend
- [ ] 4.5 CSS for new elements
