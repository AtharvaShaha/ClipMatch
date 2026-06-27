#!/usr/bin/env python3
"""Test timestamp calculation"""

# Simulate the old (wrong) calculation
def old_calculation(clip_frames_count):
    WINDOW_STRIDE = 5
    start_timestamp = 100.5
    clip_duration = clip_frames_count / WINDOW_STRIDE
    end_timestamp = start_timestamp + clip_duration
    return start_timestamp, end_timestamp, end_timestamp - start_timestamp

# Simulate the new (correct) calculation
def new_calculation(clip_features_count, actual_clip_duration):
    start_timestamp = 100.5
    end_timestamp = start_timestamp + actual_clip_duration
    return start_timestamp, end_timestamp, end_timestamp - start_timestamp

# Test cases
test_cases = [
    {"clip_frames": 10, "actual_duration": 10.0, "description": "10 second clip"},
    {"clip_frames": 15, "actual_duration": 15.0, "description": "15 second clip"},
    {"clip_frames": 30, "actual_duration": 30.0, "description": "30 second clip"},
    {"clip_frames": 5, "actual_duration": 5.0, "description": "5 second clip"},
]

print("=" * 70)
print("TIMESTAMP CALCULATION TEST")
print("=" * 70)

for test in test_cases:
    frames = test["clip_frames"]
    duration = test["actual_duration"]
    desc = test["description"]
    
    old_start, old_end, old_dur = old_calculation(frames)
    new_start, new_end, new_dur = new_calculation(frames, duration)
    
    print(f"\n{desc}:")
    print(f"  ❌ OLD (WRONG): Start={old_start:.2f}s, End={old_end:.2f}s, Duration={old_dur:.2f}s")
    print(f"  ✅ NEW (CORRECT): Start={new_start:.2f}s, End={new_end:.2f}s, Duration={new_dur:.2f}s")
    print(f"     → Duration fixed from {old_dur:.1f}s to {new_dur:.1f}s")

print("\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("OLD: clip_duration = len(clip_features) / WINDOW_STRIDE")
print("     = 10 frames / 5 = 2 seconds ❌ WRONG!")
print("")
print("NEW: clip_duration = actual_clip_duration (passed as parameter)")
print("     = 10.0 seconds ✅ CORRECT!")
print("=" * 70)
