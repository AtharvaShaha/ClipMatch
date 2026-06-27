import os
import time
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:5000/api"
TEST_DATA_DIR = Path("data/queries") # Assume a queries dir exists or use references
RESULTS_FILE = "benchmark_results.json"

def run_benchmark():
    print("=" * 60)
    print("CLIPMATCH PERFORMANCE & ACCURACY BENCHMARK")
    print("=" * 60)
    
    # Check if API is running
    try:
        requests.get(f"{API_URL}/health")
    except:
        print("Error: API is not running at http://localhost:5000")
        return

    # Get references to use as test clips if no queries exist
    if not TEST_DATA_DIR.exists():
        print(f"Test directory {TEST_DATA_DIR} not found. Using references...")
        test_files = list(Path("data/references").glob("*.mp4"))[:5]
    else:
        test_files = list(TEST_DATA_DIR.glob("*.mp4"))

    if not test_files:
        print("No video files found for benchmark!")
        return

    results = []
    
    for video_path in test_files:
        print(f"\nTesting: {video_path.name}")
        
        # Test Standard Matcher
        start = time.time()
        with open(video_path, 'rb') as f:
            response = requests.post(
                f"{API_URL}/match", 
                files={'clip': f},
                data={'clip_quality': 'original'}
            )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            match = data.get('best_match')
            conf = match['confidence'] if match else 0
            found = match['filename'] if match else "NONE"
            print(f"  [Standard] Time: {elapsed:.2f}s | Conf: {conf:.1f}% | Match: {found}")
            results.append({
                'video': video_path.name,
                'method': 'standard',
                'time': elapsed,
                'confidence': conf,
                'match': found
            })
        else:
            print(f"  [Standard] FAILED: {response.text}")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    avg_time = sum(r['time'] for r in results) / len(results) if results else 0
    print(f"Average Request Time: {avg_time:.2f}s")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Detailed results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()
