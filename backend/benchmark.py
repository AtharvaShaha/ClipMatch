"""
ClipMatch Benchmark Suite
Simple benchmarking for measuring matching accuracy and performance.

Usage:
    python benchmark.py                    # Run with default test clips
    python benchmark.py --clips-dir path/  # Run with custom clips directory
    python benchmark.py --profile          # Run with cProfile profiling
"""

import time
import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.matcher import ClipMatcher
from services.advanced_matcher import AdvancedClipMatcher
from config import UPLOADS_DIR, DATA_DIR

logger = logging.getLogger('clipmatch.benchmark')


def benchmark_single_match(clip_path: str, matcher, matcher_label: str) -> Dict:
    """
    Time a single match and return metrics.
    
    Args:
        clip_path: Path to query clip
        matcher: Matcher instance (ClipMatcher or AdvancedClipMatcher)
        matcher_label: Human-readable label for the matcher
        
    Returns:
        Dictionary with timing and result metrics
    """
    start = time.perf_counter()
    
    try:
        result = matcher.match_clip(clip_path)
        elapsed = time.perf_counter() - start
        
        # Extract key metrics
        best_match = result.get('best_match')
        all_matches = result.get('all_matches', result.get('matches', []))
        
        return {
            'clip': os.path.basename(clip_path),
            'matcher': matcher_label,
            'elapsed_seconds': round(elapsed, 3),
            'success': result.get('success', False),
            'num_matches': len(all_matches),
            'best_confidence': round(best_match['confidence'], 1) if best_match else 0,
            'best_video': best_match.get('video_filename', 'N/A') if best_match else 'N/A',
            'verification_method': best_match.get('verification_method', 'N/A') if best_match else 'N/A',
            'error': result.get('error'),
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            'clip': os.path.basename(clip_path),
            'matcher': matcher_label,
            'elapsed_seconds': round(elapsed, 3),
            'success': False,
            'num_matches': 0,
            'best_confidence': 0,
            'best_video': 'N/A',
            'verification_method': 'N/A',
            'error': str(e),
        }


def run_benchmark(clip_paths: List[str], 
                  run_basic: bool = True,
                  run_advanced: bool = True,
                  output_file: Optional[str] = None) -> List[Dict]:
    """
    Run benchmarks on a list of clips with both matchers.
    
    Args:
        clip_paths: List of paths to test clips
        run_basic: Whether to test the basic matcher
        run_advanced: Whether to test the advanced matcher
        output_file: Optional JSON file to save results
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    basic_matcher = ClipMatcher() if run_basic else None
    advanced_matcher = AdvancedClipMatcher() if run_advanced else None
    
    print(f"\n{'='*80}")
    print(f"ClipMatch Benchmark — {len(clip_paths)} clips")
    print(f"{'='*80}\n")
    
    for i, clip_path in enumerate(clip_paths, 1):
        clip_name = os.path.basename(clip_path)
        print(f"[{i}/{len(clip_paths)}] Testing: {clip_name}")
        
        if basic_matcher:
            r = benchmark_single_match(clip_path, basic_matcher, "basic")
            results.append(r)
            status = "✓" if r['success'] and r['best_confidence'] > 0 else "✗"
            print(f"  Basic:    {status} {r['elapsed_seconds']:6.2f}s | conf={r['best_confidence']:5.1f}% | {r['best_video']}")
        
        if advanced_matcher:
            r = benchmark_single_match(clip_path, advanced_matcher, "advanced")
            results.append(r)
            status = "✓" if r['success'] and r['best_confidence'] > 0 else "✗"
            print(f"  Advanced: {status} {r['elapsed_seconds']:6.2f}s | conf={r['best_confidence']:5.1f}% | {r['best_video']} [{r['verification_method']}]")
        
        print()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for matcher_label in ['basic', 'advanced']:
        matcher_results = [r for r in results if r['matcher'] == matcher_label]
        if not matcher_results:
            continue
        
        avg_time = sum(r['elapsed_seconds'] for r in matcher_results) / len(matcher_results)
        matched = sum(1 for r in matcher_results if r['best_confidence'] > 0)
        avg_conf = sum(r['best_confidence'] for r in matcher_results) / len(matcher_results) if matcher_results else 0
        
        print(f"\n  {matcher_label.upper()} MATCHER:")
        print(f"    Avg time:       {avg_time:.2f}s")
        print(f"    Matched:        {matched}/{len(matcher_results)} clips")
        print(f"    Avg confidence: {avg_conf:.1f}%")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def find_test_clips(clips_dir: str = None) -> List[str]:
    """Find test clip files in a directory."""
    if clips_dir is None:
        clips_dir = str(UPLOADS_DIR)
    
    extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv'}
    clips = []
    
    for f in Path(clips_dir).iterdir():
        if f.suffix.lower() in extensions:
            clips.append(str(f))
    
    clips.sort()
    return clips


def main():
    parser = argparse.ArgumentParser(description='ClipMatch Benchmark Suite')
    parser.add_argument('--clips-dir', type=str, help='Directory containing test clips')
    parser.add_argument('--clips', nargs='+', help='Specific clip files to test')
    parser.add_argument('--basic-only', action='store_true', help='Only test basic matcher')
    parser.add_argument('--advanced-only', action='store_true', help='Only test advanced matcher')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    parser.add_argument('--profile', action='store_true', help='Run with cProfile profiling')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Quiet during benchmarks
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Find clips
    if args.clips:
        clip_paths = args.clips
    else:
        clip_paths = find_test_clips(args.clips_dir)
    
    if not clip_paths:
        print("No test clips found. Provide clips with --clips or --clips-dir.")
        sys.exit(1)
    
    # Determine which matchers to run
    run_basic = not args.advanced_only
    run_advanced = not args.basic_only
    
    # Output file
    output_file = args.output or str(DATA_DIR / 'benchmark_results.json')
    
    if args.profile:
        import cProfile
        import pstats
        
        profile_file = str(DATA_DIR / 'benchmark_profile.prof')
        print(f"Profiling enabled. Output: {profile_file}")
        
        cProfile.runctx(
            'run_benchmark(clip_paths, run_basic, run_advanced, output_file)',
            globals(), locals(),
            filename=profile_file
        )
        
        # Print top 20 functions by cumulative time
        print(f"\n{'='*80}")
        print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
        print(f"{'='*80}\n")
        stats = pstats.Stats(profile_file)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
    else:
        run_benchmark(clip_paths, run_basic, run_advanced, output_file)


if __name__ == '__main__':
    main()
