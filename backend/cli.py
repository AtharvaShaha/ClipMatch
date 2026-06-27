"""
ClipMatch CLI Interface
Terminal-based interface for uploading videos and viewing match results
"""

import os
import sys
import json
from pathlib import Path
from colorama import Fore, Back, Style, init
from tabulate import tabulate
from typing import Optional, Dict
import time
import logging

# Configure clipmatch logging
log_level = logging.DEBUG if os.environ.get('CLIPMATCH_DEBUG') else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
# Quiet down SQLAlchemy unless debugging
logging.getLogger('sqlalchemy').setLevel(logging.WARNING)

# Suppress OpenCV warnings about device enumeration
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

# Initialize colorama for cross-platform colors
init(autoreset=True)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.advanced_matcher import advanced_matcher
from services.matcher import ClipMatcher
from services.indexer import VideoIndexer
from models.database import get_session, ReferenceVideo, MatchResult
from config import UPLOADS_DIR, REFERENCES_DIR


def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print application header"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}█████████████████████████████████████████████████████████████████████")
    print(f"{Fore.CYAN}{Fore.GREEN}  ██████╗██╗     ██╗██████╗ ███╗   ███╗ █████╗ ████████╗  ")
    print(f"{Fore.CYAN}{Fore.GREEN}  ██╔════╝██║     ██║██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝  ")
    print(f"{Fore.CYAN}{Fore.GREEN}  ██║     ██║     ██║██████╔╝██╔████╔██║███████║   ██║     ")
    print(f"{Fore.CYAN}{Fore.GREEN}  ██║     ██║     ██║██╔═══╝ ██║╚██╔╝██║██╔══██║   ██║     ")
    print(f"{Fore.CYAN}{Fore.GREEN}  ╚██████╗███████╗██║██║     ██║ ╚═╝ ██║██║  ██║   ██║     ")
    print(f"{Fore.CYAN}{Fore.GREEN}   ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝     ")
    print(f"{Fore.CYAN}█████████████████████████████████████████████████████████████████████")
    print(f"{Fore.CYAN}  Restricted Source Video Matching System - Terminal Edition")
    print(f"{Fore.CYAN}{'='*70}\n")


def print_menu():
    """Print main menu"""
    print(f"{Fore.YELLOW}┌─ MAIN MENU ──────────────────────────────────────────────────────┐")
    print(f"{Fore.YELLOW}│ 1. Upload Query Video & Find Matches                              │")
    print(f"{Fore.YELLOW}│ 2. Upload & Index Reference Videos (Build Database)               │")
    print(f"{Fore.YELLOW}│ 3. View Recent Match History                                      │")
    print(f"{Fore.YELLOW}│ 4. View Indexed Reference Videos                                  │")
    print(f"{Fore.YELLOW}│ 5. Exit                                                            │")
    print(f"{Fore.YELLOW}└────────────────────────────────────────────────────────────────────┘\n")


def upload_reference_videos():
    """Handle uploading and indexing reference videos"""
    print(f"\n{Fore.CYAN}📚 BUILD REFERENCE DATABASE\n")
    
    while True:
        print(f"{Fore.YELLOW}Enter path to reference video file (or 'done' to finish): {Style.RESET_ALL}", end="")
        video_path = input().strip()
        
        if video_path.lower() == 'done':
            break
        
        # Validate path
        if not os.path.exists(video_path):
            print(f"{Fore.RED}❌ Error: File not found: {video_path}\n")
            continue
        
        if not video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
            print(f"{Fore.RED}❌ Error: Unsupported video format. Supported: mp4, avi, mkv, mov, flv, wmv\n")
            continue
        
        # Get optional title
        print(f"{Fore.YELLOW}Enter title for this video (or press Enter to use filename): {Style.RESET_ALL}", end="")
        title = input().strip() or os.path.basename(video_path)
        
        # Copy to references directory
        filename = os.path.basename(video_path)
        ref_path = os.path.join(REFERENCES_DIR, filename)
        
        try:
            print(f"{Fore.CYAN}📋 Copying video... ", end="", flush=True)
            import shutil
            shutil.copy2(video_path, ref_path)
            print(f"{Fore.GREEN}✓\n")
        except Exception as e:
            print(f"{Fore.RED}❌ Error: Could not copy file: {e}\n")
            continue
        
        # Index the video
        print(f"{Fore.CYAN}🔍 Indexing video: {title}")
        print(f"{Fore.CYAN}This may take a moment...\n")
        
        indexer = VideoIndexer()
        
        def progress_callback(current, total, message):
            """Callback for indexing progress"""
            if total > 0:
                percent = (current / total) * 100
                bar_length = 40
                filled = int((percent / 100) * bar_length)
                bar = f"{Fore.GREEN}{'█' * filled}{Fore.WHITE}{'░' * (bar_length - filled)}"
                print(f"\r{Fore.CYAN}{message} {bar} {percent:.0f}%", end="", flush=True)
        
        try:
            result = indexer.index_video(ref_path, title=title, progress_callback=progress_callback)
            print()  # Newline after progress
            
            if result.get('success'):
                print(f"{Fore.GREEN}✅ Video indexed successfully!\n")
                print(f"{Fore.WHITE}  Video ID: {result.get('video_id')}")
                print(f"{Fore.WHITE}  Frames indexed: {result.get('frames_indexed')}")
                print(f"{Fore.WHITE}  Processing time: {result.get('processing_time', 0):.2f}s\n")
            else:
                print(f"{Fore.RED}❌ Indexing failed: {result.get('error', 'Unknown error')}\n")
                # Clean up the copied file if indexing failed
                try:
                    os.remove(ref_path)
                except:
                    pass
        except Exception as e:
            print(f"\n{Fore.RED}❌ Error during indexing: {str(e)}\n")
            import traceback
            traceback.print_exc()
            try:
                os.remove(ref_path)
            except:
                pass
    
    # Show final database status
    print(f"\n{Fore.CYAN}📊 REFERENCE DATABASE STATUS\n")
    show_indexed_videos()


def show_indexed_videos():
    """Display all indexed reference videos"""
    session = get_session()
    try:
        videos = session.query(ReferenceVideo).filter(
            ReferenceVideo.status == 'indexed'
        ).all()
        
        if not videos:
            print(f"{Fore.YELLOW}⚠️  No indexed reference videos found.")
            return
        
        print(f"\n{Fore.CYAN}📹 INDEXED REFERENCE VIDEOS ({len(videos)} total)\n")
        
        video_data = []
        for video in videos:
            video_data.append([
                video.id,
                video.filename,
                f"{video.frame_count}",
                f"{video.duration:.2f}s",
                f"{video.fps:.2f}",
                f"{video.file_size / (1024*1024):.2f}MB"
            ])
        
        headers = ["ID", "Filename", "Frames", "Duration", "FPS", "Size"]
        print(tabulate(video_data, headers=headers, tablefmt="grid"))
        print()
        
    finally:
        session.close()


def show_match_history():
    """Display recent match history"""
    session = get_session()
    try:
        matches = session.query(MatchResult).order_by(
            MatchResult.created_at.desc()
        ).limit(10).all()
        
        if not matches:
            print(f"{Fore.YELLOW}⚠️  No match history found.")
            return
        
        print(f"\n{Fore.CYAN}📊 RECENT MATCH HISTORY (Last 10)\n")
        
        match_data = []
        for match in matches:
            match_data.append([
                match.id,
                match.query_filename[:20],
                f"{match.confidence_score:.1f}%",
                f"{match.start_timestamp:.2f}s",
                f"{match.end_timestamp:.2f}s",
                match.created_at.strftime("%H:%M:%S")
            ])
        
        headers = ["ID", "Query", "Confidence", "Start", "End", "Time"]
        print(tabulate(match_data, headers=headers, tablefmt="grid"))
        print()
        
    finally:
        session.close()


def select_matching_mode():
    """Let user choose between original quality or edited/compressed matching"""
    print(f"\n{Fore.CYAN}🎯 SELECT MATCHING MODE\n")
    print(f"{Fore.YELLOW}┌─ MATCHING OPTIONS ──────────────────────────────────────┐")
    print(f"{Fore.YELLOW}│ 1. Original Quality (Standard Matcher)                   │")
    print(f"{Fore.YELLOW}│    - For videos with minimal compression                  │")
    print(f"{Fore.YELLOW}│    - Fast and accurate                                    │")
    print(f"{Fore.YELLOW}│                                                          │")
    print(f"{Fore.YELLOW}│ 2. Edited/Compressed (Advanced Matcher)                  │")
    print(f"{Fore.YELLOW}│    - For re-encoded, compressed, or watermarked videos    │")
    print(f"{Fore.YELLOW}│    - More robust to modifications                         │")
    print(f"{Fore.YELLOW}└──────────────────────────────────────────────────────────┘\n")
    
    mode = input(f"{Fore.YELLOW}Choose matching mode (1 or 2): {Style.RESET_ALL}").strip()
    
    if mode == '1':
        return 'original'
    elif mode == '2':
        return 'advanced'
    else:
        print(f"{Fore.RED}❌ Invalid choice. Using Advanced Matcher by default.\n")
        return 'advanced'


def upload_and_match():
    """Handle video upload and matching"""
    print(f"\n{Fore.CYAN}📤 UPLOAD QUERY VIDEO\n")
    
    # Get video path from user
    video_path = input(f"{Fore.YELLOW}Enter path to query video file: {Style.RESET_ALL}").strip()
    
    # Validate path
    if not os.path.exists(video_path):
        print(f"{Fore.RED}❌ Error: File not found: {video_path}\n")
        return
    
    if not video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')):
        print(f"{Fore.RED}❌ Error: Unsupported video format. Supported: mp4, avi, mkv, mov, flv, wmv\n")
        return
    
    # Copy to uploads directory
    filename = os.path.basename(video_path)
    upload_path = os.path.join(UPLOADS_DIR, filename)
    
    try:
        print(f"{Fore.CYAN}📋 Preparing video... ", end="", flush=True)
        import shutil
        shutil.copy2(video_path, upload_path)
        print(f"{Fore.GREEN}✓\n")
    except Exception as e:
        print(f"{Fore.RED}❌ Error: Could not copy file: {e}\n")
        return
    
    # Select matching mode
    matching_mode = select_matching_mode()
    
    # Run matching
    matcher_type = "Original Quality Matcher" if matching_mode == 'original' else "Advanced Matcher"
    print(f"{Fore.CYAN}🔍 Analyzing video against {get_indexed_video_count()} reference videos...")
    print(f"{Fore.CYAN}Using {matcher_type}...")
    print(f"{Fore.CYAN}This may take a moment...\n")
    
    start_time = time.time()
    
    try:
        if matching_mode == 'original':
            # Use standard matcher for original quality videos
            matcher = ClipMatcher()
            result = matcher.match_clip(upload_path)
        else:
            # Use advanced matcher for edited/compressed videos
            result = advanced_matcher.match_clip(upload_path)
        
        processing_time = time.time() - start_time
        
        display_match_results(result, processing_time)
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error during matching: {str(e)}\n")
        import traceback
        traceback.print_exc()


def get_indexed_video_count():
    """Get count of indexed videos"""
    session = get_session()
    try:
        count = session.query(ReferenceVideo).filter(
            ReferenceVideo.status == 'indexed'
        ).count()
        return count
    finally:
        session.close()


def display_match_results(result: Dict, processing_time: float):
    """Display matching results in terminal"""
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.GREEN}{'ANALYSIS COMPLETE'.center(70)}")
    print(f"{Fore.CYAN}{'='*70}\n")
    
    if not result.get('success'):
        print(f"{Fore.RED}❌ Analysis Failed\n")
        print(f"{Fore.YELLOW}Error: {result.get('error', 'Unknown error')}\n")
        return
    
    # Query info
    query_info = result.get('query', {})
    print(f"{Fore.CYAN}📁 QUERY VIDEO INFO")
    print(f"{Fore.WHITE}  Filename: {query_info.get('filename')}")
    print(f"{Fore.WHITE}  Duration: {query_info.get('duration', 0):.2f} seconds")
    print(f"{Fore.WHITE}  Frames Analyzed: {query_info.get('frames_analyzed', 0)}")
    print(f"{Fore.WHITE}  Processing Time: {processing_time:.2f} seconds\n")
    
    # Best match
    best_match = result.get('best_match')
    all_matches = result.get('all_matches', [])
    
    if best_match:
        print(f"{Fore.GREEN}{'█'*70}")
        print(f"{Fore.GREEN}  ✅ MATCH FOUND\n")
        
        # Confidence gauge
        confidence = best_match.get('confidence', 0)
        confidence_bar = get_confidence_bar(confidence)
        
        print(f"{Fore.WHITE}  CONFIDENCE SCORE")
        print(f"{Fore.GREEN}  {confidence_bar}")
        print(f"{Fore.GREEN}  {confidence:.1f}%\n")
        
        # Match quality
        quality = "🟢 VERY STRONG MATCH" if confidence >= 95 else \
                  "🟢 STRONG MATCH" if confidence >= 85 else \
                  "🟡 MODERATE MATCH" if confidence >= 70 else \
                  "🟠 WEAK MATCH" if confidence >= 50 else \
                  "🔴 POOR MATCH"
        print(f"{Fore.WHITE}  {quality}\n")
        
        # Match details
        print(f"{Fore.CYAN}  📍 MATCH DETAILS")
        print(f"{Fore.WHITE}    Source Video: {best_match.get('video_title', 'Unknown')}")
        start_time = seconds_to_mmss(best_match.get('start_timestamp', 0))
        end_time = seconds_to_mmss(best_match.get('end_timestamp', 0))
        duration = best_match.get('end_timestamp', 0) - best_match.get('start_timestamp', 0)
        duration_mmss = seconds_to_mmss(duration)
        print(f"{Fore.WHITE}    Timestamp Range: {start_time} - {end_time}")
        print(f"{Fore.WHITE}    Duration: {duration_mmss}")
        print(f"{Fore.WHITE}    Match ID: {best_match.get('video_id', 'N/A')}\n")
        
        print(f"{Fore.GREEN}{'█'*70}\n")
    else:
        print(f"{Fore.YELLOW}⚠️  NO MATCHES FOUND\n")
        print(f"{Fore.WHITE}The query video did not match any indexed reference videos.\n")
    
    # All matches summary
    if all_matches:
        print(f"{Fore.CYAN}📊 ALL MATCHES ({len(all_matches)} total)\n")
        
        match_data = []
        for i, match in enumerate(all_matches[:10], 1):  # Show top 10
            match_data.append([
                i,
                match.get('video_title', 'Unknown')[:25],
                f"{match.get('confidence', 0):.1f}%",
                seconds_to_mmss(match.get('start_timestamp', 0)),
                seconds_to_mmss(match.get('end_timestamp', 0))
            ])
        
        headers = ["#", "Video", "Confidence", "Start", "End"]
        print(tabulate(match_data, headers=headers, tablefmt="simple"))
        print()
    
    # Save to database
    if best_match and 'match_id' in result:
        print(f"{Fore.GREEN}✓ Result saved to database (ID: {result['match_id']})\n")


def get_confidence_bar(confidence: float, length: int = 40) -> str:
    """Generate a visual confidence bar"""
    filled = int((confidence / 100) * length)
    empty = length - filled
    
    # Color based on confidence
    if confidence >= 95:
        color = Fore.GREEN
    elif confidence >= 85:
        color = Fore.GREEN
    elif confidence >= 70:
        color = Fore.YELLOW
    elif confidence >= 50:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    
    bar = f"{color}{'█' * filled}{Fore.WHITE}{'░' * empty}"
    return f"  [{bar}{Fore.WHITE}]"


def seconds_to_mmss(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes}:{secs:02d}"


def main():
    """Main CLI loop"""
    clear_screen()
    print_header()
    
    while True:
        print_menu()
        choice = input(f"{Fore.YELLOW}Enter your choice (1-5): {Style.RESET_ALL}").strip()
        
        if choice == '1':
            clear_screen()
            print_header()
            upload_and_match()
        
        elif choice == '2':
            clear_screen()
            print_header()
            upload_reference_videos()
        
        elif choice == '3':
            clear_screen()
            print_header()
            show_match_history()
        
        elif choice == '4':
            clear_screen()
            print_header()
            show_indexed_videos()
        
        elif choice == '5':
            print(f"\n{Fore.GREEN}👋 Thank you for using ClipMatch. Goodbye!\n")
            break
        
        else:
            print(f"{Fore.RED}❌ Invalid choice. Please enter 1-5.\n")
        
        input(f"{Fore.YELLOW}Press Enter to continue... {Style.RESET_ALL}")
        clear_screen()
        print_header()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}\n⚠️  Application interrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
