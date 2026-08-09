"""
Migration: Add Cloudinary columns to reference_videos table.
Run once to add cloudinary_url and cloudinary_public_id columns.
"""
import sqlite3
import sys

DB_PATH = 'D:/Desktop/mismatch/data/clipmatch.db'

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if columns already exist
    cursor.execute("PRAGMA table_info(reference_videos)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    
    added = []
    
    if 'cloudinary_url' not in existing_cols:
        cursor.execute("ALTER TABLE reference_videos ADD COLUMN cloudinary_url TEXT")
        added.append('cloudinary_url')
    
    if 'cloudinary_public_id' not in existing_cols:
        cursor.execute("ALTER TABLE reference_videos ADD COLUMN cloudinary_public_id TEXT")
        added.append('cloudinary_public_id')
    
    conn.commit()
    conn.close()
    
    if added:
        print(f"Migration complete. Added columns: {', '.join(added)}")
    else:
        print("No migration needed — columns already exist.")

if __name__ == '__main__':
    migrate()
