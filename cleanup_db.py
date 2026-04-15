import sqlite3

conn = sqlite3.connect('data/clipmatch.db')
cursor = conn.cursor()

# Delete the stale video record
cursor.execute('DELETE FROM video_frames WHERE video_id = 1')
cursor.execute('DELETE FROM reference_videos WHERE id = 1')

conn.commit()
print('✓ Deleted stale videoplayback.mp4 record from database')
print('✓ You can now upload and index videoplayback.mp4 again')

conn.close()
