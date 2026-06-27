import sqlite3

conn = sqlite3.connect('data/clipmatch.db')
cursor = conn.cursor()
cursor.execute('SELECT id, filename, title FROM reference_videos WHERE filename LIKE "%videoplayback%"')
rows = cursor.fetchall()

if rows:
    print('Found indexed videos:')
    for row in rows:
        print(f'  ID: {row[0]}, File: {row[1]}, Title: {row[2]}')
else:
    print('No videos found with videoplayback in filename')
    print('\nAll indexed videos:')
    cursor.execute('SELECT id, filename FROM reference_videos')
    all_rows = cursor.fetchall()
    for row in all_rows:
        print(f'  ID: {row[0]}, File: {row[1]}')

conn.close()
