import os
import re

SOURCE_DIR = './blogs/'
DEST_DIR = '/mnt/e/data/multilingual/English-Blogs.txt'

def parse_blog_xml(file_path):
    """Parse potentially malformed XML blog files using regex patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
            
        # Use regex to extract content between <post> tags
        post_pattern = re.compile(r'<post>\s*(.*?)\s*</post>', re.DOTALL)
        posts = post_pattern.findall(content)
        
        for post in posts:
            yield post
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

if os.path.exists(DEST_DIR):
    os.remove(DEST_DIR)
    
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        with open(DEST_DIR, 'a') as dest_file:
            for post in parse_blog_xml(file_path):
                dest_file.write(post + '\n')
