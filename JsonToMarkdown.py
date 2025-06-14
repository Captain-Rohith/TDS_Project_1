import json
import html2text
from bs4 import BeautifulSoup


with open("full_posts_with_descriptions.json", "r", encoding="utf-8") as f:
    full_posts = json.load(f)


markdowner = html2text.HTML2Text()
markdowner.ignore_links = False
markdowner.ignore_images = False
markdowner.ignore_emphasis = False
markdowner.body_width = 0

def remove_avatar_images(soup):
    for img in soup.find_all("img"):
        if "user_avatar" in img.get("src", ""):
            img.decompose()
    return soup


with open("discourse_dump.md", "w", encoding="utf-8") as md_file:
    for topic in full_posts:
        title = topic.get("title", "Untitled Topic")
        topic_id = topic.get("id", "")
        posts = topic.get("post_stream", {}).get("posts", [])

        
        md_file.write(f"# {title}\n\n")
        md_file.write(f"**Topic ID:** {topic_id}\n\n")
        md_file.write("---\n\n")

        for post in posts:
            cooked = post.get("cooked", "")
            username = post.get("username", "Anonymous")
            created_at = post.get("created_at", "")

            if not cooked:
                continue

           
            soup = BeautifulSoup(cooked, "html.parser")

          
            soup = remove_avatar_images(soup)

            
            for tag in soup(["script", "style"]):
                tag.decompose()

           
            cooked_cleaned = str(soup)
            markdown_content = markdowner.handle(cooked_cleaned).strip()

           
            md_file.write(f"### 🗨️ Post by `{username}` on `{created_at}`\n\n")
            md_file.write(markdown_content + "\n\n")
            md_file.write("---\n\n")

print(" Markdown file 'discourse_dump.md' created")
