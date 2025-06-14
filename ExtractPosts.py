import requests
import json
from datetime import datetime
import time


with open("cookie.txt", "r") as file:
    cookie_string = file.read().strip()

headers = {
    "Cookie": cookie_string,
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}


with open("filtered_topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)


base_url = "https://discourse.onlinedegree.iitm.ac.in/t/{id}.json"

all_posts = []

for i, topic in enumerate(topics):
    topic_id = topic["id"]
    url = base_url.format(id=topic_id)

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            post_data = response.json()
            all_posts.append(post_data)
            print(f"✅ {i+1}/{len(topics)}: Fetched post ID {topic_id}")
        else:
            print(f"❌ Failed for ID {topic_id} with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching ID {topic_id}: {str(e)}")

    time.sleep(1)  # polite delay to avoid rate limiting


with open("full_posts.json", "w", encoding="utf-8") as f:
    json.dump(all_posts, f, indent=2, ensure_ascii=False)

print("🎯 Done! Full posts saved to full_posts.json")
