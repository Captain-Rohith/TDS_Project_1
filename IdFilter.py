import requests
import json
from urllib.parse import urlencode
from datetime import datetime, timezone


with open("cookie.txt", "r") as file:
    cookie_string = file.read().strip()


headers = {
    "Cookie": cookie_string,
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*"
}


start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 15, tzinfo=timezone.utc)


base_url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json"
page = 0
matched_topics = []
stop_loop = False


while not stop_loop:
    params = {
        "ascending": "false",
        "order": "activity",
        "page": page
    }

    full_url = f"{base_url}?{urlencode(params)}"
    response = requests.get(full_url, headers=headers)

    if response.status_code != 200:
        print(f"Request failed at page {page} with status {response.status_code}")
        break

    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to parse JSON.")
        break

    topics = data.get("topic_list", {}).get("topics", [])
    if not topics:
        print(" No more topics. Stopping.")
        break

    for topic in topics:
        last_posted_at = topic.get("last_posted_at")
        if not last_posted_at:
            continue

        posted_time = datetime.fromisoformat(last_posted_at.replace("Z", "+00:00"))

        if start_date <= posted_time <= end_date:
            matched_topics.append({
                "id": topic["id"],
                "title": topic["title"]
            })
        elif posted_time < start_date:
            stop_loop = True
            break

    print(f"-> Page {page} processed. Total matches: {len(matched_topics)}")
    page += 1


with open("filtered_topics.json", "w", encoding="utf-8") as f:
    json.dump(matched_topics, f, indent=2, ensure_ascii=False)

print("Done! Topics saved to filtered_topics.json")
