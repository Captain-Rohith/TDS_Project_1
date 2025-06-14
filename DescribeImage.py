import os
import json
import requests
import time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini client

client = genai.Client(api_key=GOOGLE_API_KEY)

# Load full post JSON
with open("full_posts.json", "r", encoding="utf-8") as f:
    full_posts = json.load(f)

# Filter out avatar/profile image URLs
def is_valid_image_url(url):
    return (
        url.startswith("http")
        and "user_avatar" not in url
        and not url.startswith("https://emoji.discourse-cdn.com")
    )

# Describe image using Gemini Flash
def describe_image_from_url(image_url):
    try:
        image_bytes = requests.get(image_url, timeout=10).content
        image = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=["Describe the image and summarise the content in the image. The Description will be used in a knowledge base for RAG. NOTE: Give the exact answer only. No metadata, no intro, no explanations, no ellipsis.", image],
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error describing image: {image_url}\n{e}")
        return "Image description unavailable."

# Process posts and replace images
for topic in full_posts:
    posts = topic.get("post_stream", {}).get("posts", [])
    for post in posts:
        cooked_html = post.get("cooked", "")
        if not cooked_html:
            continue

        soup = BeautifulSoup(cooked_html, "html.parser")
        imgs = soup.find_all("img")

        for img in imgs:
            src = img.get("src", "")
            if is_valid_image_url(src):
                print(f"🔍 Describing image: {src}")
                description = describe_image_from_url(src)
                markdown_image = f"![{description}]"
                img.replace_with(markdown_image)
                time.sleep(7)  # To avoid hitting rate limits
            else:
                img.decompose()  # Remove avatar or invalid images

        # Update the post's cooked field
        post["cooked"] = str(soup)

# Save the updated posts to a new JSON file
with open("full_posts_with_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(full_posts, f, indent=2)

print("✅ Replaced image URLs with descriptions. Saved to 'full_posts_with_descriptions.json'")
