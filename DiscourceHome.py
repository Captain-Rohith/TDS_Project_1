import requests
import json
from urllib.parse import urlencode

# Step 1: Read cookie from a file
with open("cookie.txt", "r") as file:
    cookie_string = file.read().strip()
print("🔐 Cookie Loaded:", cookie_string[:80] + "...")

# Step 2: Set headers including the cookie
headers = {
    "Cookie": cookie_string,
    "User-Agent": "Mozilla/5.0",  # Optional, but safer
    "Accept": "application/json, text/plain, */*"
}

# Step 3: Define the base URL and query parameters
base_url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json"
params = {
    "ascending": "false",
    "order": "activity"
}

# Step 4: Construct full URL with encoded parameters
full_url = f"{base_url}?{urlencode(params)}"

# Step 5: Make the GET request
response = requests.get(full_url, headers=headers)

# Step 6: Handle the response
if response.status_code == 200:
    try:
        data = response.json()
        with open("response.json", "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=4)
        print("✅ JSON response saved to 'response.json'")
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON.")
else:
    print(f"❌ Request failed with status code {response.status_code}")
    print("🔍 Response text (partial):", response.text[:500])
