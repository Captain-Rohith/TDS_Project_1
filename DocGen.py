import os
import json

MARKDOWN_DIR = "markdowns"  # Replace with your actual folder if different
OUTPUT_FILE = "documents.json"

documents = []

# Walk through all files in the directory
for root, dirs, files in os.walk(MARKDOWN_DIR):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        documents.append({
                            "content": content,
                            "source": file_path
                        })
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")

# Write to documents.json
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    json.dump(documents, outfile, indent=2)

print(f"✅ Created {OUTPUT_FILE} with {len(documents)} markdown documents.")
