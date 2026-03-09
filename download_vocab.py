import requests
import os

url = 'https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz'
destination = r'e:\复旦大学baseline\MMFN_yyt\bpe_simple_vocab_16e6.txt.gz'

print(f"Starting download from {url}...")
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes (4xx, 5xx)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Successfully downloaded to {destination}")

except requests.exceptions.RequestException as e:
    print(f"Error downloading file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
