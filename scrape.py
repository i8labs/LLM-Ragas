import requests
from bs4 import BeautifulSoup
import time
from os import system
import csv


system("clear")

def fetch_trustpilot_reviews(url, num_pages=1):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_reviews = []

    for page in range(1, num_pages + 1):
        page_url = f"{url}?page={page}"
        response = requests.get(page_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all review containers
        try:
            review_containers = soup.find_all('div', class_='styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ')
            for container in review_containers:
                # Extract review details
                title = container.find('h2', class_='typography_heading-s__f7029 typography_appearance-default__AAY17').text.strip()
                content = container.find('p', class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn').text.strip()
                rating = container.find('img')['alt']
                date = container.find('time').text.strip()
                
                review = {
                    'title': title,
                    'content': content,
                    'rating': rating,
                    'date': date
                }
                
                all_reviews.append(review)
            
            print(f"Fetched {len(review_containers)} reviews from page {page}")
            time.sleep(1)  # Be respectful to the server
        except:
            break    

    return all_reviews

# Usage
url = "https://uk.trustpilot.com/review/www.business.natwest.com"
reviews = fetch_trustpilot_reviews(url, num_pages=10)

# Print the results
# for review in reviews:
#     print(f"Title: {review['title']}")
#     print(f"Content: {review['content']}")
#     print(f"Rating: {review['rating']}")
#     print(f"Date: {review['date']}")
#     print("---")

# Specify the CSV file name
csv_file = 'reviews.csv'

# Get the keys (header) from the first dictionary in the list
keys = reviews[0].keys()

# Write to the CSV file
with open(csv_file, 'w', newline='\n') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    
    # Write the header
    dict_writer.writeheader()
    
    # Write the data
    dict_writer.writerows(reviews)

print(f"Data exported to {csv_file} successfully!")