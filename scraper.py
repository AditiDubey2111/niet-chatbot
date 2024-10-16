"""
The purpose of this file is to scrape contents from the NIET website to be context-aware while answering questions relating to NIET
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

WEBSITE_URL = "https://www.niet.co.in/blog"
MAX_PAGES = 5


def scrape_page(url):
    try:
        # Fetch the page content
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a successful response

        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find(id="primary")

        if (main_content == None):
            return

        # Extract the page text
        page_text = main_content.get_text(separator="\n").strip()

        return page_text, url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, url


def crawl_website(base_url=WEBSITE_URL, max_pages=5):
    if os.path.exists('scraped_pages'):
        # If sacred_pages already exists, skip scraping
        print("Skipping scraping because already scraped the website. To re-scrape, stop this Space and re-run it to start again (or delete the scraped_pages folder).")
        return

    visited = set()  # To keep track of visited URLs
    to_visit = [base_url]

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        # Scrape the current page
        page_text, page_url = scrape_page(current_url)

        if page_text:
            # Save the content and URL to a file
            save_to_file(page_url, page_text)
            visited.add(current_url)

            # Parse the page and find all internal links
            try:
                response = requests.get(current_url)
                response.raise_for_status()

                # Check if the page encoding is utf-8
                if response.encoding.lower() != 'utf-8':
                    print(f"Skipping {current_url} due to non-utf-8 encoding")
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')
                main_content = soup.find(id="primary")

                if (main_content == None):
                    continue

                for link in main_content.find_all('a', href=True):
                    new_url = urljoin(base_url, link['href'])
                    parsed_url = urlparse(new_url)

                    print(parsed_url.netloc == urlparse(
                        base_url).netloc and new_url not in visited and not parsed_url.fragment)
                    print(urlparse(base_url).netloc)
                    print(parsed_url.netloc)
                    print(new_url not in visited)
                    print(not parsed_url.fragment)

                    # Only consider internal links (ignore external links)
                    # and skip URLs that end with an ID identifier
                    if (parsed_url.netloc == urlparse(base_url).netloc and new_url not in visited and not parsed_url.fragment):
                        print(new_url)
                        to_visit.append(new_url)
                        print(to_visit)
            except Exception as e:
                print(f"Error processing {current_url}: {e}")

    print(f"Visited {len(visited)} pages.")


def save_to_file(url, content):
    # Make sure that the directory exists
    os.makedirs("scraped_pages", exist_ok=True)

    filename = os.path.join("scraped_pages", urlparse(
        url).path.strip("/").replace("/", "_") + ".txt")
    if not filename.endswith(".txt"):
        filename += ".txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"URL: {url}\n\n")
        file.write(content)

    print(f"Saved {url} to {filename}")


if __name__ == "__main__":
    crawl_website(WEBSITE_URL)
