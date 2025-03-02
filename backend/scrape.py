import os
from jigsawstack import JigsawStack
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("JIGSAWSTACK_API_KEY")
jigsawstack = JigsawStack()

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from jigsawstack import JigsawStack
from tqdm import tqdm
from pprint import pprint
import argparse
import re
import unicodedata
import re

def _sanitize_string(s: str) -> str:
    if not s:
        return ""
    # 1) Normalize to NFD to split off accents/diacritics
    s = unicodedata.normalize('NFD', s)
    # 2) Remove anything outside [\x20-\x7E\n\r], i.e. normal ASCII printable + line breaks
    #    Adjust the range to suit your needs (e.g., keep extended ASCII or certain symbols).
    s = re.sub(r'[^\x20-\x7E\n\r]+', '', s)
    # Trim and return
    return s.strip()


def contains_any_keyword(url: str, keywords: list) -> bool:
    """
    Check if a URL contains any of the specified keywords.
    
    Args:
        url: The URL to check
        keywords: List of keywords to search for in the URL
        
    Returns:
        True if any keyword is found in the URL, False otherwise
    """
    url_lower = url.lower()
    return not any(keyword.lower() in url_lower for keyword in keywords)

class HuggingFaceScraper:
    def __init__(self):
        # Load environment variables
        dotenv_path = Path(__file__).parent.parent / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
        else:
            load_dotenv()  # Try current directory
            
        # Initialize JigsawStack
        api_key = os.getenv("JIGSAWSTACK_API_KEY")
        if not api_key:
            raise ValueError("JIGSAWSTACK_API_KEY not found in environment variables")
        
        self.jigsawstack = JigsawStack(api_key=api_key)
        
        # Base URLs
        self.base_url = "https://huggingface.co"
        
        # Headers to mimic a browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    def search_models(self, keyword: str, max_pages: int = 5) -> List[str]:
        """
        Search for models on Hugging Face based on a keyword
        Returns a list of URLs for model pages
        """
        model_urls = {}
        print(f"Searching for models containing '{keyword}'...")
        
        # Loop through pages
        for page in range(1, max_pages + 1):
            # Build search URL

            
            try:
                search_url = search_url = f"https://huggingface.co/models?search={keyword}&p={page}"
                
                # If we get an HTML response, we need to parse it using JigsawStack
                    # Use jigsawstack to extract model links from the search results page
                scrape_params = {
                    "url": search_url,
                    "element_prompts": ["Extract all model card links"]
                }
                
                result = self.jigsawstack.web.ai_scrape(scrape_params)
                pprint(result)
                
                # Process the AI scraping results
                if result and isinstance(result, dict):
                    # Try to extract model links from the content
                    links = self._extract_model_links(result)
                    model_urls.update(links)
                    
                    # If no links found, we might have reached the end of results
                    if not links:
                        print(f"No more results found after page {page}")
                        break
                
                print(f"Processed page {page}: Found {len(model_urls)} model URLs so far")
                print(f"Unexpected response format from page {page}")
                
                # Be respectful with rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                continue
        
        # Remove duplicates and ensure all URLs are absolute
        
        print(f"Found {len(model_urls)} unique model URLs")
        return model_urls
    
    def _extract_model_links(self, content: dict, limit=5) -> dict:
        """
        Extract model links from scraped content
        """
        links = content['link']
        out = {}
        pattern = r"https://huggingface\.co/([^/]+)/([^/\s]+)"

        for idx, link in enumerate(links):
            if len(out) == limit:
                break
            # If "href" doesn't contain unwanted keywords, 
            # and it matches the huggingface model pattern
            if contains_any_keyword(link['href'], ['blob','search']) and re.match(pattern, link['href']):
                # Sanitize the link's "text" as the "model name"
                safe_name = _sanitize_string(link['text'])
                safe_href = _sanitize_string(link['href'])  # optional if you also want to sanitize URLs
                out[safe_name] = safe_href
        print(out)
        return out
                
    
    def scrape_model_details(self, model_urls: dict) -> List[Dict[str, Any]]:
        model_details = []
        print(f"Scraping details for {len(model_urls)} models...")

        for idx, model_name in enumerate(tqdm(model_urls)):
            url = model_urls[model_name]
            try:
                scrape_params = {
                    "url": url,
                    "element_prompts": [
                        "Extract the number of downloads", 
                        "Extract the number of likes",
                        "Extract the model author/organization",
                        "Extract model description"
                    ]
                }
                result = self.jigsawstack.web.ai_scrape(scrape_params)
                
                if result and isinstance(result, dict):
                    content = result.get('context', '')

                    # Extract fields
                    raw_downloads = self._extract_info(content, "Extract the number of downloads", "0")
                    raw_likes = self._extract_info(content, "Extract the number of likes", "0")
                    raw_author = self._extract_info(content, "Extract the model author/organization", "")
                    raw_description = self._extract_info(content, "Extract model description", "")

                    # Sanitize text fields
                    author = _sanitize_string(raw_author)
                    description = _sanitize_string(raw_description)
                    # We also might want to re-sanitize the model_name from earlier
                    safe_model_name = _sanitize_string(model_name)

                    # Clean numeric values
                    downloads = self._clean_numeric(raw_downloads)
                    likes = self._clean_numeric(raw_likes)

                    # Add to the results list
                    model_details.append({
                        "name": safe_model_name,
                        "url": url,  # or sanitize if you want
                        "author": author,
                        "downloads": downloads,
                        "likes": likes,
                        "description": description
                    })

                time.sleep(1)  # rate-limiting
            except Exception as e:
                print(f"Error scraping model {url}: {str(e)}")
                # fallback
                model_details.append({
                    "url": url,
                    "name": model_name, 
                    "author": "",
                    "downloads": 0,
                    "likes": 0,
                    "description": f"Error: {str(e)}"
                })

        return model_details

    
    def _extract_info(self, content: str, info_type: str, default: str) -> str:
        """
        Extract specific information from AI-scraped content
        """
        print(content)
        return content.get(info_type,default)
        
    
    def _clean_numeric(self, value: str) -> int:
        """
        Clean numeric values (downloads, likes) and convert to integers
        """
        try:
            # Remove non-numeric characters
            clean_value = ''.join(c for c in value if c.isdigit())
            return int(clean_value) if clean_value else 0
        except:
            return 0
    
    def save_results(self, model_details: List[Dict[str, Any]], output_file: str):
        """
        Save the results to a JSON file
        """
        if not model_details:
            print("No results to save")
            return
        
        # Determine file format
        base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
        
        # Save as JSON
        json_file = f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(model_details, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {json_file}")

def main():
    # Set up argument parser
    try:
        scraper = HuggingFaceScraper()
        
        # Search for models
        max_pages = 1
        keyword = "water"
        model_urls = scraper.search_models(keyword, max_pages)
        print("\n\n\n\n\nDEBUG\n\n\n\n\n\n")
        print(model_urls)
        if not model_urls:
            print(f"No models found for keyword '{keyword}'")
            return
        
        # Scrape model details
        model_details = scraper.scrape_model_details(model_urls)
        
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
