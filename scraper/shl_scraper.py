"""
SHL Assessment Catalog Scraper

Scrapes individual test solutions from SHL product catalog.
Excludes "Pre-packaged Job Solutions" category.
Target: 377+ individual assessments
"""

import json
import time
import logging
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SHLScraper:
    """
    Scraper for SHL assessment catalog
    """
    
    BASE_URL = "https://www.shl.com"
    CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
    
    def __init__(self, headless: bool = True, delay: float = 1.0):
        """
        Initialize scraper
        
        Args:
            headless: Run browser in headless mode
            delay: Delay between requests in seconds
        """
        self.headless = headless
        self.delay = delay
        self.assessments = []
        
    def extract_assessment_urls(self, page: Page) -> List[str]:
        """
        Extract all individual assessment URLs from catalog page
        
        Args:
            page: Playwright page object
            
        Returns:
            List of assessment URLs
        """
        logger.info("Extracting assessment URLs from catalog...")
        
        # Wait for content to load
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        
        # Get page content
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all assessment links
        # This selector needs to be adjusted based on actual page structure
        assessment_links = []
        
        # Strategy 1: Look for links in product catalog
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Filter for product catalog URLs
            if '/product-catalog/view/' in href:
                # Exclude pre-packaged solutions
                if 'solution' not in href.lower() or 'individual' in href.lower():
                    full_url = urljoin(self.BASE_URL, href)
                    if full_url not in assessment_links:
                        assessment_links.append(full_url)
        
        logger.info(f"Found {len(assessment_links)} potential assessment URLs")
        return assessment_links
    
    def scrape_assessment_detail(self, url: str) -> Optional[Dict]:
        """
        Scrape details from individual assessment page
        
        Args:
            url: Assessment detail page URL
            
        Returns:
            Dictionary with assessment details or None if failed
        """
        try:
            # Use requests for detail pages (faster than Playwright)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract assessment name (adjust selector as needed)
            name_elem = soup.find('h1') or soup.find('title')
            name = name_elem.get_text(strip=True) if name_elem else url.split('/')[-1]
            
            # Extract description
            description = ""
            desc_elem = soup.find('meta', {'name': 'description'})
            if desc_elem and desc_elem.get('content'):
                description = desc_elem['content']
            else:
                # Try to find description in page content
                desc_div = soup.find('div', class_=['description', 'content', 'overview'])
                if desc_div:
                    description = desc_div.get_text(strip=True)[:500]
            
            # Extract test type (K=Knowledge/Skills, P=Personality/Behavior)
            test_type = self._infer_test_type(name, description)
            
            # Extract categories/tags
            categories = self._extract_categories(soup, name, description)
            
            # Get full content for embedding
            full_content = soup.get_text(separator=' ', strip=True)[:2000]
            
            assessment = {
                'name': name,
                'url': url,
                'test_type': test_type,
                'description': description,
                'categories': categories,
                'full_content': full_content
            }
            
            logger.debug(f"Scraped: {name}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def _infer_test_type(self, name: str, description: str) -> str:
        """
        Infer test type from name and description
        
        Returns:
            'K' for Knowledge/Skills, 'P' for Personality/Behavior, 'O' for Other
        """
        text = (name + " " + description).lower()
        
        # Personality/Behavioral indicators
        personality_keywords = [
            'personality', 'opq', 'behavioral', 'behaviour', 'motivation',
            'leadership', 'culture', 'fit', 'values', 'traits'
        ]
        
        # Knowledge/Skills indicators
        knowledge_keywords = [
            'programming', 'coding', 'technical', 'java', 'python', 'sql',
            'test', 'assessment', 'skills', 'ability', 'aptitude', 'verify',
            'numerical', 'verbal', 'reasoning', 'comprehension'
        ]
        
        if any(kw in text for kw in personality_keywords):
            return 'P'
        elif any(kw in text for kw in knowledge_keywords):
            return 'K'
        else:
            return 'O'
    
    def _extract_categories(self, soup: BeautifulSoup, name: str, description: str) -> List[str]:
        """
        Extract categories/tags from page
        """
        categories = []
        
        # Look for category/tag elements
        for tag_elem in soup.find_all(['span', 'div', 'a'], class_=['tag', 'category', 'label']):
            tag_text = tag_elem.get_text(strip=True)
            if tag_text and len(tag_text) < 50:
                categories.append(tag_text)
        
        # Extract from name and description
        text = (name + " " + description).lower()
        
        # Common categories
        category_keywords = {
            'Programming': ['java', 'python', 'javascript', 'programming', 'coding'],
            'Sales': ['sales', 'selling', 'revenue'],
            'Leadership': ['leadership', 'manager', 'executive', 'coo', 'ceo'],
            'Communication': ['communication', 'english', 'verbal', 'writing'],
            'Testing': ['qa', 'testing', 'quality assurance', 'selenium'],
            'Data': ['data', 'analytics', 'sql', 'database'],
            'Administrative': ['admin', 'administrative', 'clerical'],
            'Marketing': ['marketing', 'brand', 'advertising']
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in text for kw in keywords):
                if category not in categories:
                    categories.append(category)
        
        return categories[:5]  # Limit to top 5 categories
    
    def scrape_all(self, output_path: str = 'data/assessments.json') -> List[Dict]:
        """
        Main scraping workflow
        
        Args:
            output_path: Path to save scraped data
            
        Returns:
            List of assessment dictionaries
        """
        logger.info("Starting SHL catalog scraping...")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            
            # Navigate to catalog
            logger.info(f"Navigating to {self.CATALOG_URL}")
            page.goto(self.CATALOG_URL, wait_until="networkidle")
            
            # Extract assessment URLs
            assessment_urls = self.extract_assessment_urls(page)
            
            browser.close()
        
        # Scrape each assessment
        logger.info(f"Scraping {len(assessment_urls)} assessments...")
        
        for url in tqdm(assessment_urls, desc="Scraping assessments"):
            assessment = self.scrape_assessment_detail(url)
            
            if assessment:
                self.assessments.append(assessment)
            
            # Rate limiting
            time.sleep(self.delay)
            
            # Save intermediate results every 50 assessments
            if len(self.assessments) % 50 == 0:
                self._save_intermediate(output_path)
        
        # Final save
        self._save_assessments(output_path)
        
        logger.info(f"Scraping complete! Total assessments: {len(self.assessments)}")
        
        # Validation
        if len(self.assessments) < 377:
            logger.warning(f"Expected 377+ assessments, got {len(self.assessments)}")
        else:
            logger.info(f"✓ Successfully scraped {len(self.assessments)} assessments")
        
        return self.assessments
    
    def _save_intermediate(self, output_path: str):
        """Save intermediate results"""
        temp_path = output_path.replace('.json', '_temp.json')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved intermediate results: {len(self.assessments)} assessments")
    
    def _save_assessments(self, output_path: str):
        """Save final results"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.assessments, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved assessments to {output_path}")


def main():
    """Main entry point for scraper"""
    scraper = SHLScraper(headless=True, delay=1.0)
    assessments = scraper.scrape_all('data/assessments.json')
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SCRAPING SUMMARY")
    print(f"{'='*60}")
    print(f"Total assessments scraped: {len(assessments)}")
    print(f"Test types:")
    
    from collections import Counter
    type_counts = Counter(a['test_type'] for a in assessments)
    for test_type, count in type_counts.items():
        print(f"  {test_type}: {count}")
    
    print(f"\nTop categories:")
    all_categories = [cat for a in assessments for cat in a.get('categories', [])]
    cat_counts = Counter(all_categories)
    for category, count in cat_counts.most_common(10):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
