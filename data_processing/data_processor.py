"""
Data Processing Scripts for Multilingual Pretraining
Handles Vietnamese (Curated) and Chinese (CommonCrawl) datasets
"""

import os
import logging
from typing import List, Generator, Dict
from pathlib import Path
import json
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Base class for data processing"""
    
    def __init__(self, input_dir: str, output_dir: str, language: str):
        """
        Initialize data processor
        
        Args:
            input_dir: Path to input dataset
            output_dir: Path to output processed data
            language: Language code (vi, zh)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.language = language
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def process(self) -> None:
        """Process dataset"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()


class VietnameseProcessor(DataProcessor):
    """Process Vietnamese Curated Dataset"""
    
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir, 'vi')
        
    def process(self) -> None:
        """
        Process Vietnamese Curated dataset
        Expected structure:
        - input_dir/
          - *.txt (text files)
          - *.jsonl (jsonl files)
        """
        logger.info("Processing Vietnamese Curated dataset...")
        
        output_file = self.output_dir / "vietnamese_processed.txt"
        line_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # Process .txt files
            for txt_file in self.input_dir.glob("*.txt"):
                logger.info(f"Processing {txt_file}")
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            cleaned = self._clean_text(line)
                            if cleaned:
                                f_out.write(cleaned + '\n')
                                line_count += 1
            
            # Process .jsonl files (common for curated datasets)
            for jsonl_file in self.input_dir.glob("*.jsonl"):
                logger.info(f"Processing {jsonl_file}")
                with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                    for line in f_in:
                        try:
                            item = json.loads(line)
                            # Try common field names
                            text = None
                            for field in ['text', 'content', 'body', 'document']:
                                if field in item:
                                    text = item[field]
                                    break
                            
                            if text and isinstance(text, str):
                                cleaned = self._clean_text(text)
                                if cleaned:
                                    f_out.write(cleaned + '\n')
                                    line_count += 1
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Vietnamese processing complete. Processed {line_count} lines")
        logger.info(f"Output saved to {output_file}")
        
        # Create statistics
        self._save_stats(output_file, line_count)
    
    def _save_stats(self, output_file: Path, line_count: int) -> None:
        """Save processing statistics"""
        stats = {
            'language': self.language,
            'total_lines': line_count,
            'output_file': str(output_file),
        }
        stats_file = self.output_dir / f"stats_{self.language}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)


class ChineseProcessor(DataProcessor):
    """Process Chinese CommonCrawl Dataset"""
    
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__(input_dir, output_dir, 'zh')
        
    def process(self) -> None:
        """
        Process Chinese CommonCrawl dataset
        Expected structure:
        - input_dir/
          - *.txt (raw text from CommonCrawl)
          - *.warc (WARC format files)
          - *.jsonl (jsonl format from extraction)
        """
        logger.info("Processing Chinese CommonCrawl dataset...")
        
        output_file = self.output_dir / "chinese_processed.txt"
        line_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # Process .txt files
            for txt_file in self.input_dir.glob("*.txt"):
                logger.info(f"Processing {txt_file}")
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            cleaned = self._clean_text(line)
                            if cleaned and len(cleaned) > 5:  # Filter short lines
                                f_out.write(cleaned + '\n')
                                line_count += 1
            
            # Process .jsonl files (CommonCrawl often provides JSONL extracts)
            for jsonl_file in self.input_dir.glob("*.jsonl"):
                logger.info(f"Processing {jsonl_file}")
                with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                    for line in f_in:
                        try:
                            item = json.loads(line)
                            # Common fields in CommonCrawl extracts
                            text = None
                            for field in ['text', 'content', 'body', 'html', 'raw_content']:
                                if field in item:
                                    text = item[field]
                                    break
                            
                            if text and isinstance(text, str):
                                # For HTML content, basic cleaning
                                text = self._clean_html(text)
                                cleaned = self._clean_text(text)
                                if cleaned and len(cleaned) > 5:
                                    f_out.write(cleaned + '\n')
                                    line_count += 1
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Chinese processing complete. Processed {line_count} lines")
        logger.info(f"Output saved to {output_file}")
        
        # Create statistics
        self._save_stats(output_file, line_count)
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags"""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        try:
            from html import unescape
            text = unescape(text)
        except:
            pass
        return text
    
    def _save_stats(self, output_file: Path, line_count: int) -> None:
        """Save processing statistics"""
        stats = {
            'language': self.language,
            'total_lines': line_count,
            'output_file': str(output_file),
        }
        stats_file = self.output_dir / f"stats_{self.language}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)


class DataMerger:
    """Merge processed datasets"""
    
    def __init__(self, output_dir: str):
        """
        Initialize data merger
        
        Args:
            output_dir: Path to processed data directory
        """
        self.output_dir = Path(output_dir)
        
    def merge_all(self, output_name: str = "merged_corpus.txt") -> None:
        """Merge all processed datasets"""
        logger.info("Merging all processed datasets...")
        
        output_file = self.output_dir / output_name
        
        total_lines = 0
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # Merge Vietnamese
            vi_file = self.output_dir / "vietnamese_processed.txt"
            if vi_file.exists():
                logger.info(f"Merging {vi_file}")
                with open(vi_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        f_out.write(line)
                        total_lines += 1
            
            # Merge Chinese
            zh_file = self.output_dir / "chinese_processed.txt"
            if zh_file.exists():
                logger.info(f"Merging {zh_file}")
                with open(zh_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        f_out.write(line)
                        total_lines += 1
        
        logger.info(f"Merged {total_lines} lines to {output_file}")


if __name__ == "__main__":
    # Example usage
    
    # Vietnamese processing
    vi_processor = VietnameseProcessor(
        input_dir="/path/to/vietnamese/curated/dataset",
        output_dir="./processed_data"
    )
    # vi_processor.process()
    
    # Chinese processing
    zh_processor = ChineseProcessor(
        input_dir="/path/to/chinese/commoncrawl/dataset",
        output_dir="./processed_data"
    )
    # zh_processor.process()
    
    # Merge all
    merger = DataMerger(output_dir="./processed_data")
    # merger.merge_all()
    
    print("Data processing setup complete. Configure paths and run processors.")
