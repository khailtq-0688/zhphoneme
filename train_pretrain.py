#!/usr/bin/env python3
"""
Simplified training script using ViWordFormer-based architecture
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from pretrain import main

if __name__ == "__main__":
    main()
