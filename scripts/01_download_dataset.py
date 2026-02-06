import sys
import os
sys.path.append(os.getcwd())
from src.config import CONF
from src.dataset.downloader import check_dataset_integrity

if __name__ == "__main__":
    print("Checking dataset status...")
    if check_dataset_integrity(CONF):
        print("Ready for preprocessing.")
    else:
        sys.exit(1)