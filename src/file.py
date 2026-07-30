from tqdm import tqdm
import time


files = ["a.py", "b.py", "c.md"]

for f in tqdm(files, desc="Chunking", unit="file"):
    time.sleep(1)


for i in tqdm(range(100)):
    time.sleep(0.05)
