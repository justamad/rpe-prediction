from tqdm import tqdm
from pathlib import Path

import os

prog = "E:\\Development\\Azure-Kinect-Samples-master\\body-tracking-samples\\Azure-Kinect-Extractor\\build\\bin\\Debug\\offline_processor.exe"
src_path = "G:\\Azure_Kinect_RPE"

files = list(Path(src_path).rglob('*.mkv'))
for f in tqdm(files):
    print(f)
    os.system(f"{prog} {f} --skeleton")
