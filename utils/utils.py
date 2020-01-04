import os
from pathlib import Path
def get_image_paths(data_dir="./data"):
    paths_list=[]
    for file in os.listdir(data_dir):
        if Path(file).suffix=='.JPG' and file[0] is  '0':
            paths_list.append(os.path.join(data_dir,file))
            # print(Path(file))
    return paths_list