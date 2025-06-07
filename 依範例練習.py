import tensorflow as tf
from glob import glob

root_path = "C:/Users/steve/python_data/"

nezuko_data_path = f"{root_path}AI/find_nzuko/find_nzuko/nezuko/**.png"
non_nezuko_data_path = f"{root_path}AI/find_nzuko/find_nzuko/not-nezuko"

nezuko_datas = glob(nezuko_data_path)
for nezuko_data in nezuko_datas:
    print(nezuko_data)