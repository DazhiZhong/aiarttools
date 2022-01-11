
import os
import sys
from google.colab import drive

abs_root_path = "/content"

def connect_to(folder="dev"):
    global abs_root_path
    drive.mount(abs_root_path+"/drive")
    folder_name = folder 
    if len(folder_name) > 0:
        path_tmp =  abs_root_path + "/drive/MyDrive/" + folder_name
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        abs_root_path = path_tmp
    print("Created folder & set root path to: " + abs_root_path)
    os.chdir(abs_root_path)
