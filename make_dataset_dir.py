#!/bin/python3

import os
import shutil
import fnmatch


def move_files(src_dir, dst_dir, file_pattern):
    os.makedirs(f"{dst_dir}/train")
    os.makedirs(f"{dst_dir}/test")

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            _, dir_name = os.path.split(root)
            src_file = os.path.join(root, file)

            if fnmatch.fnmatch(file, file_pattern):
                cur_dst = f"{dst_dir}/test/{dir_name}"
            else:
                cur_dst = f"{dst_dir}/train/{dir_name}"
            if not os.path.exists(cur_dst):
                os.makedirs(cur_dst)
            dst_file = os.path.join(cur_dst, file)

            shutil.move(src_file, dst_file)
            print(f"Moved '{src_file}' to '{dst_file}'")


move_files("Inputs/Caltech_256", "Inputs/Caltech_256", "*0.jpg")
