import os
import shutil
import fnmatch

def move_files(src_dir, dst_dir, file_pattern):
    """
    将符合给定模式的文件从源目录及其子目录移动到目标目录
    
    Args:
        src_dir (str): 源目录路径
        dst_dir (str): 目标目录路径
        file_pattern (str): 文件名模式(如 "*.jpg")
    """
    # 删除目标目录(如果存在)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(f"{dst_dir}/train")
    os.makedirs(f"{dst_dir}/test")
    
    # 遍历源目录及其子目录
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            _, dir_name = os.path.split(root)
            src_file = os.path.join(root, file)
            # 检查文件名是否符合模式
            if fnmatch.fnmatch(file, file_pattern):
                cur_dst = f"{dst_dir}/test/{dir_name}"
            else:
                cur_dst = f"{dst_dir}/train/{dir_name}"
            if not os.path.exists(cur_dst):
                os.makedirs(cur_dst)
            dst_file = os.path.join(cur_dst, file)
                
            # 移动文件
            shutil.move(src_file, dst_file)
            print(f"Moved '{src_file}' to '{dst_file}'")

# 使用示例
move_files("Inputs/Images", "Inputs/CVPR", "*0.jpg")
