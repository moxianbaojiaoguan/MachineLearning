import pandas as pd
import os
import csv


def merge_csv_to_txt(folder_path, output_file):
    # 确保输出文件的路径存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', newline='') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', newline='') as infile:
                    # 读取CSV文件，跳过表头
                    reader = csv.reader(infile)
                    next(reader)  # 跳过表头
                    # 写入到输出文件，使用\t作为分隔符
                    writer = csv.writer(outfile, delimiter='\t')
                    for row in reader:
                        writer.writerow(row)

#合并.csv为.txt
def merge_csv():
    # 使用示例
    source_folder = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\ResData\_A'  # 替换为你的CSV文件所在的文件夹路径
    output_file = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\ResData\result.txt'  # 替换为你想要保存合并后的TXT文件的路径
    merge_csv_to_txt(source_folder, output_file)


# 合并两个.txt文件的Python脚本

def merge_two_txt(file1, file2, output_file):
    # 打开第一个文件并读取内容
    with open(file1, 'r') as f1:
        content1 = f1.readlines()

    # 打开第二个文件并读取内容
    with open(file2, 'r') as f2:
        content2 = f2.readlines()

    # 打开输出文件并写入两个文件的内容
    with open(output_file, 'w') as output:
        output.writelines(content1)
        output.writelines(content2)

def merge_txt():
    # 指定文件名
    file1 = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\ResData\result1_num200.txt'
    file2 = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\ResData\result2_num200.txt'
    output_file = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\ResData\result200_200.txt'

    # 调用函数合并文件
    merge_two_txt(file1, file2, output_file)
    print(f"{file1} 和 {file2} 已合并到 {output_file}")

# 读取文本文件并处理数据
def process_and_save_data(file_path, chunk_size):
    with open(file_path, 'r') as file:
        chunk = []  # 存储当前chunk的数据
        file_count = 121  # 文件计数器
        #pass_line = [0, 991820]
        for line_count, line in enumerate(file):
            #if line_count in pass_line:
            #    continue
            if line_count % chunk_size == 0:
                if chunk:  # 保存当前chunk到CSV
                    df = pd.DataFrame(chunk, columns=['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time'])
                    df.to_csv(f'Data/Csv/OriData/seg2/part_{file_count}.csv', index=False)
                    print(f'File part_{file_count}.csv has been saved.')
                    file_count += 1
                    chunk = []  # 重置chunk
            chunk.append(line.strip().split(','))  # 假设数据列是用制表符分隔的

        # 保存剩余的数据
        if chunk:
            df = pd.DataFrame(chunk, columns=['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time'])
            df.to_csv(f'Data/Csv/OriData/seg2/part_{file_count}.csv', index=False)
            print(f'File part_{file_count}.csv has been saved.')


# 拆开
def seg():
    txt_file_path = r'D:\DOC\NUAA-new\course\MachineLearning\tianchi\Ali-Mobile-Recommendation-master\Data\Csv\OriData\seg2\part_120.csv'  # 文件路径
    chunk_size = 5000000  # 每次处理的行数

    process_and_save_data(txt_file_path, chunk_size)




if __name__ == '__main__':
    #merge_csv()
    merge_txt()
    #seg()