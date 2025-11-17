# coding: utf-8

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataloader import TrainDataLoader
import yaml
import re
from utils.dataset import RecDataset
import torch
from interest_matrix_computer import InterestMatrixComputer

import json
import time

def build_yaml_loader():
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return loader


def load_dataset_config(config_dict):
    file_config_dict = dict()
    file_list = []
    # get dataset and model files
    cur_dir = os.getcwd()
    cur_dir = os.path.join(cur_dir, 'configs')
    file_list.append(os.path.join(cur_dir, "overall.yaml"))
    file_list.append(os.path.join(cur_dir, "dataset", "{}.yaml".format(config_dict['dataset'])))
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            file_config_dict.update(yaml.load(f.read(), Loader=build_yaml_loader()))

    return file_config_dict

def get_history_items_per_u(dataset_name):
    config_dict = {'gpu_id': 0, 'dataset': dataset_name}

    config = load_dataset_config(config_dict)
    config.update(config_dict)
    use_gpu = config['use_gpu']
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    config['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    config['use_neighborhood_loss'] = False

    dataset = RecDataset(config)

    print(str(dataset))

    train_dataset, _, _ = dataset.split()

    print(str(train_dataset))

    train_dataset = TrainDataLoader(config, train_dataset)

    return train_dataset.history_items_per_u



data = "microlens"

alpha = [1]
beta = [1]
knn_is = [5,10,15,20]

print("=" * 60)
print("Starting PI tuning experiment")
print("=" * 60)
print(f"Dataset: {data}")
print(f"Alpha parameters: {alpha}")
print(f"Beta parameters: {beta}")
print(f"Top-K parameters: {knn_is}")
# print(f"KNN-I parameters: {knn_is}")


all_results = {}  # 存储所有PI计算结果
time_records = {}  # 添加时间记录字典
total_combinations = len(alpha) * len(beta) * len(knn_is)
current_combination = 0

print(f"Starting parameter tuning, total {total_combinations} parameter combinations")
print("=" * 60)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IM_computer=InterestMatrixComputer(device)
dataset_path = os.path.join('./data', data)
history_items_per_u = get_history_items_per_u(data)

for a in alpha:
    for b in beta:
        file = os.path.join(dataset_path, f'PI_{a}_{b}_25.pt')
        C_masked = torch.load(file)
        for knn_i in knn_is:
            current_combination += 1
            paras = f"{a}_{b}_{knn_i}"
            print(f"Processing combination {current_combination}/{total_combinations}: alpha={a}, beta={b}, knn_i={knn_i}")
            
            # 记录兴趣矩阵计算时间
            start_time = time.time()
            interest_matrix = IM_computer.compute_interest_matrix_with_mask(C_masked, history_items_per_u, knn_i, dataset_path, a, b)
            interest_computation_time = time.time() - start_time
            
            # 保存结果和时间记录
            all_results[paras] = interest_matrix
            time_records[paras] = {
                'interest_computation_time': interest_computation_time
            }
            
            print(f"  Interest matrix computation completed in {interest_computation_time:.2f} seconds")
            print(f"  Combination {current_combination}/{total_combinations} completed")
        

print("=" * 60)
print(f"Total combinations computed: {len(all_results)}")

# 计算总时间统计
total_interest_time = sum(record.get('interest_computation_time', 0) for record in time_records.values())
avg_interest_time = total_interest_time / len(time_records) if time_records else 0

print(f"Total interest matrix computation time: {total_interest_time:.2f} seconds")
print(f"Average interest matrix computation time: {avg_interest_time:.2f} seconds")
print("=" * 60)

print("Saving results...")

# 保存所有兴趣矩阵结果
print("Saving all interest matrix results...")

for paras, interest_matrix in all_results.items():
    interest_file = os.path.join(dataset_path, f'interest_{paras}.pt')
    torch.save(interest_matrix, interest_file)
    print(f"  Interest matrix {paras} saved to: {interest_file}")


#保存时间记录
time_file = os.path.join(dataset_path, f'time_records_IM_{time.time()}.json')
with open(time_file, 'w') as f:
    json.dump(time_records, f, indent=2)
print(f"Time records saved to: {time_file}")

# 显示每个参数组合的时间
print("\nDetailed time analysis:")
print("参数组合 | 兴趣矩阵计算时间")
print("-" * 50)
for params, time_info in time_records.items():
    interest_time = time_info.get('interest_computation_time', 0)
    print(f"{params:15s} | {interest_time:8.2f}秒")

print("=" * 60)
print("Computing interest matrices...")
print("=" * 60)

print("=" * 60)
print("All computations completed!")
print("=" * 60)
print("Experiment summary:")
print(f"  Total combinations computed: {len(all_results)}")
print("=" * 60)
print("Saved files:")
print(f"  Time records: {time_file}")
print("=" * 60)

