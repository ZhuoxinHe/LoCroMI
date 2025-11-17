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
import numpy as np
from utils.utils import build_sim
from PI_compute_Optimizer import PotentialInteractionOptimizer

import json
import time
import gc

# 设置CUDA内存管理环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


def get_item_features(config):

    # load parameters info
    device = config['device']

    # load encoded features here
    v_feat, t_feat = None, None
    if not config['end2end'] and config['is_multimodal_model']:
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # if file exist?
        v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        if os.path.isfile(v_feat_file_path):
            v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                device)
        if os.path.isfile(t_feat_file_path):
            t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                device)

        assert v_feat is not None or t_feat is not None, 'Features all NONE'

    return v_feat, t_feat


def load_dataset_matrix(dataset_name):
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

    train_dataset, valid_dataset, _ = dataset.split()

    # print(str(valid_dataset))
    print(str(train_dataset))
    print(str(valid_dataset))
    train_dataset, valid_dataset = (TrainDataLoader(config, train_dataset), TrainDataLoader(config, valid_dataset))

    # 直接使用稀疏矩阵，避免转换为密集矩阵
    train_coo = train_dataset.inter_matrix(form='coo')
    valid_coo = valid_dataset.inter_matrix(form='coo')
    
    # 将scipy稀疏矩阵转换为torch稀疏张量
    train_inter = scipy_coo_to_torch_sparse(train_coo, device=config['device'])
    valid_inter = scipy_coo_to_torch_sparse(valid_coo, device=config['device'])

    # valid_and_test_matrix = (valid_matrix+test_matrix).coalesce()
    # return valid_and_test_matrix
    v_feat, t_feat = get_item_features(config)
    return train_inter, valid_inter, v_feat, t_feat, config['device'], config['data_path'] + config['dataset'], train_dataset.history_items_per_u


def scipy_coo_to_torch_sparse(scipy_coo, dtype=torch.float32, device=None) -> torch.Tensor:
    """
    将SciPy的COO稀疏矩阵转换为PyTorch稀疏张量

    Args:
        scipy_coo (coo_matrix): SciPy的COO格式稀疏矩阵
        dtype (torch.dtype): 输出张量的数据类型，默认float32
        device: 目标设备，如果为None则使用CPU

    Returns:
        torch.Tensor: PyTorch稀疏张量（COO格式）
    """
    # 提取行索引、列索引和非零值
    rows = scipy_coo.row.astype(np.int32)
    cols = scipy_coo.col.astype(np.int32)
    data = scipy_coo.data.astype(np.float32)  # 根据实际数据类型调整

    # 转换为PyTorch张量
    indices = torch.from_numpy(np.vstack((rows, cols))).to(torch.long)
    values = torch.from_numpy(data).to(dtype)

    # 构造稀疏张量
    shape = scipy_coo.shape
    torch_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape,
    )

    # 如果指定了设备，则移动到该设备
    if device is not None:
        torch_sparse = torch_sparse.to(device)

    # 合并重复索引（可选，确保无重复）
    torch_sparse = torch_sparse.coalesce()

    return torch_sparse



data = "microlens"

alpha = [1]
beta = [1]
top_ks = [25]

print("=" * 60)
print("Starting PI computation experiment")
print("=" * 60)
print(f"Dataset: {data}")
print(f"Alpha parameters: {alpha}")
print(f"Beta parameters: {beta}")
print(f"Top-K parameters: {top_ks}")
# print(f"KNN-I parameters: {knn_is}")
print("=" * 60)

print("Loading dataset...")
train_inter, valid_inter, v_feat, t_feat, device, dataset_path, history_items_per_u = load_dataset_matrix(data)
v_feat = v_feat.to(device)
t_feat = t_feat.to(device)
train_inter = train_inter.to(device)
valid_inter = valid_inter.to(device)
is_large_data = train_inter.shape[0]*train_inter.shape[1] > 2000000000

print(f"Dataset loading completed")
print(f"Training interaction matrix shape: {train_inter.shape}")
print(f"Validation interaction matrix shape: {valid_inter.shape}")
print(f"Visual features shape: {v_feat.shape}")
print(f"Text features shape: {t_feat.shape}")
print(f"Device: {device}")
print(f"Dataset path: {dataset_path}")
print(f"Is large dataset: {is_large_data}")
print("=" * 60)
if not is_large_data:
    print("Computing similarity matrices...")
    # 基于文本的相似度
    print("Computing text similarity matrix...")
    S_t = build_sim(t_feat)
    print(f"Text similarity matrix shape: {S_t.shape}")

    # 基于视觉的相似度
    print("Computing visual similarity matrix...")
    S_v = build_sim(v_feat)
    print(f"Visual similarity matrix shape: {S_v.shape}")
    print("=" * 60)

print("Initializing PI optimizer...")
PI_opti = PotentialInteractionOptimizer(device)
print("PI optimizer initialization completed")

print("Computing interaction similarity matrix...")
S_r=torch.sparse.mm(train_inter.T, train_inter)
print(f"Interaction similarity matrix shape: {S_r.shape}")
print("=" * 60)

all_results = {}  # 存储所有PI计算结果
time_records = {}  # 添加时间记录字典
total_combinations = len(alpha) * len(beta) * len(top_ks)
current_combination = 0

print(f"Starting parameter tuning, total {total_combinations} parameter combinations")
print("=" * 60)

def sparse_f1(A, B):
    """
    计算两个稀疏矩阵A和B之间的F1指标
    
    参数:
    A: 预测值的稀疏矩阵 (torch.sparse_coo_tensor)
    B: 真实标记的稀疏矩阵 (torch.sparse_coo_tensor)
    
    返回:
    f1: F1分数 (精确率和召回率的调和平均数)
    """
    # 确保两个矩阵形状相同
    assert A.shape == B.shape, "Matrices must have the same shape"
    
    # 获取A和B的非零位置
    A_indices = A._indices()
    B_indices = B._indices()
    
    # 将索引转换为集合进行比较
    A_positions = set(zip(A_indices[0].tolist(), A_indices[1].tolist()))
    B_positions = set(zip(B_indices[0].tolist(), B_indices[1].tolist()))
    
    # 计算真正例(TP): A和B在同一位置都有非零值
    TP = len(A_positions & B_positions)
    
    # 计算A中非零元素总数(预测为正例的数量)
    pred_positives = len(A_positions)
    
    # 计算B中非零元素总数(实际正例的数量)
    actual_positives = len(B_positions)
    
    # 计算假正例(FP): A有非零值但B为零的位置
    FP = pred_positives - TP
    
    # 计算假负例(FN): B有非零值但A为零的位置
    FN = actual_positives - TP
    
    # 计算精确率 (Precision)
    precision = TP / max(TP + FP, 1e-8)
    
    # 计算召回率 (Recall)
    recall = TP / max(TP + FN, 1e-8)
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
    
    return f1

f1_records = {}
best_result = None
best_f1_score = -1
best_paras_key = None

for a in alpha:
    for b in beta:
        print(f"Processing parameter combination: alpha={a}, beta={b}")
        if is_large_data:
            print(f"  Computing potential interaction matrix (batch processing multiple top_k)...")
            start_time = time.time()
            results = PI_opti.compute_potential_interaction(R=train_inter, S_r=S_r, use_chunking=is_large_data,
                                                           alpha=a, beta=b, top_k=top_ks, text_features=t_feat, visual_features=v_feat)
            pi_computation_time = time.time() - start_time
            print(f"  Potential interaction matrix computation completed in {pi_computation_time:.2f} seconds")
            
            # 处理返回的列表
            for i, k in enumerate(top_ks):
                current_combination += 1
                paras = f"{a}_{b}_{k}"
                result = results[i]
                f1_score = sparse_f1(result, valid_inter)
                f1_records[paras] = f1_score
                
                # 检查是否为最佳结果
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_result = result
                    best_paras_key = paras
                    print(f"  New best F1 score: {f1_score:.6f} for {paras}")
                
                print(f"  Processing combination {current_combination}/{total_combinations}: alpha={a}, beta={b}, k={k}, F1={f1_score:.6f}")
                time_records[paras] = {
                    f'pi_computation_time_by_{len(top_ks)}_times': pi_computation_time,
                }
                print(f"  Combination {current_combination}/{total_combinations} completed")
        else:
            # 修复列表解包语法
            print(f"  Computing potential interaction matrix (batch processing)...")
            start_time = time.time()
            results = PI_opti.compute_potential_interaction(R=train_inter, S_r=S_r, use_chunking=is_large_data,
                                                  alpha=a, beta=b, top_k=top_ks, S_t=S_t, S_v=S_v)
            pi_computation_time = time.time() - start_time
            
            # 更激进的内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                # 额外的内存清理
                torch.cuda.reset_peak_memory_stats()
            print(f"  Potential interaction matrix computation completed in {pi_computation_time:.2f} seconds")
            
            # 处理返回的列表
            for i, k in enumerate(top_ks):
                current_combination += 1
                paras = f"{a}_{b}_{k}"
                result = results[i]
                f1_score = sparse_f1(result, valid_inter)
                f1_records[paras] = f1_score
                
                # 检查是否为最佳结果
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_result = result
                    best_paras_key = paras
                    print(f"  New best F1 score: {f1_score:.6f} for {paras}")
                
                print(f"  Processing combination {current_combination}/{total_combinations}: alpha={a}, beta={b}, k={k}, F1={f1_score:.6f}")
                time_records[paras] = {
                    f'pi_computation_time_by_{len(top_ks)}_times': pi_computation_time,
                }
        print(f"Parameter combination alpha={a}, beta={b} processing completed")
        
        # 每个参数组合处理完后进行内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        
        print("-" * 40)

print("=" * 60)
print("Parameter tuning completed!")
print("=" * 60)
print(f"Total combinations computed: {len(f1_records)}")
print("=" * 60)

print("Saving results...")

print(f"Best F1 score: {best_f1_score:.6f} for parameters: {best_paras_key}")

# 只保存F1分数最高的PI矩阵
print("Saving only the best PI matrix...")
PI_file = os.path.join(dataset_path, f'PI_{best_paras_key}.pt')
torch.save(best_result, PI_file)
print(f"  Best PI matrix {best_paras_key} (F1={best_f1_score:.6f}) saved to: {PI_file}")

#保存所有F1结果到JSON文件
print("Saving all F1 results to JSON...")
f1_json_file = os.path.join(dataset_path, f'f1_scores_{time.time()}.json')
with open(f1_json_file, 'w') as f:
    json.dump(f1_records, f, indent=2)
print(f"  F1 scores saved to: {f1_json_file}")

# 打印所有F1分数供参考
print("\nAll F1 scores for reference:")
for paras, f1_score in sorted(f1_records.items(), key=lambda x: x[1], reverse=True):
    print(f"  {paras}: {f1_score:.6f}")

#保存时间记录
time_file = os.path.join(dataset_path, f'time_records_PI_{time.time()}.json')
with open(time_file, 'w') as f:
    json.dump(time_records, f, indent=2)
print(f"Time records saved to: {time_file}")

print("=" * 60)
print("Computing interest matrices...")
print("=" * 60)

# #计算兴趣矩阵
# interest_time_records = {}  # 添加兴趣矩阵时间记录
# if is_large_data:
#     print("Large dataset mode: Computing interest matrices using original features")
#     for i, knn_i in enumerate(knn_is):
#         print(f"Computing interest matrix {i+1}/{len(knn_is)}: knn_i={knn_i}")
#         start_time = time.time()
#         PI_opti.compute_potential_interaction(R=train_inter, S_r=S_r, use_chunking=is_large_data,
#                                         alpha=best_paras[0], beta=best_paras[1], top_k=best_paras[2], text_features=t_feat, visual_features=v_feat,
#                                         only_compute_interest = True, history_items_per_u=history_items_per_u, knn_i=knn_i, dataset_path=dataset_path)
#         interest_time = time.time() - start_time
#         interest_time_records[f"knn_i_{knn_i}"] = interest_time
#         print(f"Interest matrix knn_i={knn_i} computation completed in {interest_time:.2f} seconds")
# else:
#     print("Small dataset mode: Computing interest matrices using precomputed similarity matrices")
#     for i, knn_i in enumerate(knn_is):
#         print(f"Computing interest matrix {i+1}/{len(knn_is)}: knn_i={knn_i}")
#         start_time = time.time()
#         PI_opti.compute_potential_interaction(R=train_inter, S_r=S_r, use_chunking=is_large_data,
#                                                   alpha=best_paras[0], beta=best_paras[1], top_k=best_paras[2], S_t=S_t, S_v=S_v,
#                                                   only_compute_interest = True, history_items_per_u=history_items_per_u, knn_i=knn_i, dataset_path=dataset_path)
#         interest_time = time.time() - start_time
#         interest_time_records[f"knn_i_{knn_i}"] = interest_time
#         print(f"Interest matrix knn_i={knn_i} computation completed in {interest_time:.2f} seconds")

# #保存兴趣矩阵时间记录
# interest_time_file = os.path.join(dataset_path, 'interest_time_records.json')
# with open(interest_time_file, 'w') as f:
#     json.dump(interest_time_records, f, indent=2)
# print(f"Interest matrix time records saved to: {interest_time_file}")

print("=" * 60)
print("All computations completed!")
print("=" * 60)
print("Experiment summary:")
print(f"  Total combinations computed: {len(all_results)}")
print("=" * 60)
print("Saved files:")
print(f"  Time records: {time_file}")
print("=" * 60)

