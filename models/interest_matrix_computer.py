import torch
import torch.nn.functional as F
import os
import gc


class InterestMatrixComputer:
    def __init__(self, device):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 设置PyTorch内存分配策略以减少碎片化
        if torch.cuda.is_available():
            # 清理缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _get_memory_info(self):
        """获取内存使用信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    
    def _aggressive_memory_cleanup(self):
        """激进的内存清理"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 强制垃圾回收
            torch.cuda.ipc_collect()
    
    def build_non_zero_graph_sparse(self, sparse_matrix, norm_type='sym'):
        """
        在稀疏矩阵上构建非零图并进行归一化，避免使用密集矩阵
        
        参数:
            sparse_matrix: 输入稀疏矩阵
            norm_type: 归一化类型 ('sym' 对称归一化, 'row' 行归一化)
            
        返回:
            归一化后的稀疏矩阵
        """
        # 确保输入是稀疏矩阵
        if not sparse_matrix.is_sparse:
            sparse_matrix = sparse_matrix.to_sparse()
        
        # 确保稀疏矩阵已合并
        sparse_matrix = sparse_matrix.coalesce()
        
        # 计算度矩阵（行和）
        row_indices = sparse_matrix.indices()[0]
        values = sparse_matrix.values()
        n_rows = sparse_matrix.shape[0]
        
        # 计算每行的和
        row_sums = torch.zeros(n_rows, device=self.device, dtype=values.dtype)
        row_sums.scatter_add_(0, row_indices, values)
        
        # 避免除以零
        row_sums[row_sums == 0] = 1
        
        if norm_type == 'sym':
            # 对称归一化: D^(-1/2) * A * D^(-1/2)
            degree_sqrt = torch.sqrt(row_sums)
            degree_inv_sqrt = 1.0 / degree_sqrt
            
            # 获取稀疏矩阵的索引和值
            indices = sparse_matrix.indices()
            values = sparse_matrix.values()
            
            # 应用对称归一化
            row_deg = degree_inv_sqrt[indices[0]]
            col_deg = degree_inv_sqrt[indices[1]]
            normalized_values = values * row_deg * col_deg
            
            # 清理中间变量
            del degree_sqrt, degree_inv_sqrt, row_deg, col_deg
            self._aggressive_memory_cleanup()
            
            # 创建归一化后的稀疏矩阵
            normalized_sparse = torch.sparse_coo_tensor(
                indices, normalized_values, sparse_matrix.shape, device=self.device
            )
        else:
            pass
        
        return normalized_sparse
    
    def compute_interest_matrix_with_mask(self, C_masked, history_items_per_u, knn_i, dataset_path, alpha, beta):
        """
        使用掩码后的C矩阵计算兴趣矩阵，避免for循环
        
        参数:
            C_masked: 已经应用历史交互掩码的C矩阵
            history_items_per_u: 每个用户的历史交互物品字典
            knn_i: 最近邻数量
            dataset_path: 数据集路径
            alpha: 文本相似度权重
            beta: 视觉相似度权重
            
        返回:
            兴趣矩阵
        """
        print(f"Computing interest matrix with masked C matrix, knn_i={knn_i}")
        C_masked = C_masked.coalesce()
        n_users, n_items = C_masked.shape
        
        # 初始化稀疏矩阵的索引和值列表
        rows = []
        cols = []
        values = []
        
        # 直接处理所有用户
        for user in range(n_users):
            # 输出处理进度
            if user % 1000 == 0 or user == n_users - 1:
                progress = (user + 1) / n_users * 100
                print(f"处理用户进度: {user + 1}/{n_users} ({progress:.1f}%)")
            
            items = history_items_per_u[user]
            items_tensor = torch.tensor(list(items), device=self.device)
            
            # 获取当前用户的top-k物品
            if C_masked.is_sparse:
                # 对于稀疏矩阵，获取当前用户的非零元素
                user_mask = C_masked.indices()[0] == user
                if user_mask.any():
                    user_cols = C_masked.indices()[1][user_mask]
                    user_vals = C_masked.values()[user_mask]
                    # 获取top-k物品
                    _, topk_idx = torch.topk(user_vals, min(knn_i, len(user_vals)))
                    topk_cols = user_cols[topk_idx]
                else:
                    # 如果用户没有非零元素，跳过
                    continue
            else:
                # 对于密集矩阵，直接获取top-k
                _, topk_cols = torch.topk(C_masked[user, :], min(knn_i, n_items))
            
            # 添加稀疏矩阵元素
            rows.extend(items_tensor.repeat_interleave(len(topk_cols)).tolist())
            cols.extend(topk_cols.repeat(len(items_tensor)).tolist())
            values.extend([1.0] * (len(items_tensor) * len(topk_cols)))
        
        # 构建稀疏矩阵
        if rows and cols and values:
            # 转换为张量
            rows_tensor = torch.tensor(rows, device=self.device)
            cols_tensor = torch.tensor(cols, device=self.device)
            values_tensor = torch.tensor(values, device=self.device)
            
            # 创建稀疏矩阵
            interest_matrix = torch.sparse_coo_tensor(
                torch.stack([rows_tensor, cols_tensor]), 
                values_tensor, 
                (n_items, n_items), 
                device=self.device
            )
            
            # 合并重复元素（如果有）
            interest_matrix = interest_matrix.coalesce()
        
        # 构建非零图并进行归一化
        interest_matrix = self.build_non_zero_graph_sparse(interest_matrix)
        
        # 保存兴趣矩阵
        interest_file = os.path.join(dataset_path, f'interest_{alpha}_{beta}_{knn_i}.pt')
        torch.save(interest_matrix, interest_file)
        print(f"Interest matrix saved to {interest_file}")
        
        return interest_matrix
