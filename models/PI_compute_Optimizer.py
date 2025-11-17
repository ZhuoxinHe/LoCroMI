import torch
import torch.nn.functional as F
import time
import os
import gc
from utils.utils import build_non_zero_graph

class PotentialInteractionOptimizer:
    def __init__(self, device):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 设置PyTorch内存分配策略以减少碎片化
        if torch.cuda.is_available():
            # 设置最大分割大小以减少碎片化
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
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
    
    def _sparsify_matrix(self, matrix, keep_ratio=0.1):
        """
        将密集矩阵稀疏化，只保留指定比例的最大值
        
        参数:
            matrix: 输入密集矩阵
            keep_ratio: 保留比例，默认0.1（10%）
            
        返回:
            稀疏矩阵
        """
        if matrix.is_sparse:
            return matrix
        
        # 获取矩阵形状
        n_rows, n_cols = matrix.shape
        
        # 计算要保留的元素数量
        total_elements = n_rows * n_cols
        keep_elements = int(total_elements * keep_ratio)
        
        # 获取所有元素的值和索引
        matrix_flat = matrix.flatten()
        values, indices = torch.topk(matrix_flat, keep_elements)
        
        # 清理临时张量
        del matrix_flat
        self._aggressive_memory_cleanup()
        
        # 将一维索引转换为二维索引
        row_indices = indices // n_cols
        col_indices = indices % n_cols
        
        # 创建稀疏矩阵
        sparse_matrix = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), 
            values, 
            (n_rows, n_cols), 
            device=self.device
        )
        
        return sparse_matrix.coalesce()
    
        
    def compute_similarity_chunk(self, features, chunk_idx, chunk_size, n_items, features_norm=None):
        """计算单个块的相似度（优化版本）"""
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, n_items)
        
        # 如果提供了预标准化的特征，直接使用
        if features_norm is not None:
            features_norm_chunk = features_norm[start:end]
            # 计算相似度矩阵块
            similarity_chunk = torch.mm(features_norm, features_norm_chunk.t())
            # 清理中间变量
            del features_norm_chunk
            self._aggressive_memory_cleanup()
        else:
            # 如果没有预计算特征，使用原始特征
            if features is None:
                raise ValueError("必须提供features或features_norm")
            # 提取当前块的特征
            features_chunk = features[start:end]
            # 标准化特征
            features_norm = F.normalize(features, p=2, dim=1)
            features_norm_chunk = F.normalize(features_chunk, p=2, dim=1)
            # 计算相似度矩阵块
            similarity_chunk = torch.mm(features_norm, features_norm_chunk.t())
            # 清理中间变量
            del features_norm_chunk
            self._aggressive_memory_cleanup()
        
        return similarity_chunk, start, end
    
    def precompute_normalized_features(self, text_features, visual_features):
        """预计算标准化特征，避免重复计算"""
        print("Precomputing normalized features...")
        text_norm = F.normalize(text_features, p=2, dim=1)
        visual_norm = F.normalize(visual_features, p=2, dim=1)
        
        # 清理原始特征以节省内存
        del text_features, visual_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return text_norm, visual_norm
    

    def topk_sparse_with_mask(self, matrix, top_k, mask):
        """
        对矩阵进行top-k稀疏化，同时应用掩码（历史交互置零）
        
        参数:
            matrix: 输入矩阵
            top_k: top-k值
            mask: 掩码矩阵，True表示需要置零的位置
            
        返回:
            稀疏矩阵
        """
            
        # 应用掩码，将历史交互位置置为负无穷（这样它们不会被选为top-k）
        # 避免clone()，直接修改原矩阵的副本
        if matrix.is_sparse:
            # 对于稀疏矩阵，直接在稀疏格式上应用掩码
            # 获取稀疏矩阵的索引和值
            indices = matrix.indices()
            values = matrix.values()
            
            # 应用掩码：将掩码位置的值设为负无穷
            masked_values = values.clone()
            masked_indices = indices.clone()
            
            # 找到需要掩码的位置
            mask_positions = mask[masked_indices[0], masked_indices[1]]
            masked_values[mask_positions] = -float('inf')
            
            # 创建掩码后的稀疏矩阵
            matrix_masked = torch.sparse_coo_tensor(
                masked_indices, masked_values, matrix.shape, device=self.device
            )
        else:
            matrix_masked = matrix.clone()
            matrix_masked[mask] = -float('inf')
        
        # 获取矩阵形状
        n_rows, n_cols = matrix_masked.shape
        
        # 确保top_k是整数
        top_k = int(top_k)
        
        # 检查矩阵大小，如果太大则分块处理
        if matrix_masked.is_sparse and matrix_masked._values().numel() > 10000000:  # 1000万非零元素
            print("WARNING: Sparse matrix too large, using chunked top-k processing")
            return self._topk_sparse_chunked(matrix_masked, top_k, mask)
        elif not matrix_masked.is_sparse and matrix_masked.numel() > 500000000:  # 5亿元素
            print("WARNING: Dense matrix too large, using chunked top-k processing")
            return self._topk_sparse_chunked(matrix_masked, top_k, mask)
        
        # 对每一行进行top-k操作
        if matrix_masked.is_sparse:
            # 对于稀疏矩阵，先转换为密集矩阵，然后进行top-k处理
            matrix_dense = matrix_masked.to_dense()
            sparse_matrix = self._sparse_topk_by_row(matrix_dense, top_k)
            # 清理密集矩阵
            del matrix_dense
            self._aggressive_memory_cleanup()
        else:
            # 对于密集矩阵，直接进行top-k操作
            values, indices = torch.topk(matrix_masked, k=min(top_k, n_cols), dim=1)
            
            # 构建稀疏矩阵
            rows = torch.arange(n_rows, device=self.device).repeat_interleave(min(top_k, n_cols))
            cols = indices.flatten()
            values = values.flatten()
            
            # 创建稀疏矩阵
            sparse_matrix = torch.sparse_coo_tensor(
                torch.stack([rows, cols]), values, (n_rows, n_cols), device=self.device
            )
        
        # 确保稀疏矩阵已合并
        return sparse_matrix.coalesce()
    
    def _sparse_topk_by_row(self, matrix_dense, top_k):
        """
        在密集矩阵上按行进行top-k操作
        
        参数:
            matrix_dense: 输入密集矩阵
            top_k: top-k值
            
        返回:
            稀疏矩阵
        """
        n_rows, n_cols = matrix_dense.shape
        
        # 确保top_k是整数
        top_k = int(top_k)
        
        # 向量化top-k操作
        topk_values, topk_indices = torch.topk(matrix_dense, k=min(top_k, n_cols), dim=1)
        
        # 只保留非负无穷的值
        valid_mask = topk_values > -float('inf')
        
        # 获取有效值的行和列索引
        # 生成行索引：为每个有效位置分配正确的行号
        row_indices = torch.arange(n_rows, device=self.device).unsqueeze(1).expand(-1, min(top_k, n_cols))
        valid_rows = row_indices[valid_mask]
        valid_values = topk_values[valid_mask]
        valid_indices = topk_indices[valid_mask]
        
        # 创建结果稀疏矩阵
        if len(valid_values) > 0:
            sparse_result = torch.sparse_coo_tensor(
                torch.stack([valid_rows, valid_indices]), 
                valid_values, 
                (n_rows, n_cols), 
                device=self.device
            )
        else:
            # 如果没有任何元素，创建空稀疏矩阵
            sparse_result = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
                (n_rows, n_cols),
                device=self.device
            )
        
        return sparse_result.coalesce()
    
    def _sparsify_sparse_matrix_by_row(self, matrix_dense, keep_ratio=0.02):
        """
        在密集矩阵上按行进行稀疏化
        
        参数:
            matrix_dense: 输入密集矩阵
            keep_ratio: 每行保留比例，默认0.02（2%）
            
        返回:
            稀疏矩阵
        """
        n_rows, n_cols = matrix_dense.shape
        
        # 计算每行要保留的元素数量
        keep_elements_per_row = max(1, int(n_cols * keep_ratio))
        
        # 向量化稀疏化操作
        topk_values, topk_indices = torch.topk(matrix_dense, k=keep_elements_per_row, dim=1)
        
        # 只保留非负无穷的值
        valid_mask = topk_values > -float('inf')
        
        # 获取有效值的行和列索引
        # 生成行索引：为每个有效位置分配正确的行号
        row_indices = torch.arange(n_rows, device=self.device).unsqueeze(1).expand(-1, keep_elements_per_row)
        valid_rows = row_indices[valid_mask]
        valid_values = topk_values[valid_mask]
        valid_indices = topk_indices[valid_mask]
        
        all_rows = valid_rows.tolist()
        all_cols = valid_indices.tolist()
        all_values = valid_values.tolist()
        
        # 创建结果稀疏矩阵
        if all_values:
            result_rows = torch.tensor(all_rows, device=self.device)
            result_cols = torch.tensor(all_cols, device=self.device)
            result_values = torch.tensor(all_values, device=self.device)
            
            sparse_result = torch.sparse_coo_tensor(
                torch.stack([result_rows, result_cols]), 
                result_values, 
                (n_rows, n_cols), 
                device=self.device
            )
        else:
            # 如果没有任何元素，创建空稀疏矩阵
            sparse_result = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
                (n_rows, n_cols),
                device=self.device
            )
        
        return sparse_result.coalesce()
    
    def _topk_sparse_chunked(self, matrix_masked, top_k, mask, chunk_size=10000):
        """
        分块处理top-k稀疏化，避免内存溢出（纯稀疏矩阵处理）
        """
        print(f"Processing top-k in chunks, chunk size: {chunk_size}")
        
        n_rows, n_cols = matrix_masked.shape
        all_rows = []
        all_cols = []
        all_values = []
        
        if matrix_masked.is_sparse:
            # 对于稀疏矩阵，按行分块处理
            # 确保稀疏矩阵已合并
            matrix_masked = matrix_masked.coalesce()
            indices = matrix_masked.indices()
            values = matrix_masked.values()
            
            # 分块处理行
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                print(f"  Processing row chunk {start}-{end}")
                
                # 筛选出属于当前行块的行
                row_mask = (indices[0] >= start) & (indices[0] < end)
                chunk_indices = indices[:, row_mask]
                chunk_values = values[row_mask]
                
                if len(chunk_values) == 0:
                    continue
                
                # 调整行索引
                chunk_indices[0] = chunk_indices[0] - start
                
                # 创建当前块的稀疏矩阵
                chunk_sparse = torch.sparse_coo_tensor(
                    chunk_indices, chunk_values, (end - start, n_cols), device=self.device
                ).coalesce()
                
                # 对当前块进行top-k处理
                chunk_dense = chunk_sparse.to_dense()
                
                # 应用掩码（如果提供）
                if mask is not None:
                    mask_chunk = mask[start:end, :]
                    chunk_dense[mask_chunk] = -float('inf')
                
                # 立即清理稀疏矩阵
                del chunk_sparse
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 向量化处理所有行的top-k
                if chunk_dense.numel() > 0:
                    # 确保top_k是整数
                    top_k = int(top_k)
                    # 对每一行进行top-k（向量化操作）
                    topk_values, topk_indices = torch.topk(chunk_dense, min(top_k, chunk_dense.shape[1]), dim=1)
                    
                    # 只保留非负无穷的值
                    valid_mask = topk_values > -float('inf')
                    
                    # 获取有效值的行和列索引
                    # 生成行索引：为每个有效位置分配正确的行号
                    row_indices = torch.arange(chunk_dense.shape[0], device=self.device).unsqueeze(1).expand(-1, min(top_k, chunk_dense.shape[1]))
                    valid_rows = row_indices[valid_mask]
                    valid_values = topk_values[valid_mask]
                    valid_indices = topk_indices[valid_mask]
                    
                    # 调整行索引到全局位置
                    global_rows = valid_rows + start
                    
                    # 收集结果
                    all_rows.extend(global_rows.tolist())
                    all_cols.extend(valid_indices.tolist())
                    all_values.extend(valid_values.tolist())
                
                # 清理内存
                del chunk_dense
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # 对于密集矩阵，使用原来的分块逻辑
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                print(f"  Processing row chunk {start}-{end}")
                
                # 获取当前行块
                matrix_chunk = matrix_masked[start:end, :]
                mask_chunk = mask[start:end, :] if mask is not None else None
                
                # 对当前块进行top-k
                if mask_chunk is not None:
                    matrix_chunk[mask_chunk] = -float('inf')
                
                # 确保top_k是整数
                top_k = int(top_k)
                values, indices = torch.topk(matrix_chunk, k=min(top_k, n_cols), dim=1)
                
                # 收集结果
                rows = torch.arange(start, end, device=self.device).repeat_interleave(min(top_k, n_cols))
                all_rows.append(rows)
                all_cols.append(indices.flatten())
                all_values.append(values.flatten())
                
                # 清理内存
                del matrix_chunk, values, indices, rows
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合并所有结果 - 优化中间变量使用
        print("Merging chunk results...")
        if all_values:
            # 展平所有张量列表
            all_rows_flat = torch.cat(all_rows, dim=0)
            all_cols_flat = torch.cat(all_cols, dim=0)
            all_values_flat = torch.cat(all_values, dim=0)
            
            # 直接创建张量，避免中间变量
            sparse_matrix = torch.sparse_coo_tensor(
                torch.stack([all_rows_flat, all_cols_flat]), 
                all_values_flat, 
                (n_rows, n_cols), 
                device=self.device
            )
            
            # 清理中间变量
            del all_rows_flat, all_cols_flat, all_values_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # 如果没有任何元素，创建空稀疏矩阵
            sparse_matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
                (n_rows, n_cols),
                device=self.device
            )
        
        # 清理内存
        if 'all_rows' in locals():
            del all_rows, all_cols, all_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return sparse_matrix.coalesce()
    
    def row_normalize_sparse(self, sparse_matrix):
        """
        对稀疏矩阵进行行归一化
        
        参数:
            sparse_matrix: 输入稀疏矩阵
            
        返回:
            行归一化后的稀疏矩阵
        """
        # 确保稀疏矩阵已合并
        sparse_matrix = sparse_matrix.coalesce()
        
        # 计算行和（使用稀疏矩阵操作）
        row_indices = sparse_matrix.indices()[0]
        values = sparse_matrix.values()
        n_rows = sparse_matrix.shape[0]
        
        # 计算每行的和
        row_sums = torch.zeros(n_rows, device=self.device, dtype=values.dtype)
        row_sums.scatter_add_(0, row_indices, values)
        
        # 避免除以零
        row_sums[row_sums == 0] = 1
        
        # 应用行归一化
        indices = sparse_matrix.indices()
        normalized_values = values / row_sums[indices[0]]
        
        # 清理中间变量
        del row_sums, values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 创建归一化后的稀疏矩阵
        normalized_sparse = torch.sparse_coo_tensor(
            indices, normalized_values, sparse_matrix.shape, device=self.device
        )
        
        return normalized_sparse
    
    def process_chunk_with_similarity(self, R_chunk, S_r_chunk, text_features, visual_features, 
                                    alpha, beta, top_k, n_items, sim_chunk_size, precomputed_features=None):
        """
        处理单个数据块，同时计算相似度矩阵块，并应用历史交互置零
        
        参数:
            R_chunk: 当前块的稀疏交互矩阵
            S_r_chunk: 稀疏相似度矩阵块
            text_features: 文本特征矩阵
            visual_features: 视觉特征矩阵
            alpha: 文本相似度权重
            beta: 视觉相似度权重
            top_k: top-k值
            n_items: 总物品数
            sim_chunk_size: 相似度矩阵分块大小
            
        返回:
            稀疏矩阵 C_chunk
        """
        # 确保R_chunk是稀疏格式
        if not R_chunk.is_sparse:
            R_chunk = R_chunk.to_sparse()
            
        # 创建历史交互掩码
        if R_chunk.is_sparse:
            # 确保稀疏矩阵已合并
            R_chunk = R_chunk.coalesce()
            # 对于稀疏矩阵，提取非零位置作为掩码
            indices = R_chunk.indices()
            mask = torch.zeros((R_chunk.shape[0], n_items), dtype=torch.bool, device=self.device)
            mask[indices[0], indices[1]] = True
        else:
            # 对于密集矩阵，直接使用布尔掩码
            mask = R_chunk > 0
            
        # 初始化结果矩阵
        n_users = R_chunk.shape[0]
        if n_items > 50000:
            # 对于大数据集，使用稀疏矩阵初始化
            C_chunk = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty(0, device=self.device),
                (n_users, n_items),
                device=self.device
            )
        else:
            # 对于小数据集，使用密集矩阵
            C_chunk = torch.zeros((n_users, n_items), device=self.device)
        
        # 计算相似度矩阵块的数量
        n_sim_chunks = (n_items + sim_chunk_size - 1) // sim_chunk_size
        
        # 预计算标准化特征（如果未提供）
        if precomputed_features is None:
            text_norm, visual_norm = self.precompute_normalized_features(text_features, visual_features)
            # 确保特征在正确设备上
            text_norm = text_norm.to(self.device)
            visual_norm = visual_norm.to(self.device)
        else:
            text_norm, visual_norm = precomputed_features
            # 确保预计算特征在正确设备上
            text_norm = text_norm.to(self.device)
            visual_norm = visual_norm.to(self.device)
            # 如果使用预计算特征，清理原始特征
            del text_features, visual_features
            self._aggressive_memory_cleanup()
        
        for i in range(n_sim_chunks):
            # 内存监控
            if i % 5 == 0:
                print(f"  Memory status: {self._get_memory_info()}")
            
            # 计算文本相似度块（使用预计算的特征）
            if precomputed_features is not None:
                # 使用预计算特征，传递None作为原始特征
                S_t_subchunk, start, end = self.compute_similarity_chunk(
                    None, i, sim_chunk_size, n_items, text_norm
                )
                
                # 计算视觉相似度块（使用预计算的特征）
                S_v_subchunk, _, _ = self.compute_similarity_chunk(
                    None, i, sim_chunk_size, n_items, visual_norm
                )
            else:
                # 使用原始特征
                S_t_subchunk, start, end = self.compute_similarity_chunk(
                    text_features, i, sim_chunk_size, n_items, text_norm
                )
                
                # 计算视觉相似度块
                S_v_subchunk, _, _ = self.compute_similarity_chunk(
                    visual_features, i, sim_chunk_size, n_items, visual_norm
                )
            
            # 获取当前块的S_r并确保设备一致性
            if S_r_chunk.is_sparse:
                # 对于稀疏矩阵，直接使用稀疏矩阵切片（避免转换为密集矩阵）
                # 提取稀疏矩阵的列范围
                indices = S_r_chunk.indices()
                values = S_r_chunk.values()
                
                # 筛选出属于当前列范围的元素
                col_mask = (indices[1] >= start) & (indices[1] < end)
                chunk_indices = indices[:, col_mask]
                chunk_values = values[col_mask]
                
                # 调整列索引
                chunk_indices[1] = chunk_indices[1] - start
                
                # 创建稀疏子矩阵
                S_r_subchunk = torch.sparse_coo_tensor(
                    chunk_indices, chunk_values, 
                    (S_r_chunk.shape[0], end - start), 
                    device=self.device
                ).coalesce()
            else:
                S_r_subchunk = S_r_chunk[:, start:end].to(self.device)
            
            # 确保相似度块在正确设备上
            S_t_subchunk = S_t_subchunk.to(self.device)
            S_v_subchunk = S_v_subchunk.to(self.device)

            if n_items > 50000:
                # 对S_t_subchunk进行稀疏化，只保留10%的最大值
                S_t_subchunk = self._sparsify_matrix(S_t_subchunk, keep_ratio=0.1)
                # 对S_v_subchunk进行稀疏化，只保留10%的最大值
                S_v_subchunk = self._sparsify_matrix(S_v_subchunk, keep_ratio=0.1)
            
            # 确保alpha和beta是标量
            alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
            beta_val = beta.item() if torch.is_tensor(beta) else beta
            
            # 计算组合相似度块 - 避免中间变量，直接计算
            if n_items > 50000:
                # 分步计算，避免一次性创建大型稀疏矩阵
                # 确保加法顺序正确：密集矩阵在前，稀疏矩阵在后
                if alpha_val != 0:
                    S_combined_subchunk = S_r_subchunk + alpha_val * S_t_subchunk
                    # 立即清理S_t_subchunk
                    del S_t_subchunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    S_combined_subchunk = S_r_subchunk
                
                if beta_val != 0:
                    S_combined_subchunk = S_combined_subchunk + beta_val * S_v_subchunk
                    # 立即清理S_v_subchunk
                    del S_v_subchunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 只在必要时进行coalesce
                if S_combined_subchunk.is_sparse:
                    S_combined_subchunk = S_combined_subchunk.coalesce()
            else:
                # 小数据集直接计算，避免中间变量
                # 确保加法顺序正确：密集矩阵在前，稀疏矩阵在后
                if S_r_subchunk.is_sparse:
                    # 如果S_r_subchunk是稀疏矩阵，先计算密集部分，再与稀疏矩阵相加
                    dense_part = alpha_val * S_t_subchunk + beta_val * S_v_subchunk
                    S_combined_subchunk = dense_part + S_r_subchunk
                    del dense_part
                else:
                    # 如果S_r_subchunk是密集矩阵，直接计算
                    S_combined_subchunk = S_r_subchunk + alpha_val * S_t_subchunk + beta_val * S_v_subchunk
                
                # 立即清理不需要的变量
                del S_t_subchunk, S_v_subchunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 执行稀疏-密集矩阵乘法
            C_subchunk = torch.sparse.mm(R_chunk, S_combined_subchunk)
            
            # 如果n_items大于5万，确保C_subchunk也是稀疏格式
            if n_items > 50000:
                # 对C_subchunk进行稀疏化，每行最多保留2%的值
                if C_subchunk.is_sparse:
                    # 直接在稀疏格式上进行稀疏化，避免转换为密集矩阵
                    C_subchunk = self._sparsify_sparse_matrix_by_row(C_subchunk.to_dense(), keep_ratio=0.01)
                else:
                    # 直接进行稀疏化
                    C_subchunk = self._sparsify_sparse_matrix_by_row(C_subchunk, keep_ratio=0.01)

            
            # 在每个子块处理后进行内存清理
            if i % 3 == 0:
                self._aggressive_memory_cleanup()
            
            # 累加到结果矩阵 - 优化中间变量使用
            if n_items > 50000:
                # 对于大数据集，需要将C_subchunk的列索引调整到正确位置
                if C_chunk._values().numel() == 0:
                    # 如果C_chunk是空的，直接调整C_subchunk的列索引，避免中间变量
                    indices = C_subchunk.indices()
                    indices[1] = indices[1] + start  # 直接修改索引
                    C_chunk = torch.sparse_coo_tensor(
                        indices, C_subchunk.values(), (C_chunk.shape[0], C_chunk.shape[1]), device=self.device
                    ).coalesce()
                else:
                    # 如果C_chunk已有数据，直接调整并合并，避免创建中间变量
                    indices = C_subchunk.indices()
                    indices[1] = indices[1] + start  # 直接修改索引
                    
                    # 直接创建调整后的稀疏矩阵并合并
                    C_chunk = C_chunk + torch.sparse_coo_tensor(
                        indices, C_subchunk.values(), C_chunk.shape, device=self.device
                    ).coalesce()
                    # 合并重复索引
                    C_chunk = C_chunk.coalesce()
            else:
                # 对于小数据集，使用密集矩阵累加
                C_chunk[:, start:end] = C_subchunk
        
        # 对完整行进行top-k稀疏化，同时应用历史交互掩码
        sparse_C = self.topk_sparse_with_mask(C_chunk, top_k, mask)
        
        # 行归一化
        normalized_sparse_C = self.row_normalize_sparse(sparse_C)
        
        return normalized_sparse_C
    
    def _compute_sparse_mm_chunked(self, R, S_combined, chunk_size=5000):
        """
        分块计算稀疏矩阵乘法，避免内存溢出
        """
        print(f"Computing matrix multiplication in chunks, chunk size: {chunk_size}")
        
        n_users, n_items = R.shape
        result_chunks = []
        
        # 分块处理用户
        for start in range(0, n_users, chunk_size):
            end = min(start + chunk_size, n_users)
            print(f"  Processing user chunk {start}-{end}")
            
            # 获取当前用户块
            if R.is_sparse:
                # 稀疏矩阵分块
                indices = R.indices()
                mask = (indices[0] >= start) & (indices[0] < end)
                chunk_indices = indices[:, mask]
                chunk_values = R.values()[mask]
                
                # 调整行索引
                chunk_indices[0] = chunk_indices[0] - start
                
                R_chunk = torch.sparse_coo_tensor(
                    chunk_indices, chunk_values, (end - start, n_items), device=self.device
                )
            else:
                R_chunk = R[start:end, :].to(self.device)
                if not R_chunk.is_sparse:
                    R_chunk = R_chunk.to_sparse()
            
            # 计算当前块的结果 - R_chunk是(n_users_chunk, n_items)，S_combined是(n_items, n_items)
            if R_chunk.is_sparse:
                # 使用稀疏矩阵乘法
                C_chunk = torch.sparse.mm(R_chunk, S_combined)
            else:
                C_chunk = torch.mm(R_chunk, S_combined)
            result_chunks.append(C_chunk)
            
            # 清理内存
            del R_chunk
            self._aggressive_memory_cleanup()
        
        # 合并结果
        print("Merging chunk results...")
        C = torch.cat(result_chunks, dim=0)
        
        # 清理内存
        del result_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return C
    
    
    def compute_potential_interaction(self, R, 
                                 S_r, alpha, beta, top_k, use_chunking, text_features=None, visual_features=None,
                                 chunk_size=2500, sim_chunk_size=10000, S_t=None, S_v=None, precomputed_features=None):
        """
        计算潜在交互矩阵 C = topksparse(R(S_r + alpha*S_t + beta*S_v))
        
        参数:
            
        返回:
            稀疏矩阵 C
        """
        start_time = time.time()
        print(f"Starting computation. Initial memory: {self._get_memory_info()}")
        
        # 确保alpha和beta是标量
        if torch.is_tensor(alpha):
            alpha = alpha.item()
        if torch.is_tensor(beta):
            beta = beta.item()
        
        n_items = R.shape[1]
        n_users = R.shape[0]
            
        if not use_chunking:
            # 小数据集直接计算
            print("Computing combined similarity matrix...")
            if S_r.is_sparse:
                # 对于稀疏矩阵，直接使用稀疏格式
                S_r_sparse = S_r
            else:
                S_r_sparse = S_r.to_sparse()
            
            # 如果提供了预计算特征，直接使用；否则使用传入的S_t和S_v
            if precomputed_features is not None:
                text_norm, visual_norm = precomputed_features
                # 使用预计算特征计算相似度矩阵
                S_t = torch.mm(text_norm, text_norm.t())
                S_v = torch.mm(visual_norm, visual_norm.t())
                # 确保设备一致性
                S_t = S_t.to(self.device)
                S_v = S_v.to(self.device)
                # 清理预计算特征以节省内存
                del text_norm, visual_norm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                # 确保传入的S_t和S_v在正确设备上
                S_t = S_t.to(self.device)
                S_v = S_v.to(self.device)
                
            # 确保S_r_dense在正确设备上
            S_r_sparse = S_r_sparse.to(self.device)
            
            # 分步计算S_combined，避免内存峰值和中间变量
            print("Computing combined similarity matrix step by step...")
            # 直接使用S_r_dense，避免clone()
            S_combined = S_r_sparse
            
            # 添加文本相似度，避免中间变量
            if S_t is not None:
                S_combined =  alpha * S_t + S_combined
                # 立即清理S_t
                del S_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 添加视觉相似度，避免中间变量
            if S_v is not None:
                S_combined =  beta * S_v + S_combined
                # 立即清理S_v
                del S_v
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 释放原始矩阵内存
            del S_r_sparse
            self._aggressive_memory_cleanup()
            
            print("Computing matrix product...")
            # 确保R是稀疏格式并在正确设备上
            if not R.is_sparse:
                R = R.to_sparse()
            R = R.to(self.device)
            
            # 检查内存使用情况，如果S_combined太大则分块处理
            if S_combined.numel() > 100000000:  # 4亿个元素
                print("WARNING: Matrix too large, using chunked processing to avoid memory overflow")
                C = self._compute_sparse_mm_chunked(R, S_combined, chunk_size=3000)  # 减小分块大小
            else:
                # 在移动到GPU前进行内存检查
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    if current_memory > total_memory * 0.8:  # 如果已使用超过80%内存
                        print("WARNING: High memory usage detected, switching to chunked processing")
                        C = self._compute_sparse_mm_chunked(R, S_combined, chunk_size=3000)
                    else:
                        S_combined = S_combined.to(self.device)
                        C = torch.sparse.mm(R, S_combined)
                        
                        # 立即释放S_combined内存
                        del S_combined
                        torch.cuda.empty_cache()
                else:
                    S_combined = S_combined.to(self.device)
                    C = torch.sparse.mm(R, S_combined)
                    
                    # 立即释放S_combined内存
                    del S_combined

            # 创建历史交互掩码
            if R.is_sparse:
                indices = R.indices()
                mask = torch.zeros((R.shape[0], n_items), dtype=torch.bool, device=self.device)
                mask[indices[0], indices[1]] = True
            else:
                mask = R > 0
            
            if isinstance(top_k,list):
                result = []
                for k in top_k:
                    print("Applying top-k sparsification and historical interaction masking...")
                    sparse_C = self.topk_sparse_with_mask(C, k, mask)
                    
                    print("Applying row normalization...")
                    normalized_C = self.row_normalize_sparse(sparse_C)
                    result.append(normalized_C)
                    
            else:
                print("Applying top-k sparsification and historical interaction masking...")
                sparse_C = self.topk_sparse_with_mask(C, top_k, mask)
                
                print("Applying row normalization...")
                result = self.row_normalize_sparse(sparse_C)
                
        else:
            # 大数据集分块处理
            print("Using chunked processing...")
            n_chunks = (n_users + chunk_size - 1) // chunk_size
            
            # 预计算标准化特征（避免重复计算）
            if precomputed_features is None:
                precomputed_features = self.precompute_normalized_features(text_features, visual_features)
                # 确保预计算特征在正确设备上
                text_norm, visual_norm = precomputed_features
                precomputed_features = (text_norm.to(self.device), visual_norm.to(self.device))
                # 清理原始特征以节省内存
                del text_features, visual_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 检查是否需要处理多个top_k值
            if isinstance(top_k, list):
                print(f"Processing multiple top_k values: {top_k}")
                # 为每个top_k值初始化结果列表
                result_chunks_list = [[] for _ in top_k]
            else:
                result_chunks = []
            

            for i in range(n_chunks):
                print(f"Processing user chunk {i+1}/{n_chunks}...")
                print(f"Memory status: {self._get_memory_info()}")
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_users)
                
                # 获取当前用户块的R
                if R.is_sparse:
                    # 对于稀疏矩阵，提取指定行范围
                    indices = R.indices()
                    values = R.values()
                    
                    # 筛选出属于当前块的行
                    mask = (indices[0] >= start) & (indices[0] < end)
                    chunk_indices = indices[:, mask]
                    chunk_values = values[mask]
                    
                    # 调整行索引
                    chunk_indices[0] = chunk_indices[0] - start
                    
                    R_chunk = torch.sparse_coo_tensor(
                        chunk_indices, chunk_values, (end - start, n_items), device=self.device
                    )
                else:
                    R_chunk = R[start:end, :]
                    if not R_chunk.is_sparse:
                        R_chunk = R_chunk.to_sparse()
                
                # 获取当前块的S_r并确保设备一致性
                if S_r.is_sparse:
                    S_r_chunk = S_r.to(self.device)
                else:
                    S_r_chunk = S_r[start:end, :].to(self.device)
                
                # 处理当前块（同时计算相似度矩阵块）
                if isinstance(top_k, list):
                    # 为每个top_k值计算稀疏化结果
                    for j, k in enumerate(top_k):
                        print(f"  Processing sparsification for top_k={k}...")
                        C_chunk = self.process_chunk_with_similarity(
                            R_chunk, S_r_chunk, None, None,
                            alpha, beta, k, n_items, sim_chunk_size, precomputed_features
                        )
                        result_chunks_list[j].append(C_chunk)
                else:
                    C_chunk = self.process_chunk_with_similarity(
                        R_chunk, S_r_chunk, None, None,
                        alpha, beta, top_k, n_items, sim_chunk_size, precomputed_features
                    )
                    result_chunks.append(C_chunk)

                
                # 定期清理内存
                if i % 3 == 0:  # 更频繁的清理
                    self._aggressive_memory_cleanup()
                    print(f"  After cleanup: {self._get_memory_info()}")
                
                # 清理当前块的中间变量
                del C_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            

            # 合并结果
            if isinstance(top_k, list):
                # 处理多个top_k值的结果
                result = []
                for j, k in enumerate(top_k):
                    print(f"Merging results for top_k={k}...")
                    if all(chunk.is_sparse for chunk in result_chunks_list[j]):
                        # 合并稀疏矩阵
                        indices_list = []
                        values_list = []
                        offset = 0
                        
                        for chunk in result_chunks_list[j]:
                            # 确保稀疏矩阵已合并
                            chunk = chunk.coalesce()
                            chunk_indices = chunk.indices()
                            chunk_values = chunk.values()
                            
                            # 调整行索引
                            chunk_indices[0] = chunk_indices[0] + offset
                            indices_list.append(chunk_indices)
                            values_list.append(chunk_values)
                            
                            offset += chunk.shape[0]
                        
                        # 合并所有索引和值
                        all_indices = torch.cat(indices_list, dim=1)
                        all_values = torch.cat(values_list, dim=0)
                        
                        result.append(torch.sparse_coo_tensor(
                            all_indices, all_values, (n_users, n_items), device=self.device
                        ))
                    else:
                        # 合并密集矩阵
                        result.append(torch.cat(result_chunks_list[j], dim=0))
            else:
                # 处理单个top_k值的结果
                if all(chunk.is_sparse for chunk in result_chunks):
                    # 合并稀疏矩阵 - 优化中间变量使用
                    indices_list = []
                    values_list = []
                    offset = 0
                    
                    for chunk in result_chunks:
                        # 确保稀疏矩阵已合并
                        chunk = chunk.coalesce()
                        chunk_indices = chunk.indices()
                        chunk_values = chunk.values()
                        
                        # 直接修改行索引，避免创建副本
                        chunk_indices[0] = chunk_indices[0] + offset
                        indices_list.append(chunk_indices)
                        values_list.append(chunk_values)
                        
                        offset += chunk.shape[0]
                        # 立即清理chunk
                        del chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # 合并所有索引和值，避免中间变量
                    result = torch.sparse_coo_tensor(
                        torch.cat(indices_list, dim=1), 
                        torch.cat(values_list, dim=0), 
                        (n_users, n_items), 
                        device=self.device
                    )
                    
                    # 清理中间列表
                    del indices_list, values_list
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    # 合并密集矩阵
                    result = torch.cat(result_chunks, dim=0)
            


        elapsed = time.time() - start_time
        print(f"Computation completed, elapsed time: {elapsed:.2f} seconds")
        print(f"Final memory: {self._get_memory_info()}")
        
        # 最终内存清理
        self._aggressive_memory_cleanup()
        print(f"After final cleanup: {self._get_memory_info()}")
        
        return result

