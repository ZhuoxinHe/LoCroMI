# coding: utf-8
import math
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_mixed_graph, build_non_zero_graph, build_knn_normalized_graph,build_graph_from_adj

class LoCroMI(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LoCroMI, self).__init__(config, dataset)
        self.config = config  # 保存配置以便后续使用
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.cl_loss2 = config['cl_loss2']
        self.n_ui_layers = config['n_ui_layers']
        self.n_layers = config['n_layers']
        self.embedding_dim = config['embedding_size']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.PI_loss = config['PI_loss']
        self.top_k = config['top_k']
        self.knn_k = config['knn_k']
        self.knn_i = config['knn_i']
        # self.co_lambda = config['co_lambda']
        self.O = config['O']
        self.relabel_rate = config['relabel_rate']

        self.eval_inter = None
        self.his_F1 = [0]

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # denoise info
        self.loss_mean = torch.sparse_coo_tensor(
            indices=torch.empty((2, 0), dtype=torch.float32),
            values=torch.empty((0,)),
            size=(self.n_users, self.n_items)).to(self.device)
        # self.loss_mean = torch.zeros(self.n_users, self.n_items, dtype=torch.float32)
        self.loss_variance = torch.zeros_like(self.loss_mean).to(self.device)
        # ############

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)

        if self.n_items <= 50000:
            self.image_adj = build_sim(self.image_embedding.weight.detach())
            self.text_adj = build_sim(self.text_embedding.weight.detach())
            self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
            self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        else:
            # 处理大规模数据集：分段计算并稀疏化
            image_adj, text_adj = self._build_large_scale_adjacency_matrices()
            self.image_adj = image_adj.to(self.device)
            self.text_adj = text_adj.to(self.device)

        # ##############################first stage start######################################
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        PI_file = os.path.join(dataset_path, 'PI_{}_{}_{}.pt'.format(self.alpha, self.beta, self.top_k))
        self.sparse_C = torch.load(PI_file, map_location=self.device, weights_only=True).coalesce()

        interest_file = os.path.join(dataset_path, 'interest_{}_{}_{}.pt'.format(self.alpha, self.beta, self.knn_i))
        interest_matrix = torch.load(interest_file, map_location=self.device, weights_only=True)

        # mixed graph
        self.image_adj = torch.add(interest_matrix, self.image_adj)
        self.text_adj = torch.add(interest_matrix, self.text_adj)
        # ###########################first stage end####################################

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.map_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.map_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        

    # def compute_DR_matrix_vectorized(self, X: torch.Tensor, block_size=1000) -> torch.Tensor:
    #     """
    #     向量化计算 DR 矩阵
    #     """
    #     n, d = X.shape
    #     DR = torch.zeros((n, n), device=X.device)
    #
    #     for i in range(0, n, block_size):
    #         for j in range(0, n, block_size):
    #             # 获取当前块的起始和结束索引
    #             i_end = min(i + block_size, n)
    #             j_end = min(j + block_size, n)
    #
    #             # 提取当前块的子矩阵
    #             X_i = X[i:i_end].unsqueeze(1)  # (block_size, 1, d)
    #             X_j = X[j:j_end].unsqueeze(0)  # (1, block_size, d)
    #
    #             # 计算当前块的 A 和 D
    #             A = (X_i >= X_j).sum(dim=2).float()  # (block_size, block_size)
    #             D = (X_i < X_j).sum(dim=2).float()  # (block_size, block_size)
    #
    #             # 填充 DR 矩阵的对应块
    #             DR[i:i_end, j:j_end] = torch.abs(A - D) / d
    #
    #     return DR

    def scipy_coo_to_torch_sparse(self, scipy_coo, dtype=torch.float32) -> torch.Tensor:
        """
        将SciPy的COO稀疏矩阵转换为PyTorch稀疏张量

        Args:
            scipy_coo (coo_matrix): SciPy的COO格式稀疏矩阵
            dtype (torch.dtype): 输出张量的数据类型，默认float32

        Returns:
            torch.Tensor: PyTorch稀疏张量（COO格式）
        """
        # 提取行索引、列索引和非零值
        rows = scipy_coo.row.astype(np.int32)
        cols = scipy_coo.col.astype(np.int32)
        data = scipy_coo.data.astype(np.float32)  # 根据实际数据类型调整

        # 转换为PyTorch张量
        indices = torch.from_numpy(np.vstack((rows, cols))).to(self.device)
        values = torch.from_numpy(data).to(dtype).to(self.device)

        # 构造稀疏张量
        shape = scipy_coo.shape
        torch_sparse = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=shape,
        ).to(self.device)

        # 合并重复索引（可选，确保无重复）
        torch_sparse = torch_sparse.coalesce()

        return torch_sparse

    def pre_epoch_processing(self):
        pass
    
    def _get_memory_usage_gb(self):
        """Get current memory usage (GB) using torch"""
        if torch.cuda.is_available():
            # Use CUDA memory if available
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            cached = torch.cuda.memory_reserved() / (1024 ** 3)
            return allocated + cached
        else:
            # Fallback: estimate based on matrix size
            return 0.0
    
    def _force_garbage_collection(self):
        """强制垃圾回收"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _estimate_matrix_memory_gb(self, shape):
        """估算矩阵内存需求"""
        return (shape[0] * shape[1] * 4) / (1024 ** 3)  # 4字节per float32
    
    def _should_use_large_scale_mode(self, matrix_shape):
        """Determine whether to use large-scale matrix processing mode based on matrix size only"""
        # Simple size-based judgment
        return (matrix_shape[0] > 100000 or matrix_shape[1] > 100000)

    def get_adj_mat(self):
        """
        大矩阵处理版本的邻接矩阵构建
        使用分块处理和稀疏矩阵优化来避免内存溢出
        """
        # Check matrix size and decide processing mode
        total_size = self.n_users + self.n_items
        matrix_shape = (total_size, total_size)
        
        # Simple size-based judgment
        use_large_scale = self._should_use_large_scale_mode(matrix_shape)
        
        print(f"Matrix size: {total_size}x{total_size}")
        
        if use_large_scale:
            print("Using large-scale matrix processing mode...")
            return self._get_adj_mat_large_scale()
        else:
            print("Using standard processing mode...")
            return self._get_adj_mat_standard()
    
    def _get_adj_mat_standard(self):
        """标准邻接矩阵构建方法"""
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_inv[np.isnan(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()
    
    def _get_adj_mat_large_scale(self):
        """
        大矩阵处理版本的邻接矩阵构建
        使用分块处理和稀疏矩阵优化
        """
        print("Starting large-scale adjacency matrix construction...")
        
        try:
            # Monitor memory usage
            start_memory = self._get_memory_usage_gb()
            print(f"Memory usage at start: {start_memory:.2f}GB")
            
            # Use COO format directly to avoid LIL conversion
            R_coo = self.interaction_matrix.tocoo()
            print(f"Interaction matrix: {R_coo.shape}, non-zero elements: {R_coo.nnz}")
            
            # Build block adjacency matrix
            adj_mat = self._build_block_adjacency_matrix(R_coo)
            
            # Monitor memory usage
            after_adj_memory = self._get_memory_usage_gb()
            print(f"Memory usage after adjacency matrix: {after_adj_memory:.2f}GB")
            
            # Block normalization
            norm_adj_mat = self._block_normalize_adjacency(adj_mat)
            
            # Monitor memory usage
            after_norm_memory = self._get_memory_usage_gb()
            print(f"Memory usage after normalization: {after_norm_memory:.2f}GB")
            
            # Extract R matrix
            self.R = self._extract_R_matrix(norm_adj_mat)
            
            # Clean up intermediate variables
            del adj_mat
            self._force_garbage_collection()
            
            final_memory = self._get_memory_usage_gb()
            print(f"Final memory usage: {final_memory:.2f}GB")
            
            return norm_adj_mat.tocsr()
            
        except MemoryError as e:
            print(f"Memory error: {e}")
            print("Trying more aggressive memory optimization...")
            return self._get_adj_mat_ultra_large_scale()
        except Exception as e:
            print(f"Large-scale processing error: {e}")
            print("Falling back to standard processing mode...")
            return self._get_adj_mat_standard()
    
    def _build_block_adjacency_matrix(self, R_coo):
        """
        分块构建邻接矩阵，避免内存溢出
        """
        n_users, n_items = self.n_users, self.n_items
        total_size = n_users + n_items
        
        print(f"Building block adjacency matrix: {total_size}x{total_size}")
        
        # Create empty COO matrix to store results
        all_rows = []
        all_cols = []
        all_data = []
        
        # Add user-item interactions (R matrix)
        if R_coo.nnz > 0:
            all_rows.extend(R_coo.row.tolist())
            all_cols.extend((R_coo.col + n_users).tolist())
            all_data.extend(R_coo.data.tolist())
        
        # Add item-user interactions (R.T matrix)
        if R_coo.nnz > 0:
            all_rows.extend((R_coo.col + n_users).tolist())
            all_cols.extend(R_coo.row.tolist())
            all_data.extend(R_coo.data.tolist())
        
        # Create sparse matrix
        if all_rows:
            adj_mat = sp.coo_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(total_size, total_size),
                dtype=np.float32
            )
        else:
            # Create empty matrix if no data
            adj_mat = sp.coo_matrix((total_size, total_size), dtype=np.float32)
        
        print(f"Adjacency matrix construction completed, non-zero elements: {adj_mat.nnz}")
        return adj_mat
    
    def _block_normalize_adjacency(self, adj_mat):
        """
        分块归一化邻接矩阵 - 只使用稀疏矩阵操作
        """
        print("Starting block normalization...")
        
        # 使用稀疏矩阵操作计算度矩阵，避免密集矩阵
        rowsum_sparse = adj_mat.sum(axis=1)
        
        # 转换为稀疏格式处理
        if hasattr(rowsum_sparse, 'A'):
            rowsum = rowsum_sparse.A.flatten()
        else:
            rowsum = rowsum_sparse.toarray().flatten()
        
        # Handle zero-degree nodes
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_inv[np.isnan(d_inv)] = 0.
        
        # Create sparse degree matrix
        d_mat_inv = sp.diags(d_inv)
        
        # Block normalization calculation using sparse operations only
        norm_adj = self._block_matrix_multiply(d_mat_inv, adj_mat, d_mat_inv)
        
        print("Block normalization completed")
        return norm_adj
    
    def _block_matrix_multiply(self, D_inv, A, D_inv2):
        """
        分块矩阵乘法，避免内存溢出
        """
        # Step 1: D_inv * A
        print("Computing D^(-1/2) * A...")
        res = D_inv.dot(A)
        
        # Step 2: (D_inv * A) * D_inv
        print("Computing (D^(-1/2) * A) * D^(-1/2)...")
        res = res.dot(D_inv2)
        
        return res.tocoo()
    
    def _extract_R_matrix(self, norm_adj_mat):
        """
        从归一化邻接矩阵中提取R矩阵 - 使用稀疏操作避免密集矩阵
        """
        print("Extracting R matrix...")
        
        # 确保输入是稀疏格式
        if not sp.isspmatrix(norm_adj_mat):
            norm_adj_mat = norm_adj_mat.tocoo()
        
        # 使用稀疏矩阵切片操作提取用户-物品部分
        # 避免创建密集矩阵
        if sp.isspmatrix_coo(norm_adj_mat):
            # 对于COO格式，直接操作索引
            mask = (norm_adj_mat.row < self.n_users) & (norm_adj_mat.col >= self.n_users)
            R_rows = norm_adj_mat.row[mask]
            R_cols = norm_adj_mat.col[mask] - self.n_users  # 调整列索引
            R_data = norm_adj_mat.data[mask]
            
            R_coo = sp.coo_matrix(
                (R_data, (R_rows, R_cols)),
                shape=(self.n_users, self.n_items),
                dtype=np.float32
            )
        else:
            # 对于其他格式，转换为COO后处理
            norm_adj_coo = norm_adj_mat.tocoo()
            mask = (norm_adj_coo.row < self.n_users) & (norm_adj_coo.col >= self.n_users)
            R_rows = norm_adj_coo.row[mask]
            R_cols = norm_adj_coo.col[mask] - self.n_users
            R_data = norm_adj_coo.data[mask]
            
            R_coo = sp.coo_matrix(
                (R_data, (R_rows, R_cols)),
                shape=(self.n_users, self.n_items),
                dtype=np.float32
            )
                  
        print(f"R matrix extraction completed, shape: {R_coo.shape}, non-zero elements: {R_coo.nnz}")
        return R_coo
    
    def _get_adj_mat_ultra_large_scale(self):
        """
        超大规模矩阵处理，使用最激进的内存优化策略
        """
        print("Using ultra-large-scale processing mode...")
        
        try:
            # Use smaller chunk size
            self.chunk_size = 100000  # Reduce chunk size
            
            # Build sparse matrix directly, avoid intermediate storage
            R_coo = self.interaction_matrix.tocoo()
            
            # Block processing, convert to sparse format immediately
            norm_adj_mat = self._ultra_large_scale_normalize(R_coo)
            
            # Extract R matrix
            self.R = self._extract_R_matrix(norm_adj_mat)
            
            return norm_adj_mat.tocsr()
            
        except Exception as e:
            print(f"Ultra-large-scale processing also failed: {e}")
            print("Using minimal memory mode...")
            return self._get_adj_mat_minimal_memory()
    
    def _ultra_large_scale_normalize(self, R_coo):
        """
        超大规模归一化，使用最小内存策略
        """
        print("Starting ultra-large-scale normalization...")
        
        n_users, n_items = self.n_users, self.n_items
        total_size = n_users + n_items
        
        # Block compute degree matrix
        degrees = self._compute_degrees_in_chunks(R_coo, total_size)
        
        # Create normalized matrix
        norm_adj = self._create_normalized_adjacency_chunked(R_coo, degrees, total_size)
        
        return norm_adj
    
    def _compute_degrees_in_chunks(self, R_coo, total_size):
        """分块计算度矩阵 - 使用稀疏操作避免密集矩阵"""
        print("Computing degree matrix in chunks...")
        
        degrees = np.zeros(total_size, dtype=np.float32)
        
        # User degrees - 使用稀疏操作
        user_degrees_sparse = R_coo.sum(axis=1)
        if hasattr(user_degrees_sparse, 'A'):
            user_degrees = user_degrees_sparse.A.flatten()
        else:
            user_degrees = user_degrees_sparse.toarray().flatten()
        degrees[:self.n_users] = user_degrees
        
        # Item degrees - 使用稀疏操作
        item_degrees_sparse = R_coo.sum(axis=0)
        if hasattr(item_degrees_sparse, 'A'):
            item_degrees = item_degrees_sparse.A.flatten()
        else:
            item_degrees = item_degrees_sparse.toarray().flatten()
        degrees[self.n_users:] = item_degrees
        
        # Handle zero-degree nodes
        degrees = np.power(degrees, -0.5)
        degrees[np.isinf(degrees)] = 0.
        degrees[np.isnan(degrees)] = 0.
        
        return degrees
    
    def _create_normalized_adjacency_chunked(self, R_coo, degrees, total_size):
        """分块创建归一化邻接矩阵"""
        print("Creating normalized adjacency matrix in chunks...")
        
        # Block process R matrix
        chunk_size = 10000
        all_rows = []
        all_cols = []
        all_data = []
        
        # Process user-item part
        for start_row in range(0, self.n_users, chunk_size):
            end_row = min(start_row + chunk_size, self.n_users)
            
            # Get current block interactions
            mask = (R_coo.row >= start_row) & (R_coo.row < end_row)
            if not mask.any():
                continue
                
            chunk_rows = R_coo.row[mask]
            chunk_cols = R_coo.col[mask] + self.n_users
            chunk_data = R_coo.data[mask]
            
            # Apply normalization
            norm_data = chunk_data * degrees[chunk_rows] * degrees[chunk_cols]
            
            all_rows.extend(chunk_rows.tolist())
            all_cols.extend(chunk_cols.tolist())
            all_data.extend(norm_data.tolist())
            
            # Add transpose part
            all_rows.extend(chunk_cols.tolist())
            all_cols.extend(chunk_rows.tolist())
            all_data.extend(norm_data.tolist())
        
        # Create sparse matrix
        if all_rows:
            norm_adj = sp.coo_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(total_size, total_size),
                dtype=np.float32
            )
        else:
            norm_adj = sp.coo_matrix((total_size, total_size), dtype=np.float32)
        
        return norm_adj
    
    def _get_adj_mat_minimal_memory(self):
        """
        最小内存模式，只构建必要的矩阵 - 避免密集矩阵
        """
        print("Using minimal memory mode...")
        
        # Only build R matrix, not the complete adjacency matrix
        R_coo = self.interaction_matrix.tocoo()
        
        # Calculate R matrix normalization using sparse operations
        user_degrees_sparse = R_coo.sum(axis=1)
        item_degrees_sparse = R_coo.sum(axis=0)
        
        # Convert to arrays safely
        if hasattr(user_degrees_sparse, 'A'):
            user_degrees = user_degrees_sparse.A.flatten()
        else:
            user_degrees = user_degrees_sparse.toarray().flatten()
            
        if hasattr(item_degrees_sparse, 'A'):
            item_degrees = item_degrees_sparse.A.flatten()
        else:
            item_degrees = item_degrees_sparse.toarray().flatten()
        
        # Normalization
        user_degrees = np.power(user_degrees, -0.5)
        item_degrees = np.power(item_degrees, -0.5)
        
        user_degrees[np.isinf(user_degrees)] = 0.
        item_degrees[np.isinf(item_degrees)] = 0.
        
        # Create normalized R matrix
        norm_data = R_coo.data * user_degrees[R_coo.row] * item_degrees[R_coo.col]
        
        self.R = sp.coo_matrix(
            (norm_data, (R_coo.row, R_coo.col)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        # Create a dummy adjacency matrix (not actually used)
        dummy_adj = sp.identity(self.n_users + self.n_items, dtype=np.float32, format='csr')
        
        print("Minimal memory mode completed")
        return dummy_adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """
        优化的稀疏矩阵转换方法
        支持大矩阵处理，减少内存占用
        """
        # 检查矩阵大小
        estimated_memory_mb = (sparse_mx.nnz * 8) / (1024 ** 2)  # 估算内存需求(MB)
        
        if estimated_memory_mb > 1000:  # If over 1GB, use block processing
            print(f"Large matrix conversion: {sparse_mx.shape}, non-zero elements: {sparse_mx.nnz}, estimated memory: {estimated_memory_mb:.2f}MB")
            return self._sparse_mx_to_torch_large_scale(sparse_mx)
        else:
            return self._sparse_mx_to_torch_standard(sparse_mx)
    
    def _sparse_mx_to_torch_standard(self, sparse_mx):
        """标准稀疏矩阵转换"""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
    
    def _sparse_mx_to_torch_large_scale(self, sparse_mx):
        """
        大矩阵稀疏转换，使用分块处理
        """
        print("Starting large matrix sparse conversion...")
        
        # Ensure COO format
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        
        # Block process indices and values
        chunk_size = 1000000  # Process 1 million non-zero elements at a time
        nnz = sparse_mx.nnz
        
        if nnz == 0:
            # Empty matrix
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64),
                torch.empty(0, dtype=torch.float32),
                torch.Size(sparse_mx.shape)
            )
        
        # Block processing
        all_indices = []
        all_values = []
        
        for start_idx in range(0, nnz, chunk_size):
            end_idx = min(start_idx + chunk_size, nnz)
            
            # Extract current block data
            chunk_rows = sparse_mx.row[start_idx:end_idx]
            chunk_cols = sparse_mx.col[start_idx:end_idx]
            chunk_data = sparse_mx.data[start_idx:end_idx]
            
            # Convert to torch tensors
            chunk_indices = torch.from_numpy(
                np.vstack([chunk_rows, chunk_cols]).astype(np.int64)
            )
            chunk_values = torch.from_numpy(chunk_data.astype(np.float32))
            
            all_indices.append(chunk_indices)
            all_values.append(chunk_values)
        
        # Merge all blocks
        final_indices = torch.cat(all_indices, dim=1)
        final_values = torch.cat(all_values)
        
        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            final_indices,
            final_values,
            torch.Size(sparse_mx.shape)
        )
        
        print(f"Large matrix conversion completed, shape: {sparse_tensor.shape}, non-zero elements: {sparse_tensor._nnz()}")
        return sparse_tensor

    def forward(self, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

            # Feature ID Embedding
        image_item_embeds = torch.multiply(self.item_embedding.weight, self.map_v(image_feats))
        text_item_embeds = torch.multiply(self.item_embedding.weight, self.map_t(text_feats))

        item_embeds = self.item_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Interest-Aware Item Graph Convolution
        for i in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        for i in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Attention Fuser
        att_common = torch.cat([self.attention(image_embeds), self.attention(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        side_embeds = (image_embeds + text_embeds - common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        return bpr_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def get_batch_PI_rel(self, batch_users):
        # 提取所有非零用户-物品对
        all_users, all_items = self.sparse_C.indices()
        all_weights = self.sparse_C.values()

        # 筛选属于当前批次的用户
        mask = torch.isin(all_users, batch_users)
        batch_users = all_users[mask]
        batch_items = all_items[mask]
        batch_weights = all_weights[mask]

        return batch_users, batch_items, batch_weights

    def cal_PI_loss(self, batch_users, ua_embeddings, ia_embeddings, is_post_process=False):
        batch_users, batch_items, batch_weights = self.get_batch_PI_rel(batch_users)
        users_embed = ua_embeddings[batch_users]
        items_embed = ia_embeddings[batch_items]

        # batch_weights = (batch_weights-torch.min(batch_weights))/(torch.max(batch_weights)-min(batch_weights))
        # 计算内积并加权
        inner_prod = F.logsigmoid(torch.sum(torch.mul(users_embed, items_embed), dim=1))  #

        if is_post_process:
            fuzzy_loss_all = -inner_prod # * batch_weights
            return batch_users, batch_items, batch_weights, fuzzy_loss_all

        weighted_loss = -torch.mean(inner_prod * batch_weights)  #
        return weighted_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        # item-item constractive loss
        cl_loss = (self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2))
        # user-item constractive loss
        cl_loss2 = (self.InfoNCE(u_g_embeddings, content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            u_g_embeddings, side_embeds_items[pos_items], 0.2))
        PI_loss = self.cal_PI_loss(users, ua_embeddings, ia_embeddings)

        # self.noised_sample_relabel(epoch_idx, batch_users, batch_items, batch_weights, fuzzy_loss_all)

        return bpr_loss + self.cl_loss*cl_loss + self.cl_loss2*cl_loss2 + self.PI_loss*PI_loss

    def sparse_soft_process(self, x):
        """阻尼函数实现"""
        values = x.values()
        processed_values = torch.log(1 + values + values ** 2 / 2)
        return torch.sparse_coo_tensor(
            x.indices(),
            processed_values,
            size=x.size()
        ).coalesce()

    def extract_non_overlap(self, full_tensor, overlap_tensor):
        """高性能提取非重叠部分的稀疏矩阵（向量化实现）"""
        full_tensor = full_tensor.coalesce()
        overlap_tensor = overlap_tensor.coalesce()

        # 获取索引张量（形状为 [ndim, nnz]）
        full_indices = full_tensor.indices()
        overlap_indices = overlap_tensor.indices()

        # --- 方法1：使用 torch.isin （PyTorch >= 1.10）---
        # 将多维索引合并为单维哈希（适用于COO格式）
        full_hashes = full_indices[0] * (full_tensor.size(1) + 1) + full_indices[1]
        overlap_hashes = overlap_indices[0] * (full_tensor.size(1) + 1) + overlap_indices[1]
        non_overlap_mask = ~torch.isin(full_hashes, overlap_hashes)

        # --- 方法2：自定义哈希比对（更高效，适合超大规模）---
        # 若 torch.isin 仍慢，可改用以下代码（需确保无哈希冲突）：
        # max_size = max(full_tensor.size()) + 1
        # full_hashes = full_indices[0] * max_size + full_indices[1]
        # overlap_hashes = overlap_indices[0] * max_size + overlap_indices[1]
        # non_overlap_mask = ~(full_hashes.unsqueeze(1) == overlap_hashes).any(dim=1)

        # 构建非重叠部分
        return torch.sparse_coo_tensor(
            indices=full_indices[:, non_overlap_mask],
            values=full_tensor.values()[non_overlap_mask],
            size=full_tensor.size(),
            device=full_tensor.device
        ).coalesce()

    def clean_sparse_tensor(self, sparse_tensor):
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        non_zero_mask = values != 0
        return torch.sparse_coo_tensor(
            indices[:, non_zero_mask],
            values[non_zero_mask],
            size=sparse_tensor.size()
        ).coalesce()

    def post_epoch_processing(self, epoch_idx, eval_inter_raw=None):
        if self.relabel_rate > 0:
            if eval_inter_raw is not None and self.eval_inter is None:
                self.eval_inter = self.scipy_coo_to_torch_sparse(eval_inter_raw)
            with torch.no_grad():
                self.old_sparse_C = self.sparse_C
                num_zeros = 0
                for batch_start in range(0, self.n_users, 4096):
                    batch_end = min(batch_start + 4096, self.n_users)
                    batch_users = torch.arange(batch_start, batch_end).to(self.device)
                    ua_embeddings, ia_embeddings = self.forward(train=False)
                    batch_users, batch_items, batch_weights, fuzzy_loss_all = self.cal_PI_loss(batch_users, ua_embeddings, ia_embeddings, is_post_process=True)

                    loss_mat = torch.sparse_coo_tensor(
                        indices=torch.stack([batch_users, batch_items]),
                        values=fuzzy_loss_all.detach(),
                        size=(self.n_users, self.n_items)).to(self.device).coalesce()
                    # weight_mat = torch.sparse_coo_tensor(indices=torch.stack([batch_users, batch_items]),
                    # values=batch_weights.detach(),
                    # size=(self.n_users, self.n_items)).to(self.device).coalesce()
                    # compute damped mean loss and variance loss
                    damped_loss = self.sparse_soft_process(loss_mat).to(self.device)
                    # damped_loss = loss_mat
                    if epoch_idx == 0:
                        self.loss_mean += damped_loss
                        continue

                    self.loss_mean = self.loss_mean.coalesce()
                    self.loss_variance = self.loss_variance.coalesce()

                    # 提取damped_loss的非零索引和值
                    damped_indices = damped_loss.indices()
                    damped_values = damped_loss.values()

                    # --- 更新 loss_mean ---
                    # 提取当前均值中与damped_loss重叠的部分（稀疏掩码）
                    overlap_mean = self.loss_mean.sparse_mask(damped_loss).coalesce()
                    # 计算非重叠部分（原均值 - 重叠部分）
                    non_overlap_mean = self.clean_sparse_tensor((self.loss_mean - overlap_mean).coalesce())
                    # 计算新均值（仅更新重叠部分）
                    new_mean_values = (overlap_mean.values() * epoch_idx + damped_values) / (epoch_idx + 1)
                    # 构建更新后的稀疏均值张量
                    updated_mean = torch.sparse_coo_tensor(damped_indices, new_mean_values, self.loss_mean.size()).coalesce()
                    # 合并非重叠部分与更新后的均值
                    self.loss_mean = (non_overlap_mean + updated_mean).coalesce()

                    # --- 更新 loss_variance ---
                    # 提取当前方差中与damped_loss重叠的部分
                    overlap_var = self.loss_variance.sparse_mask(damped_loss).coalesce()
                    # 计算非重叠部分（原方差 - 重叠部分）
                    non_overlap_var = self.clean_sparse_tensor((self.loss_variance - overlap_var).coalesce())
                    # 提取旧均值用于方差计算
                    old_mean_values = overlap_mean.values()
                    # 计算方差更新项
                    term1 = (epoch_idx / (epoch_idx + 1)) * overlap_var.values()
                    term2 = (epoch_idx / (epoch_idx + 1) ** 2) * (old_mean_values - damped_values) ** 2
                    new_var_values = term1 + term2
                    # 构建更新后的稀疏方差张量
                    updated_var = torch.sparse_coo_tensor(damped_indices, new_var_values, self.loss_variance.size()).coalesce()
                    # 合并非重叠部分与更新后的方差
                    self.loss_variance = (non_overlap_var + updated_var).coalesce()

                    # compute confident lower bound
                    # 确保所有稀疏张量已对齐索引（假设 updated_var 是已更新的稀疏方差）
                    updated_var = updated_var.coalesce()
                    common_indices = updated_var.indices()  # 使用 updated_var 的稀疏索引

                    # --- 计算 numerator（稀疏张量）---
                    s = float(epoch_idx + 1)  # 标量
                    log_term = math.log(2 * s)  # 标量
                    # 稀疏张量运算：直接操作 values()
                    numerator_values = updated_var.values() * (s + (updated_var.values() * log_term) / (s ** 2))
                    numerator = torch.sparse_coo_tensor(
                        indices=common_indices,
                        values=numerator_values,
                        size=updated_var.size(),
                        device = updated_var.device
                    ).coalesce()

                    # --- 计算 denominator（稀疏张量）---
                    # 提取与 updated_var 相同位置的 weight_mat 值
                    weight_masked = self.sparse_C.sparse_mask(updated_var).coalesce()
                    weight_values = weight_masked.values() + 1e-10

                    # 稀疏张量运算：直接操作 values()
                    denominator_values = (torch.tensor(1) / weight_values) - updated_var.values() + 1e-10
                    denominator = torch.sparse_coo_tensor(
                        indices=common_indices,
                        values=denominator_values,
                        size=updated_var.size(),
                        device=updated_var.device
                    ).coalesce()

                    # --- 计算 l_bound（稀疏张量）---
                    # 稀疏张量减法与除法（要求 numerator 和 denominator 索引一致）
                    l_bound = updated_mean.values() - (numerator.values() / denominator.values())

                    l_bound = torch.sparse_coo_tensor(
                        indices=common_indices,
                        values=l_bound,
                        size=updated_mean.size(),
                        device=updated_mean.device
                    ).coalesce()

                    common_indices = l_bound.indices()
                    common_values = l_bound.values()

                    # --- 步骤1：选择高置信下界样本的稀疏坐标 ---
                    relabel_ratio = min(self.relabel_rate, self.relabel_rate*epoch_idx/self.O)
                    sorted_values, sorted_positions = torch.sort(common_values, descending=True)
                    num_select = int(relabel_ratio * len(sorted_positions))
                    selected_positions = sorted_positions[:num_select]
                    selected_indices = common_indices[:, selected_positions]  # shape: (2, num_select)

                    # # 条件2：属于候选样本（假设已通过 selected_indices 构建候选掩码）
                    # candidate_mask = torch.sparse_coo_tensor(
                    #     selected_indices,
                    #     torch.ones(selected_indices.shape[1], device=l_bound.device),
                    #     size=l_bound.size()
                    # ).coalesce()

                    # 提取满足条件的稀疏索引和权重值
                    # 计算哈希值（行 * 列数 + 列）
                    rows, cols = l_bound.shape
                    a_hash = l_bound.indices().T[:, 0] * cols + l_bound.indices().T[:, 1]
                    b_hash = selected_indices.T[:, 0] * cols + selected_indices.T[:, 1]

                    # 找到共同的哈希值（即匹配的坐标对）
                    combined_hashes = torch.cat([a_hash, b_hash])
                    unique_hashes, counts = torch.unique(combined_hashes, return_counts=True)
                    common_hashes = unique_hashes[counts > 1]

                    # 筛选出 A 中的匹配索引
                    mask = torch.isin(a_hash, common_hashes)
                    valid_indices = l_bound.indices()[:, mask]  # 直接使用 loss_mat 的索引
                    valid_values = weight_masked.values()[mask]  # 提取对应位置的权重值

                    # --- 步骤3：构建减法矩阵 ---
                    subtract_matrix = torch.sparse_coo_tensor(
                        indices=valid_indices,
                        values=valid_values,
                        size=self.sparse_C.size(),
                        device=self.sparse_C.device
                    ).coalesce()

                    num_zeros += subtract_matrix.values().numel()
                    self.sparse_C = self.clean_sparse_tensor((self.sparse_C - subtract_matrix).coalesce())
                    
                self.his_F1.append(self.sparse_f1(self.sparse_C, self.eval_inter))
                # 负提升
                if self.his_F1[-1] <= max(self.his_F1[:-1]):
                    self.sparse_C = self.old_sparse_C
                    torch.cuda.empty_cache()
                    return

            torch.cuda.empty_cache()
            return "set zero numbers: {}, F1: {}".format(num_zeros, self.his_F1[-1])

    def sparse_f1(self, A, B):
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

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def _build_large_scale_adjacency_matrices(self):
        """
        为大规模数据集构建邻接矩阵：分段计算并稀疏化
        """
        # 检查文件是否存在
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        image_adj_file = os.path.join(dataset_path, f'image_adj_large_{self.knn_k}.pt')
        text_adj_file = os.path.join(dataset_path, f'text_adj_large_{self.knn_k}.pt')
        
        # 如果文件存在，直接加载
        if os.path.exists(image_adj_file) and os.path.exists(text_adj_file):
            print(f"Loading saved adjacency matrices: {image_adj_file}, {text_adj_file}")
            image_adj = torch.load(image_adj_file, map_location=self.device)
            text_adj = torch.load(text_adj_file, map_location=self.device)
            return image_adj, text_adj
        
        print("Starting chunked computation of large-scale adjacency matrices...")
        
        # 分段计算图像邻接矩阵
        image_adj = self._build_chunked_similarity_matrix(
            self.image_embedding.weight.detach(), 
            'image'
        )
        
        # 分段计算文本邻接矩阵
        text_adj = self._build_chunked_similarity_matrix(
            self.text_embedding.weight.detach(), 
            'text'
        )
        
        # 保存到文件
        os.makedirs(dataset_path, exist_ok=True)
        torch.save(image_adj, image_adj_file)
        torch.save(text_adj, text_adj_file)
        print(f"Adjacency matrices saved to: {image_adj_file}, {text_adj_file}")
        
        return image_adj, text_adj

    def _build_chunked_similarity_matrix(self, embeddings, modality_name):
        """
        分段计算相似度矩阵并稀疏化
        """
        n_items = embeddings.shape[0]
        chunk_size = 10000  # 每段处理10000个物品
        device = embeddings.device
        
        print(f"Starting chunked computation of {modality_name} similarity matrix, total items: {n_items}")
        
        # 分段计算相似度矩阵并直接进行稀疏化
        all_sparse_blocks = []
        
        for i in range(0, n_items, chunk_size):
            end_i = min(i + chunk_size, n_items)
            print(f"Processing {modality_name} chunk {i//chunk_size + 1}/{(n_items-1)//chunk_size + 1}: items {i}-{end_i-1}")
            
            # 获取当前块的嵌入
            chunk_embeddings = embeddings[i:end_i]
            
            # 计算当前块与所有物品的相似度
            chunk_sim = self._compute_chunk_similarity(chunk_embeddings, embeddings)
            
            # 对当前块直接进行top-k稀疏化
            sparse_chunk = self._sparsify_chunk_similarity(chunk_sim, i, end_i, n_items)
            all_sparse_blocks.append(sparse_chunk)
            
            # 清理内存
            del chunk_embeddings, chunk_sim
            torch.cuda.empty_cache()
        
        # 合并所有稀疏块
        final_adj = self._merge_sparse_blocks(all_sparse_blocks, n_items)
        
        return final_adj

    def _compute_chunk_similarity(self, chunk_embeddings, all_embeddings):
        """
        计算当前块与所有物品的相似度
        """
        # 归一化嵌入
        chunk_norm = chunk_embeddings.div(torch.norm(chunk_embeddings, p=2, dim=-1, keepdim=True))
        chunk_norm[torch.isnan(chunk_norm)] = 0.
        all_norm = all_embeddings.div(torch.norm(all_embeddings, p=2, dim=-1, keepdim=True))
        all_norm[torch.isnan(all_norm)] = 0.
        # 计算余弦相似度
        similarity = torch.mm(chunk_norm, all_norm.transpose(0, 1))
        
        return similarity
    
    def _sparsify_chunk_similarity(self, chunk_sim, start_idx, end_idx, n_items):
        """
        对当前相似度块进行top-k稀疏化
        
        参数:
            chunk_sim: 当前块的相似度矩阵
            start_idx: 起始行索引
            end_idx: 结束行索引
            n_items: 总物品数
            
        返回:
            稀疏化的相似度块
        """
        device = chunk_sim.device
        chunk_rows = chunk_sim.shape[0]
        
        # 对每一行进行top-k选择
        k = min(self.config['knn_k'], n_items)  # 使用配置中的knn_k参数
        
        # 使用torch.topk进行向量化top-k选择
        topk_values, topk_indices = torch.topk(chunk_sim, k=k, dim=1)
        
        # 只保留正相似度
        positive_mask = topk_values > 0
        
        if positive_mask.any():
            # 获取有效值的行和列索引
            valid_rows = torch.arange(chunk_rows, device=device).unsqueeze(1).expand(-1, k)[positive_mask]
            valid_cols = topk_indices[positive_mask]
            valid_values = topk_values[positive_mask]
            
            # 调整行索引到全局位置
            global_rows = valid_rows + start_idx
            
            # 创建稀疏张量
            sparse_chunk = torch.sparse_coo_tensor(
                torch.stack([global_rows, valid_cols]),
                valid_values,
                (n_items, n_items),
                device=device
            )
        else:
            # 如果没有有效值，创建空稀疏张量
            sparse_chunk = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device),
                (n_items, n_items),
                device=device
            )
        
        return sparse_chunk.coalesce()

    def _merge_sparse_blocks(self, sparse_blocks, n_items):
        """
        合并所有稀疏块
        """
        if not sparse_blocks:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty((0,)),
                size=(n_items, n_items)
            )
        
        # 收集所有索引和值
        all_indices = []
        all_values = []
        
        for block in sparse_blocks:
            if block._nnz() > 0:
                all_indices.append(block.indices())
                all_values.append(block.values())
        
        if not all_indices:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty((0,)),
                size=(n_items, n_items)
            )
        
        # 合并所有索引和值
        merged_indices = torch.cat(all_indices, dim=1)
        merged_values = torch.cat(all_values)
        
        # 创建合并后的稀疏矩阵
        merged_sparse = torch.sparse_coo_tensor(
            indices=merged_indices,
            values=merged_values,
            size=(n_items, n_items),
            device=merged_indices.device
        )
        
        # 应用对称归一化
        merged_sparse = self._apply_symmetric_normalization(merged_sparse)
        
        return merged_sparse

    def _apply_symmetric_normalization(self, sparse_matrix):
        """
        对稀疏矩阵应用对称归一化
        """
        # 计算度矩阵
        degrees = torch.sparse.sum(sparse_matrix, dim=1).to_dense()
        degrees = torch.pow(degrees, -0.5)
        degrees[torch.isinf(degrees)] = 0.0
        
        # 创建度矩阵的稀疏表示
        n = sparse_matrix.size(0)
        degree_indices = torch.arange(n, device=sparse_matrix.device).unsqueeze(0).repeat(2, 1)
        degree_values = degrees
        
        degree_matrix = torch.sparse_coo_tensor(
            indices=degree_indices,
            values=degree_values,
            size=(n, n),
            device=sparse_matrix.device
        )
        
        # 应用对称归一化: D^(-1/2) * A * D^(-1/2)
        normalized = torch.sparse.mm(degree_matrix, sparse_matrix)
        normalized = torch.sparse.mm(normalized, degree_matrix)
        
        return normalized
