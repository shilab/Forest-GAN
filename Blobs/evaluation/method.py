# 调用时，直接查看函数 evaluation_precision_recall()。

import torch
import random
import numpy as np
                                
def block_EuclideanDistance(U, V):
    U = torch.tensor(U)
    V = torch.tensor(V)
    pow_u = U.pow(2)
    pow_v = V.pow(2)
    norm_u = torch.sum(pow_u, dim=1).reshape((-1, 1))
    norm_v = torch.sum(pow_v, dim=1).reshape((1, -1))
    u_v = torch.matmul(U, V.t())
    distance_block = norm_u - 2 * u_v + norm_v
    #print("-----------------##------------------")
    #print(distance_block)
    return distance_block

# EuclideanDistance的功能是计算两个特征矩阵，相互之间的距离
# 担心特征矩阵的维度过大，内存容不下，
# 在实现上，对特侦进行分块，特征块与特征块之间的距离，调用block_EuclideanDistance来完成。
def EuclideanDistance(real_features, gene_features, **args):
    topk = args["topk"]
    seq = np.arange(topk+1)
    batch_size = args["batch_size"]
    num_real = len(real_features)
    num_gene = len(gene_features)
    
    # 计算real_features自己元素间的距离
    distance_self = np.zeros([num_real, num_real], dtype=np.float64)
    distance_batch = np.zeros([batch_size, num_real], dtype=np.float64)
    
    for begin1 in range(0, num_real, batch_size):
        end1 = min(begin1 + batch_size, num_real)
        row_batch = real_features[begin1:end1]
        
        for begin2 in range(0, num_real, batch_size):
            end2 = min(begin2 + batch_size, num_real)
            col_batch = real_features[begin2:end2]
            
            distance_batch[0:end1-begin1, begin2:end2] = block_EuclideanDistance(row_batch, col_batch)
        distance_self[begin1:end1, :] = distance_batch[0:end1-begin1, :]
        
    distance_self_topk = np.partition(distance_self, seq, axis=1)[:, topk]

    # 计算real_features与gene_feature之间的距离关系
    distance = np.zeros([num_gene, num_real], dtype=np.float64)
    nearst_indices = np.zeros([num_gene], dtype=np.float64)
    distance_batch = np.zeros([batch_size, num_real], dtype=np.float64)
    
    for begin1 in range(0, num_gene, batch_size):
        end1 = min(begin1 + batch_size, num_gene)
        row_batch = gene_features[begin1:end1]
        
        for begin2 in range(0, num_real, batch_size):
            end2 = min(begin2 + batch_size, num_real)
            col_batch = real_features[begin2:end2]
            distance_batch[0:end1-begin1, begin2:end2] = block_EuclideanDistance(row_batch, col_batch)
        
        distance[begin1:end1, :] = distance_batch[0:end1-begin1, :]
    
    # 对于real_feature，只返回每个特征的第K近的邻居的距离，即distance_self_topk
    # real_features与gene_feature之间的距离关系，全部都返回
    return distance_self_topk, distance

# precision: the percentage of generation samples that fall into the real samples
# recall: the percentage of real samples that fall into the generation samples
def evaluate(ref_self, evl_ref_distance, evl_self,lambd=2):
    num_ref, num_evl = np.array(ref_self).shape[0], np.array(evl_ref_distance).shape[0]
    
    lower = 1.0/lambd
    upper = lambd*1.0
    count  = 0
    
    for index in range(num_evl):
        
        evl_ref_array = evl_ref_distance[index]
        evl_ref_list = evl_ref_array < ref_self
        
        d_k_evl = evl_self[index]
        upper = d_k_evl * lambd
        lower = d_k_evl / lambd
        
        upper_list = ref_self < upper
        lower_list = ref_self > lower
        
        result_list = np.logical_and(upper_list, lower_list)
        result_list = np.logical_and(result_list, evl_ref_list)
        if np.sum(result_list) > 0:
            count += 1
                
    return (count * 1.0) / num_evl
                
# evaluation_precision_recall的功能是：生成数据的precision和recall
# 其中，真实数据和生成数据都是保存在".npy"文件当中的，在".npy"文件中，行数表示特征元素的个数，列数表示一个特征元素的维度
# 超参数解析：
# real_filename：保存真实数据的文件名，“.npy”文件
# fake_filename: 保存生成数据的文件名，“.npy”文件
# lambd:定义相对密度的范围；相对密度的范围是[1/lambd, lambd]
# topk: 定义第k个最近邻居，
# batch_size: 为了节省内存，在计算两个数据集之间元素与元素的距离时，我们采用分块计算的方法，batch_size指定每一块的大小（每一块的特征元素个数）
def evaluation_precision_recall(real_filename, fake_filename, lambd=3, topk= 5, batch_size=10000):
    real_sample = np.load(real_filename).astype(np.float64)
    fake_sample = np.load(fake_filename).astype(np.float64)

    real_sample = torch.tensor(real_sample)
    gene_sample = torch.tensor(fake_sample)

    # block_EuclideanDistance(real_sample, gene_sample)
    real_self, gene_real_distance = EuclideanDistance(real_sample, gene_sample, topk=topk, batch_size=batch_size)
    gene_self, real_gene_distance = EuclideanDistance(gene_sample, real_sample, topk=topk, batch_size=batch_size)
    
    # evaluation
    precision = evaluate(real_self, gene_real_distance, gene_self, lambd=lambd)
    recall = evaluate(gene_self, real_gene_distance, real_self, lambd=lambd)
    return precision, recall