"""Resolution bucketing and batch samplers.

Extracted from the vendored ``data/data.py``, which also held a second,
superseded ``UnifiedDataset`` implementation. The live dataset is
``data/tar_parquet_data_v2.py``; it imports the five classes below and nothing
else from here, so the unreachable half was dropped during vendoring:
``UnifiedDataset``/``create_unified_dataset`` (duplicates keyed on a different
config schema), ``get_dataset_configs`` (hardcoded paths from another cluster),
``open_image``, ``process_task_mapping_batch``, ``generate_resolutions``, and
the ``data/dataloading.py`` module they depended on.

The classes themselves are unchanged from upstream DialogEdit_qwen.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class ResolutionBucket:
    """
    Resolution bucket for grouping images by resolution, supports millions of data records
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.indices = np.array([], dtype=np.int32)
        self.capacity = 1000 
        self.size = 0
        self._buffer = np.zeros(self.capacity, dtype=np.int32)
    
    def add(self, idx: int):
        if self.size >= self.capacity:
            self._expand_capacity()
        
        self._buffer[self.size] = idx
        self.size += 1
    
    def _expand_capacity(self):
        new_capacity = self.capacity * 2
        new_buffer = np.zeros(new_capacity, dtype=np.int32)
        new_buffer[:self.size] = self._buffer[:self.size]
        self._buffer = new_buffer
        self.capacity = new_capacity
    
    def finalize(self):
        if self._buffer is not None:
            self.indices = self._buffer[:self.size].copy()
            self._buffer = None
            self.capacity = 0
    
    def __len__(self):
        return self.size

class ResolutionBucketer:
    def __init__(self, common_resolutions: Optional[List[Tuple[int, int]]] = None):
        if common_resolutions is None:
            self.common_resolutions = [
                (768, 1152),
                (768, 1344),
                (864, 1152),
                (1024, 1024),
                (1152, 768),
                (1152, 864),
                (1344, 768),
                (1360, 768)
            ]
            print(f"Resolution bucket: {self.common_resolutions}")
        else:
            self.common_resolutions = common_resolutions
        
        self.buckets = {res: ResolutionBucket(res[0], res[1]) for res in self.common_resolutions}
        self.aspect_ratios = {res: res[0] / res[1] for res in self.common_resolutions}
        self.resolution_cache = {}
        self.current_target_resolution = None
    
    def find_best_resolution(self, width: int, height: int) -> Tuple[int, int]:
        cache_key = (width, height)
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]
        
        aspect_ratio = width / height if height != 0 else float('inf')
        
        best_res = None
        min_diff = float('inf')
        
        for res, res_aspect_ratio in self.aspect_ratios.items():
            diff = abs(aspect_ratio - res_aspect_ratio)
            
            if diff < min_diff:
                min_diff = diff
                best_res = res
        
        self.resolution_cache[cache_key] = best_res
        return best_res
    
    def add_image(self, idx: int, width: int, height: int):
        best_res = self.find_best_resolution(width, height)
        self.buckets[best_res].add(idx)

    # def add_image(self, idx: int, width: int, height: int, task: str):
    #     best_res = self.find_best_resolution(width, height)
    #     task_tag = 't2i' if task.lower() == 't2i' else 'edit'
    #     self.buckets[task_tag][best_res].add(idx)
    
    def finalize(self):
        for bucket in self.buckets.values():
            bucket.finalize()
    
    def get_bucket_indices(self) -> Dict[Tuple[int, int], np.ndarray]:
        self.finalize()
        return {res: bucket.indices for res, bucket in self.buckets.items()}
    

class TaskResolutionBucketer:
    def __init__(self, common_resolutions: Optional[List[Tuple[int, int]]] = None, common_tasks=['t2i', 'edit']):
        if common_resolutions is None:
            self.common_resolutions = [
                (768, 1152),
                (768, 1344),
                (864, 1152),
                (1024, 1024),
                (1152, 768),
                (1152, 864),
                (1344, 768),
                (1360, 768)
            ]
            print(f"Resolution bucket: {self.common_resolutions}")
        else:
            self.common_resolutions = common_resolutions

        self.tasks = common_tasks
        self.buckets = {}
        for common_task in common_tasks:
            task_buckets = {(res[0], res[1], common_task.lower()): ResolutionBucket(res[0], res[1]) for res in self.common_resolutions}
            self.buckets.update(task_buckets)
        
        self.aspect_ratios = {res: res[0] / res[1] for res in self.common_resolutions}
        self.resolution_cache = {}
        self.current_target_resolution = None

    def find_best_resolution(self, width: int, height: int) -> Tuple[int, int]:
        cache_key = (width, height)
        if cache_key in self.resolution_cache:
            return self.resolution_cache[cache_key]

        aspect_ratio = width / height if height != 0 else float('inf')

        best_res = None
        min_diff = float('inf')

        for res, res_aspect_ratio in self.aspect_ratios.items():
            diff = abs(aspect_ratio - res_aspect_ratio)

            if diff < min_diff:
                min_diff = diff
                best_res = res

        self.resolution_cache[cache_key] = best_res
        return best_res

    def find_best_task(self, task_name: str) -> str:
        best_task_tag = 'edit' # by default
        for task_tag in self.tasks:
            if task_name.lower() == task_tag.lower():
                best_task_tag = task_tag.lower()
                break
        # print(f"best_task_tag:{best_task_tag}")
        return best_task_tag

    def add_image(self, idx: int, width: int, height: int, task: str):
        best_res = self.find_best_resolution(width, height)
        best_task = self.find_best_task(task)

        self.buckets[(best_res[0], best_res[1], best_task)].add(idx)
    
    def finalize(self):
        for bucket in self.buckets.values():
            bucket.finalize()
    
    def get_bucket_indices(self) -> Dict[Tuple[int, int, str], np.ndarray]:
        self.finalize()
        return {res_task: bucket.indices for res_task, bucket in self.buckets.items()}
    

    
class ResolutionBucketSampler_distributed(Sampler):
    """
    分辨率分类采样器，确保每个批次中的图像具有相同的分辨率。
    支持分布式训练。
    """
    def __init__(self, resolution_bucketer, batch_size, shuffle=True, seed=42):
        """
        初始化分辨率分类采样器

        Args:
            resolution_bucketer: 分辨率分类器
            batch_size: 批次大小
            shuffle: 是否打乱数据
            seed: 随机种子，用于保证分布式环境下 shuffle 的一致性
        """
        # Try to get rank and world_size from torch.distributed if initialized
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            # Fallback to environment variables
            # These are typically set by the distributed launcher (e.g., torchrun, SLURM, or PyTorch Lightning)
            rank_env = os.environ.get("RANK")
            world_size_env = os.environ.get("WORLD_SIZE")
            
            if rank_env is not None and world_size_env is not None:
                try:
                    self.rank = int(rank_env)
                    self.num_replicas = int(world_size_env)
                except ValueError:
                    # Optional: Add a warning if parsing fails
                    # warnings.warn("Sampler: Could not parse RANK/WORLD_SIZE from env. Defaulting to non-distributed.")
                    self.num_replicas = 1
                    self.rank = 0
            else:
                # Optional: Add a warning if env vars are not set
                # warnings.warn("Sampler: RANK or WORLD_SIZE env var not set. Defaulting to non-distributed.")
                self.num_replicas = 1
                self.rank = 0

        self.resolution_bucketer = resolution_bucketer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0 # 用于 shuffle 的 epoch 计数

        # 获取所有分辨率桶的索引
        self.bucket_indices = resolution_bucketer.get_bucket_indices()

        # 计算全局总批次数 (在 __iter__ 中计算更准确，因为 drop_last)
        self.num_batches_per_replica = -1 # Will be calculated in __iter__

    def __iter__(self):
        # 使用 epoch 和 seed 确保不同 rank 上的 shuffle 顺序一致
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_batches = []
        for res, indices in self.bucket_indices.items():
            # 使用生成器进行 shuffle
            if self.shuffle:
                indices = list(indices)
                perm = torch.randperm(len(indices), generator=g).tolist()
                shuffled_indices = [indices[i] for i in perm]
            else:
                shuffled_indices = list(indices)

            # 创建批次
            for i in range(0, len(shuffled_indices), self.batch_size):
                batch = shuffled_indices[i:i+self.batch_size]
                if len(batch) == self.batch_size: 
                    batch = [int(idx) for idx in batch]
                    all_batches.append(batch)

        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        batches_for_rank = all_batches[self.rank::self.num_replicas]
        self.num_batches_per_replica = len(batches_for_rank)
        return iter(batches_for_rank)

    def __len__(self):
        if self.num_batches_per_replica == -1:
             # Estimate length before first iteration (might be slightly inaccurate due to drop_last)
             total_samples = sum(len(indices) for indices in self.bucket_indices.values())
             total_batches = total_samples // self.batch_size
             # Approximate per-replica length
             return total_batches // self.num_replicas
        return self.num_batches_per_replica

    def set_epoch(self, epoch: int) -> None:
         """
         Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
         use a different random ordering for each epoch. Otherwise, the next iteration of this
         sampler will yield the same ordering.

         Args:
             epoch (int): Epoch number.
         """
         self.epoch = epoch

class ResolutionBucketSampler(Sampler):
    """
    分辨率分类采样器，确保每个批次中的图像具有相同的分辨率
    """
    def __init__(self, resolution_bucketer, batch_size, shuffle=True, accelerator=None, seed=42, global_batch_size=-1):
        """
        初始化分辨率分类采样器
        
        Args:
            resolution_bucketer: 分辨率分类器
            batch_size: 批次大小
            shuffle: 是否打乱数据
            accelerator: 加速器对象
        """
        self.resolution_bucketer = resolution_bucketer

        if global_batch_size > batch_size:
            assert global_batch_size % batch_size == 0
            
        self.global_batch_size = global_batch_size
        self.batch_size = batch_size
        
        self.shuffle = shuffle
        self.accelerator = accelerator
        
        # 获取所有分辨率桶的索引
        self.bucket_indices = resolution_bucketer.get_bucket_indices()
        
        # 计算每个桶的批次数
        self.bucket_batches = {}
        for res, indices in self.bucket_indices.items():
            self.bucket_batches[res] = len(indices) // self.batch_size

        num_samples = 0
        for res, indices in self.bucket_indices.items():
            num_samples += len(indices)
                
        self.num_batches = sum(self.bucket_batches.values())
        self.generator = None
        self.seed = seed
        self.epoch = 0 # 用于 shuffle 的 epoch 计数

    
    def __iter__(self):
        # 为每个桶创建批次
        all_batches = []
        uneven_batches = []
        
        if self.generator is None:
            # seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        else:
            generator = self.generator

        for res, indices in self.bucket_indices.items():
            indices = list(indices)

            # 打乱索引
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]

            if self.global_batch_size > self.batch_size:
                batch_size = self.global_batch_size
            else:
                batch_size = self.batch_size
            
            # 创建批次
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]

                if len(batch) == batch_size:  # 只使用完整的批次
                    # 将batch中的numpy.int32转换为Python int
                    batch = [int(idx) for idx in batch]
                    # 将批次添加到列表中
                    all_batches.append(batch)
                else:
                    uneven_batches.append(batch)
                    
        # 打乱所有批次
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=generator).tolist()
            all_batches = [all_batches[i] for i in perm]
            if len(uneven_batches) > 0:
                perm = torch.randperm(len(uneven_batches), generator=generator).tolist()
                uneven_batches = [uneven_batches[i] for i in perm]
            
            
        if self.global_batch_size > self.batch_size:
            all_batches_per_gpu = []
            for batch in all_batches:
                for i in range(0, len(batch), self.batch_size):
                    batch_per_gpu = batch[i:i+self.batch_size]

                    if len(batch_per_gpu) == self.batch_size:  # 只使用完整的批次
                        # 将batch中的numpy.int32转换为Python int
                        batch_per_gpu = [int(idx) for idx in batch_per_gpu]
                        # 将批次添加到列表中
                        all_batches_per_gpu.append(batch_per_gpu)
            
            all_batches = all_batches_per_gpu
            for batch in uneven_batches:
                for i in range(0, len(batch), self.batch_size):
                    batch_per_gpu = batch[i:i+self.batch_size]

                    if len(batch_per_gpu) == self.batch_size:  # 只使用完整的批次
                        # 将batch中的numpy.int32转换为Python int
                        batch_per_gpu = [int(idx) for idx in batch_per_gpu]
                        # 将批次添加到列表中
                        all_batches.append(batch_per_gpu)
        
        # print('total batches', len(all_batches))
        # print('number_of_batches', self.num_batches)
        
        # 返回迭代器
        return iter(all_batches)
    
    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch: int) -> None:
         """
         Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
         use a different random ordering for each epoch. Otherwise, the next iteration of this
         sampler will yield the same ordering.

         Args:
             epoch (int): Epoch number.
         """
         self.epoch = epoch

