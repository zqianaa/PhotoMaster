import os
import sys
import ast
import json
import random
import glob
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Iterator
from pathlib import Path
import re
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
import tarfile
from io import BytesIO
import zipfile
import io
import copy
import time
import torchvision.transforms as T

# 使用线程池并行处理分辨率分桶
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
from accelerate.logging import get_logger

from tqdm import tqdm
from data.resolution_bucketing import (
    ResolutionBucket,
    ResolutionBucketer,
    TaskResolutionBucketer,
    ResolutionBucketSampler_distributed,
    ResolutionBucketSampler
)
from data.indexed_tarfile_reader import IndexedTarReader
from multiprocessing import Pool, cpu_count
from functools import partial

PREFERRED_KONTEXT_RESOLUTIONS = [
    (1360, 768),   # 16:9
    (1248, 832),   # 3:2
    (1152, 864),   # 4:3
    (1024, 1024),  # 1:1
    (864, 1152),   # 3:4
    (832, 1248),   # 2:3
    (768, 1360),   # 9:16
]

PREFERRED_QWEN_EDIT_2511_NATIVE_RESOLUTIONS = [
    (1664, 928),   # 16:9
    (1584, 1056),  # 3:2
    (1472, 1104),  # 4:3
    (1328, 1328),  # 1:1
    (1104, 1472),  # 3:4
    (1056, 1584),  # 2:3
    (928, 1664),   # 9:16
]
# PREFERRED_KONTEXT_RESOLUTIONS = [
#     (720, 1456),
#     (752, 1392),
#     (784, 1312),
#     (832, 1248),
#     (880, 1184),
#     (944, 1104),
#     (1024, 1024),
#     (1104, 944),
#     (1184, 880),
#     (1248, 832),
#     (1312, 784), 
#     (1392, 752),
#     (1456, 720),
# ]

# BUCKET_PROBS = [0.05, 0.012, 0.006, 0.029, 0.025, 0.067, 0.499, 0.064, 0.025, 0.075, 0.036, 0.045, 0.067]

# PREFERRED_KONTEXT_RESOLUTIONS_512 = [
#      (336, 784),
#      (336, 752),
#      (352, 720),
#      (368, 688),
#      (400, 656),
#      (416, 624),
#      (432, 592),
#      (464, 544),
#      (512, 512),
#      (544, 464),
#      (592, 432),
#      (624, 416),
#      (656, 400),
#      (688, 368),
#      (720, 352),
#      (752, 336),
#      (784, 336)
# ]

PREFERRED_KONTEXT_RESOLUTIONS_512 = [
    (512, 512), 
    (416, 624), 
    (624, 416), 
    (432, 592), 
    (592, 432), 
    (448, 576), 
    (576, 448), 
    (384, 672), 
    (672, 384), 
    (768, 336)
]

QWEN_EDIT_RESOLUTION_PRESETS = {
    "qwen_edit_1024": PREFERRED_KONTEXT_RESOLUTIONS,
    "qwen_edit_2511_native": PREFERRED_QWEN_EDIT_2511_NATIVE_RESOLUTIONS,
    "qwen_edit_512": PREFERRED_KONTEXT_RESOLUTIONS_512,
}
DEFAULT_QWEN_EDIT_RESOLUTION_PRESET = "qwen_edit_1024"


def resolve_qwen_edit_resolution_preset(resolution_preset=None, size=1024):
    """Return a named Qwen Edit resolution bucket list."""
    if resolution_preset is None:
        resolution_preset = DEFAULT_QWEN_EDIT_RESOLUTION_PRESET if size == 1024 else "qwen_edit_512"
    resolution_preset = str(resolution_preset).strip() or DEFAULT_QWEN_EDIT_RESOLUTION_PRESET
    if resolution_preset not in QWEN_EDIT_RESOLUTION_PRESETS:
        valid = ", ".join(sorted(QWEN_EDIT_RESOLUTION_PRESETS))
        raise ValueError(f"Unknown Qwen Edit resolution preset: {resolution_preset}. Valid presets: {valid}")
    return resolution_preset, QWEN_EDIT_RESOLUTION_PRESETS[resolution_preset]


logger = get_logger(__name__)

import math
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


def is_row_valid(row):
    task = row.get('task')
    parquet_file = row.get("parquet_file")
    
    if "pangu_ct" in parquet_file:
        limit_len = 2500
    elif "banana-t2i-chinese" in parquet_file or "banana-haoran-T2I-zh" in parquet_file:
        limit_len = 500
    elif "banana-t2i-english" in parquet_file or "banana-haoran-T2I-EN" in parquet_file:
        limit_len = 1250
    elif task == "t2i":
        limit_len = 1500
    elif "filtered" in parquet_file or "_gold_" in parquet_file or "pico_400K_v2" in parquet_file or "sfs_mtp_hmz" in parquet_file:
        limit_len = 800
    else:
        limit_len = None

    check_keys = ["instruction_ch", "instruction_en"]

    for key in check_keys:
        val = row.get(key)
        if pd.isna(val):
            continue

        s = str(val)

        if '```' in s or "{}" in s or '"actions"' in s or '"action"' in s or "the the the the" in s:
            # print(s)
            return False

        if task != "t2i" and ("I am" in s):
            return False

        if limit_len is not None and len(s) > limit_len:
            # print(f"* Filtered * [{parquet_file}] ({len(s)}/{limit_len}):\n{s}")
            return False

    instruction_ch = row.get("instruction_ch")
    instruction_en = row.get("instruction_en")

    if (instruction_ch is None or pd.isna(instruction_ch) or len(instruction_ch) == 0) and (instruction_en is None or pd.isna(instruction_en) or len(instruction_en) == 0):
        return False

    return True


def format_instruction(x):
    if x is None or pd.isna(x):
        return ''
    if x.startswith('[') and x.endswith(']'):
        instruction = ast.literal_eval(x)
        if isinstance(instruction, list) and len(instruction) > 0:
            instruction = random.choice(instruction)
    else:
        instruction = x
    if len(instruction) == 0:
        instruction = ''
    return instruction


# WORKER FUNCTION - Must be at module level for multiprocessing
def _prepare_dataset_worker(pd_file: str, zip_file_path: str, ratio: float = 1, 
                           use_index_tar_cache: bool = True, exclude_task: list = None):
    """
    Worker function to prepare a single dataset from parquet file.
    This must be a module-level function to be picklable for multiprocessing.
    
    Args:
        pd_file: Path to parquet file
        zip_file_path: Path to corresponding zip/tar file
        ratio: Sampling ratio
        use_index_tar_cache: Whether to use indexed tar reader
        exclude_task: List of task values to exclude from the dataset
        
    Returns:
        HuggingFace Dataset or None if failed
    """
    try:
        # Load only the required columns from parquet file for better performance
        # required_columns = ["original_img_path", "edited_img_path", "width", "height", "task", "instruction_ch", "instruction_en", "additional_info"]
        required_columns = ["original_img_path", "edited_img_path", "width", "height", "task", "instruction_ch", "instruction_en"]  # TODO
        dataset = pd.read_parquet(pd_file, columns=required_columns)

        # TODO: filter
        original_dataset_len = len(dataset)
        dataset.loc[:, 'parquet_file'] = pd_file
        mask = dataset.apply(is_row_valid, axis=1)
        dataset = dataset[mask]
        # print(f"[{pd_file}] {len(dataset)}/{original_dataset_len}")
        
        # Exclude samples based on task values if specified
        if exclude_task is not None and len(exclude_task) > 0:
            # Convert both task values and exclude_task to lowercase for case-insensitive comparison
            exclude_task_lower = [task.lower() for task in exclude_task]
            dataset = dataset[~dataset['task'].str.lower().isin(exclude_task_lower)]
            if len(dataset) == 0:
                logger.info(f"All samples from {pd_file} were excluded based on task filter")
                return None

        # Apply sampling if needed
        if ratio != 1:
            # modify by jqxu
            n_rows = len(dataset)
            target_rows = int(n_rows * ratio)
            
            int_part = int(ratio)
            remainder_rows = target_rows - (n_rows * int_part)
            
            parts = []
            
            if int_part > 0:
                parts.append(pd.concat([dataset] * int_part, ignore_index=True))
            
            if remainder_rows > 0:
                parts.append(dataset.sample(n=remainder_rows, replace=False, random_state=42))
            
            if not parts:
                return None

            dataset = pd.concat(parts, ignore_index=True)
            # print(f"n_rows: {n_rows}, ratio: {ratio}, dataset2: {len(dataset2)}")
        
        # Select required columns
        dataset2 = dataset[["original_img_path", "edited_img_path", "width", "height", "task"]].copy()
        
        # Convert width and height to integers
        dataset2['width'] = dataset2['width'].astype(int)
        dataset2['height'] = dataset2['height'].astype(int)
        
        # Apply instruction extraction
        def get_instruction_type(row):
            """Extract instruction from row with probability sampling"""
            import random
            
            # Check additional_info
            if 'additional_info' in row.index:
                additional_info = row['additional_info']
                if additional_info is not None and isinstance(additional_info, dict):
                    long_instruction_en = additional_info.get('long_instruction_en')
                    long_instruction_ch = additional_info.get('long_instruction_ch')
                    
                    # Case 1: Both long instructions exist - sample with 0.5 probability
                    if long_instruction_en is not None and long_instruction_ch is not None and not pd.isna(long_instruction_en) and not pd.isna(long_instruction_ch):
                    # if long_instruction_en is not None and long_instruction_ch is not None:
                        if random.random() < 0.5:
                            return long_instruction_en, 'long_instruction_en'
                        else:
                            return long_instruction_ch, 'long_instruction_ch'
                    
                    # Case 2: Only one long instruction exists - use it directly
                    if long_instruction_en is not None and not pd.isna(long_instruction_en):
                    # if long_instruction_en is not None:
                        return long_instruction_en, 'long_instruction_en'
                    if long_instruction_ch is not None and not pd.isna(long_instruction_ch):
                    # if long_instruction_ch is not None:
                        return long_instruction_ch, 'long_instruction_ch'
            
            # Fallback to instruction_en and instruction_ch
            instruction_en = row.get('instruction_en')
            instruction_ch = row.get('instruction_ch')
            
            # Both exist and are non-None - sample with 0.5 probability
            if instruction_en is not None and instruction_ch is not None and not pd.isna(instruction_en) and not pd.isna(instruction_ch) and len(instruction_en) > 0 and len(instruction_ch) > 0:
                if random.random() < 0.5:
                    return instruction_en, 'instruction_en'
                else:
                    return instruction_ch, 'instruction_ch'
            
            # Only one exists - use it
            if instruction_en is not None and not pd.isna(instruction_en) and len(instruction_en) > 0:
                return instruction_en, 'instruction_en'
            if instruction_ch is not None and not pd.isna(instruction_ch) and len(instruction_ch) > 0:
                return instruction_ch, 'instruction_ch'

            print(f"\n* no instruction: {row}\n")
            
            return None, 'none'

        # TODO: filter instructions and choose if list
        dataset['instruction_en'] = dataset['instruction_en'].apply(format_instruction)
        dataset['instruction_ch'] = dataset['instruction_ch'].apply(format_instruction)
        
        dataset2[['instruction', 'instruction_type']] = dataset.apply(
            get_instruction_type, axis=1, result_type='expand'
        )
        
        # Add zip_file column
        dataset2.loc[:, 'zip_file'] = zip_file_path

        # Convert to HuggingFace Dataset
        dataset2 = Dataset.from_pandas(dataset2, preserve_index=False)
        
        return dataset2
        
    except Exception as e:
        logger.error(f"Error loading {pd_file}: {str(e)}")
        import traceback
        # traceback.print_exc()
        return None



def _load_single_dataset_config(dataset_config: Dict, num_workers: int, 
                                use_index_tar_cache: bool) -> Tuple[List, Dict]:
    """
    Worker function to load a single dataset configuration.
    This must be a module-level function to be picklable for multiprocessing.
    
    Args:
        dataset_config: Dataset configuration dictionary
        num_workers: Number of workers for internal parallelization
        use_index_tar_cache: Whether to use indexed tar reader
        
    Returns:
        Tuple of (list of datasets, dict with metadata including tar_files to cache)
    """
    dataset_name = dataset_config['dataset_name']
    image_dir = dataset_config['image_dir']
    annotation_dir = dataset_config['annotation_dir']
    ratio = dataset_config.get('ratio', 1)
    exclude_task = dataset_config.get('exclude_task', None)
    
    parquet_list = glob.glob(os.path.join(annotation_dir, '*.parquet'))
    
    if not parquet_list:
        logger.warning(f"No parquet files found in {annotation_dir}")
        return [], {'dataset_name': dataset_name, 'tar_files': []}
    
    tic = time.time()
    
    # Prepare file pairs (parquet, image_file)
    file_pairs = []
    tar_files = []
    for pp in parquet_list:
        zip_file_path = os.path.join(image_dir, os.path.basename(pp).replace('.parquet', '.zip'))
        tar_file_path = os.path.join(image_dir, os.path.basename(pp).replace('.parquet', '.tar'))
        
        if not os.path.exists(zip_file_path) and not os.path.exists(tar_file_path):
            logger.warning(f'{zip_file_path} or {tar_file_path} not exist')
            continue
        image_file_path = zip_file_path if os.path.exists(zip_file_path) else tar_file_path
        file_pairs.append((pp, image_file_path, ratio))
        
        if image_file_path.endswith('.tar'):
            tar_files.append(image_file_path)
    
    # Check if we're in a daemon process
    current_process = mp.current_process()
    is_daemon = current_process.daemon
    
    # STRATEGY SELECTION based on number of files
    # Disable multiprocessing if we're in a daemon process
    if len(file_pairs) <= 5 or is_daemon:
        # For small number of files, sequential is faster (avoid overhead)
        reason = "in daemon process" if is_daemon else "small file count"
        logger.info(f"Loading {len(file_pairs)} parquet files sequentially for {dataset_name} ({reason})...")
        cur_datasets = []
        for pp, img_path, r in file_pairs:
            one_dataset = _prepare_dataset_worker(pp, img_path, r, use_index_tar_cache, exclude_task)
            if one_dataset is not None:
                cur_datasets.append(one_dataset)
    else:
        # For larger number of files, use multiprocessing
        # Adaptive worker count: min(num_workers, num_files)
        effective_workers = min(num_workers, len(file_pairs))
        logger.info(f"Loading {len(file_pairs)} parquet files with {effective_workers} workers for {dataset_name}...")
        
        # Create a worker function that doesn't depend on self
        worker_func = partial(
            _prepare_dataset_worker,
            use_index_tar_cache=use_index_tar_cache,
            exclude_task=exclude_task
        )
        
        # Use process pool
        with Pool(processes=effective_workers) as pool:
            cur_datasets = pool.starmap(worker_func, file_pairs)
        
        # Filter out None results (failed loads)
        cur_datasets = [d for d in cur_datasets if d is not None]
    
    cur_dataset_length = sum([len(d) for d in cur_datasets])
    logger.info(f'✓ Loaded {dataset_name} from {image_dir}: {len(file_pairs)} files, '
               f'{cur_dataset_length} samples in {(time.time() - tic):.2f}s')
    
    return cur_datasets, {
        'dataset_name': dataset_name,
        'tar_files': tar_files,
        'num_samples': cur_dataset_length,
        'num_files': len(file_pairs)
    }


def read_image_from_zip(zip_path, image_name):
    """从zip包中读取指定图片并返回PIL对象"""
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        # 读取图片数据
        with zip_file.open(image_name) as image_file:
            image_data = image_file.read()
            # 使用 PIL 读取图片
            return Image.open(io.BytesIO(image_data))


def read_image_from_tar(tar_path, image_name):
    """从tar包中读取指定图片并返回PIL对象"""
    with tarfile.open(tar_path, "r") as tar:
        try:
            member = tar.getmember(image_name)
            file_obj = tar.extractfile(member)
            return Image.open(io.BytesIO(file_obj.read()))
        except KeyError:
            # print(f"Error: {image_name} not found in {tar_path}")
            pass
        try:
            member = tar.getmember(os.path.basename(image_name))
            file_obj = tar.extractfile(member)
            return Image.open(io.BytesIO(file_obj.read()))
        except KeyError:
            logger.warning(f"Error: {image_name} not found in {tar_path}")

            
            
class UnifiedDataset(Dataset):
    """
    统一数据集类，支持多种数据加载、处理和分辨率归类处理
    """
    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        accelerator=None,
        seed: int = 42,
        transform: Optional[Callable] = None,
        resolution_bucketing: bool = True,
        common_resolutions: Optional[List[Tuple[int, int]]] = None,
        task_bucketing: bool = True,
        common_tasks: Optional[str] = ['t2i', 'edit'],
        cache_dir: Optional[str] = None,
        shuffle: bool = False,
        size=1024,
        resolution_preset: Optional[str] = None,
        select_tasks: Optional[List[str]] = None,
        exclude_tasks: Optional[List[str]] = None,
        num_workers: int = None
    ):
        """
        初始化统一数据集
        
        Args:
            dataset_configs: 数据集配置列表，每个配置包含数据集类型和参数
            accelerator: 加速器对象，用于分布式训练
            seed: 随机种子
            transform: 数据转换函数
            resolution_bucketing: 是否启用分辨率分类
            common_resolutions: 常用分辨率列表
            cache_dir: 缓存目录
            shuffle: 是否在创建数据集时打乱数据
            select_tasks: 要保留的任务列表，与exclude_tasks互斥
            exclude_tasks: 要排除的任务列表，与select_tasks互斥
            num_workers: Number of workers for parallel loading (None = auto)
        
        Raises:
            ValueError: 如果同时指定了select_tasks和exclude_tasks
        """
        self.dataset_configs = dataset_configs
        self.accelerator = accelerator
        self.seed = seed
        self.transform = transform
        self.resolution_bucketing = resolution_bucketing
        self.cache_dir = cache_dir
        self.use_index_tar_cache = True

        # Determine optimal number of workers
        if num_workers is None:
            # Use 75% of available CPUs, leaving some for system
            num_workers = max(1, int(cpu_count() * 0.75))
        self.num_workers = num_workers
        logger.info(f"Using {self.num_workers} workers for parallel dataset loading")
        
        # 通过tar目录进行索引
        self.tar_file_cache = {}
        
        # 加载数据集
        self.processed_datasets = self._load_datasets()
        self.combined_dataset = self._combine_datasets()

        # 任务过滤逻辑：只能选择select_tasks或exclude_tasks之一
        if select_tasks is not None and exclude_tasks is not None:
            raise ValueError("Cannot specify both select_tasks and exclude_tasks. Please use only one.")
        
        if select_tasks is not None:
            self.select_task(select_tasks)
        elif exclude_tasks is not None:
            self.exclude_task(exclude_tasks)
        
        # 如果需要，先打乱数据集
        if shuffle:
            self.shuffle(seed)
        if common_resolutions is None:
            resolution_preset, common_resolutions = resolve_qwen_edit_resolution_preset(resolution_preset, size)
            logger.info(f"Using Qwen Edit resolution preset {resolution_preset}: {common_resolutions}")
            
        # 分辨率分类
        if resolution_bucketing and not task_bucketing:
            logger.info("------- 分辨率分桶 ---")
            start_time = time.time()
            self.resolution_bucketer = ResolutionBucketer(common_resolutions=common_resolutions)
            elapsed_time = time.time() - start_time
            logger.info(f"建立ResolutionBucketer完成，耗时: {elapsed_time:.2f}秒")
            start_time = time.time()
            self._bucket_by_resolution()
            elapsed_time = time.time() - start_time
            logger.info(f"分桶完成，耗时: {elapsed_time:.2f}秒")
        elif resolution_bucketing and task_bucketing:
            logger.info("------- 分辨率+Task分桶 ---")
            start_time = time.time()
            self.resolution_bucketer = TaskResolutionBucketer(common_resolutions=common_resolutions, common_tasks=common_tasks)
            elapsed_time = time.time() - start_time
            logger.info(f"建立TaskResolutionBucketer完成，耗时: {elapsed_time:.2f}秒")
            start_time = time.time()
            self._bucket_by_resolution_task()
            elapsed_time = time.time() - start_time
            logger.info(f"分桶完成，耗时: {elapsed_time:.2f}秒")
        else:
            self.resolution_bucketer = None
        
        # 打印数据集信息
        self._print_dataset_info()
    
    def shuffle(self, seed: Optional[int] = None):
        """
        打乱数据集，在resolution bucketing之前调用
        
        Args:
            seed: 随机种子，如果为None则使用初始化时的seed
        
        Returns:
            打乱后的数据集实例（self）
        """
        if seed is None:
            seed = self.seed
        
        # 设置随机种子
        rng = np.random.RandomState(seed)
        
        # 获取当前数据集的长度
        dataset_length = len(self.combined_dataset)
        
        # 创建打乱的索引
        indices = np.arange(dataset_length)
        rng.shuffle(indices)
        
        # 使用indices重新排序数据集
        self.combined_dataset = self.combined_dataset.select(indices)
        
        logger.info(f"✓ Dataset shuffled with seed {seed}")
        return self

    def select_task(self, tasks: List[str]) -> None:
        """
        根据指定的任务列表过滤数据集
        
        Args:
            tasks: 要保留的任务列表
        
        Raises:
            ValueError: 如果指定的任务在数据集中不存在
        """
        # 获取数据集中所有的任务
        available_tasks = set(task.lower() for task in self.combined_dataset["task"])
        
        # 检查指定的任务是否都在数据集中
        requested_tasks = set(task.lower() for task in tasks)
        invalid_tasks = requested_tasks - available_tasks
        
        if invalid_tasks:
            raise ValueError(f"Invalid tasks specified: {invalid_tasks}. Available tasks: {available_tasks}")
        
        # 使用enumerate获取匹配的样本索引
        matched_ids = []
        for idx, task in enumerate(self.combined_dataset["task"]):
            if task.lower() in requested_tasks:
                matched_ids.append(idx)
        
        # 使用select方法获取数据子集
        self.combined_dataset = self.combined_dataset.select(matched_ids)
        logger.info(f"✓ Selected tasks {requested_tasks}, remaining samples: {len(self.combined_dataset)}")

    def exclude_task(self, tasks: List[str]) -> None:
        """
        根据指定的任务列表排除数据集中的样本（与select_task互补）
        
        Args:
            tasks: 要排除的任务列表
        
        Raises:
            ValueError: 如果指定的任务在数据集中不存在
        """
        # 获取数据集中所有的任务
        available_tasks = set(task.lower() for task in self.combined_dataset["task"])
        
        # 检查指定的任务是否都在数据集中
        requested_tasks = set(task.lower() for task in tasks)
        invalid_tasks = requested_tasks - available_tasks
        
        if invalid_tasks:
            raise ValueError(f"Invalid tasks specified: {invalid_tasks}. Available tasks: {available_tasks}")
        
        # 使用enumerate获取不匹配的样本索引（排除指定任务）
        matched_ids = []
        for idx, task in enumerate(self.combined_dataset["task"]):
            if task.lower() not in requested_tasks:
                matched_ids.append(idx)
        
        # 使用select方法获取数据子集
        self.combined_dataset = self.combined_dataset.select(matched_ids)
        logger.info(f"✓ Excluded tasks {requested_tasks}, remaining samples: {len(self.combined_dataset)}")


    def _load_datasets(self) -> List[Any]:
        """
        加载所有数据集 - Optimized with multiprocessing
        
        Returns:
            数据集列表
        """
        # Set umask to 002 for group-writable files when loading datasets
        os.umask(0o002)
        datasets = []
        
        # Detect if we're in distributed training and if this is the main process
        is_distributed = dist.is_available() and dist.is_initialized()
        is_main_process = not is_distributed or dist.get_rank() == 0
        
        # Check if we're already in a worker process (daemon)
        # If so, disable multiprocessing to avoid "daemonic processes cannot have children" error
        current_process = mp.current_process()
        is_daemon = current_process.daemon
        
        if is_daemon:
            logger.warning("Running in daemon process - disabling nested multiprocessing")
        
        num_configs = len(self.dataset_configs)
        
        # STRATEGY SELECTION: Parallelize dataset configs if we have multiple configs
        # Disable multiprocessing if we're in a daemon process
        if num_configs <= 2 or is_daemon:
            # For 1-2 configs, sequential processing is simpler
            logger.info(f"Loading {num_configs} dataset config(s) sequentially...")
            
            # Use tqdm progress bar only on main process
            iterator = tqdm(self.dataset_configs, desc="Loading datasets", disable=not is_main_process) if is_main_process else self.dataset_configs
            
            for dataset_config in iterator:
                cur_datasets, metadata = _load_single_dataset_config(
                    dataset_config, 
                    self.num_workers, 
                    self.use_index_tar_cache
                )
                
                # Cache tar files if needed
                if self.use_index_tar_cache:
                    for tar_file in metadata['tar_files']:
                        if tar_file not in self.tar_file_cache:
                            tar_reader = IndexedTarReader(tar_file)
                            self.tar_file_cache[tar_file] = tar_reader
                
                datasets.extend(cur_datasets)
        else:
            # For multiple configs, use parallel processing
            # Use fewer workers for config-level parallelization (they spawn their own workers)
            config_workers = min(2, num_configs)  # Limit to 4 to avoid oversubscription
            logger.info(f"Loading {num_configs} dataset configs with {config_workers} parallel workers...")
            
            # Create worker function
            worker_func = partial(
                _load_single_dataset_config,
                num_workers=max(1, self.num_workers // config_workers),  # Divide workers among configs
                use_index_tar_cache=self.use_index_tar_cache
            )
            
            # Use process pool with progress tracking
            with Pool(processes=config_workers) as pool:
                # Use imap for progress tracking (processes results as they complete)
                results_iterator = pool.imap(worker_func, self.dataset_configs)
                
                # Wrap with tqdm only on main process
                if is_main_process:
                    results_iterator = tqdm(results_iterator, total=num_configs, desc="Loading datasets")
                
                # Collect results
                for cur_datasets, metadata in results_iterator:
                    # Cache tar files if needed
                    if self.use_index_tar_cache:
                        for tar_file in metadata['tar_files']:
                            if tar_file not in self.tar_file_cache:
                                tar_reader = IndexedTarReader(tar_file)
                                self.tar_file_cache[tar_file] = tar_reader
                    
                    datasets.extend(cur_datasets)
        
        # Restore default umask before returning
        os.umask(0o022)
        return datasets
    
    
    def _combine_datasets(self) -> Any:
        """
        Merge all processed datasets
        
        Returns:
            Combined dataset
        """
        if len(self.processed_datasets) == 0:
            raise ValueError("No available datasets")
        
        if len(self.processed_datasets) == 1:
            return self.processed_datasets[0]

        combined_dataset = concatenate_datasets(self.processed_datasets)
        logger.info(f"✓ Datasets merged successfully: {len(combined_dataset)} total records")
        return combined_dataset
    
    def _bucket_by_resolution(self):
        for i, (height, width) in enumerate(zip(self.combined_dataset["height"], self.combined_dataset["width"])):
            try:
                self.resolution_bucketer.add_image(i, width, height)
            except Exception as e:
                continue

    def _bucket_by_resolution_task(self):
        for i, (height, width, task) in enumerate(zip(self.combined_dataset["height"], self.combined_dataset["width"], self.combined_dataset["task"])):
            try:
                # print(f'_bucket_by_resolution_task:{task}')
                self.resolution_bucketer.add_image(i, width, height, task)
            except Exception as e:
                continue

    def _print_dataset_info(self):
        """
        Print dataset information
        """
        from collections import Counter # 导入 Counter

        logger.info('╭────────────────────────────────────────────╮')
        logger.info(f"│ Total dataset size: {len(self.combined_dataset):>24} │")
        logger.info('╰────────────────────────────────────────────╯')
        
        if self.resolution_bucketing and self.resolution_bucketer is not None:
            bucket_indices = self.resolution_bucketer.get_bucket_indices()
            logger.info('╭────────────────────────────────────────────╮')
            logger.info('│           Resolution Distribution          │')
            logger.info('├────────────────────────────────────────────┤')
            for res, indices in bucket_indices.items():
                bucket_task = f"{res[2]} |" if len(res) == 3 else ''
                logger.info(f"│ {bucket_task} {res[0]}×{res[1]}: {len(indices):>30} images │")
            logger.info('╰────────────────────────────────────────────╯')

        # 添加 task 类型统计
        logger.info('╭────────────────────────────────────────────╮')
        logger.info('│              Instruction Distribution             │')
        logger.info('├────────────────────────────────────────────┤')
        instruction_type_list = [instruction_type.lower() for instruction_type in self.combined_dataset["instruction_type"]]

        instruction_type_counts = Counter(instruction_type_list)
        for instruction_type, count in instruction_type_counts.items():
            # 确保名称和数量对齐打印
            logger.info(f"│  {str(instruction_type):<20}: {count:>20} samples │")
        logger.info('╰────────────────────────────────────────────╯')

        # 添加 instruction 类型统计
        logger.info('╭────────────────────────────────────────────╮')
        logger.info('│              Task Distribution             │')
        logger.info('├────────────────────────────────────────────┤')
        try:
            # task_list = [json.loads(metadata.decode('utf-8'))["task"] for metadata in self.combined_dataset["metadata"]]
            task_list = [task.lower() for task in self.combined_dataset["task"]]

            task_counts = Counter(task_list)
            for task, count in task_counts.items():
                # 确保 task 名称和数量对齐打印
                logger.info(f"│  {str(task):<20}: {count:>20} samples │")
        except KeyError:
            logger.info("│  'task' column not found in dataset.      │")
        except Exception as e:
            logger.info(f"│  Error counting tasks: {str(e)[:30]}...   │") # 打印部分错误信息
        logger.info('╰────────────────────────────────────────────╯')
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        item = self.combined_dataset[idx]
        if self.transform:
            item = self.transform(item)
        return item
    
    def get_dataloader(
        self, 
        batch_size: int, 
        shuffle: bool = True, 
        num_workers: int = 0, 
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        stateful: bool = False,
        distributed: bool = False,
        global_batch_size: int = -1,
        
    ) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            drop_last: 是否丢弃最后一个不完整的批次
            collate_fn: 整理函数
            stateful: 是否使用StatefulDataLoader
            
        Returns:
            数据加载器
        """
        if collate_fn is None:
            collate_fn = self._default_collate_fn
        
        if self.resolution_bucketing and self.resolution_bucketer is not None:
            if distributed:
                sampler = ResolutionBucketSampler_distributed(
                    self.resolution_bucketer,
                    batch_size,
                    shuffle,
                    seed=self.seed
                )
            else:
                sampler = ResolutionBucketSampler(
                    self.resolution_bucketer,
                    batch_size,
                    shuffle,
                    global_batch_size=global_batch_size
                    
                    # seed=self.seed
                )

            if stateful:
                return StatefulDataLoader(
                    self,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    # batch_size=batch_size,
                    collate_fn=collate_fn
                )
            else:
                return DataLoader(
                    self,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    # batch_size=batch_size,
                    collate_fn=collate_fn
                )
        else:
            # 非 bucketing 情况下的分布式采样
            sampler = None
            if dist.is_available() and dist.is_initialized() and shuffle:
                 sampler = torch.utils.data.distributed.DistributedSampler(
                     self,
                     num_replicas=dist.get_world_size(),
                     rank=dist.get_rank(),
                     shuffle=shuffle,
                     seed=self.seed # 使用统一的 seed
                 )
                 shuffle = False # DistributedSampler 内部处理 shuffle

            if stateful:
                return StatefulDataLoader(
                    self,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler, # Pass the DistributedSampler here
                    num_workers=num_workers,
                    drop_last=drop_last,
                    collate_fn=collate_fn
                )
            else:
                return DataLoader(
                    self,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler, # Pass the DistributedSampler here
                    num_workers=num_workers,
                    drop_last=drop_last,
                    collate_fn=collate_fn
                )
 
    def _default_collate_fn(self, examples):
        """
        默认的整理函数，对每个样本进行 resize 和 center crop 处理
        注意：由于使用了 ResolutionBucketSampler，同一批次中的所有样本应该属于同一个分辨率桶
        """
        # 过滤掉无效的样本
        valid_examples = []
        error_sources = {}
        assert len(examples) == 1, 'current only support bs=1'
        for idx, example in enumerate(examples):
           
            try:
                if example['zip_file'].endswith('tar'):
                    if example['zip_file'] in self.tar_file_cache:
                        tar_reader = self.tar_file_cache[example['zip_file']]
                        example["edited_image"] = tar_reader.read_image(example["edited_img_path"])
                    else:
                        example["edited_image"] = read_image_from_tar(example['zip_file'], example["edited_img_path"]).convert("RGB")
                else:
                    example["edited_image"] = read_image_from_zip(example['zip_file'], example["edited_img_path"]).convert("RGB")
                
                if example['task'].lower() == 't2i':
                    edited_shape = example['edited_image'].size
                    # example["original_image"] = Image.new("RGB", edited_shape, color=(0, 0, 0))
                    example["original_image"] = None

                else:
                    if example['zip_file'].endswith('tar'):
                        if example['zip_file'] in self.tar_file_cache:
                            tar_reader = self.tar_file_cache[example['zip_file']]
                            example["original_image"] = tar_reader.read_image(example["original_img_path"])
                        else:
                            example["original_image"] = read_image_from_tar(example['zip_file'], example["original_img_path"]).convert("RGB")
                    else:
                        example["original_image"] = read_image_from_zip(example['zip_file'], example["original_img_path"]).convert("RGB")

            except Exception as e:
                import traceback
                # Get the actual traceback as a string
                tb_str = traceback.format_exc()
                logger.error(f"Error loading data from {example['zip_file']}\n"
                           f"Image paths: original={example.get('original_img_path', 'N/A')}, "
                           f"edited={example.get('edited_img_path', 'N/A')}\n"
                           f"Error: {type(e).__name__}: {str(e)}\n"
                           f"Traceback:\n{tb_str}",  
                           main_process_only=False)
                continue
          
            valid_examples.append(example)
            
        
        if not valid_examples:
            return {}
        if len(valid_examples) != len(examples):
            return {}
        
        # 这里需要从 ResolutionBucketSampler 获取当前批次的目标分辨率
        target_width, target_height = self.resolution_bucketer.find_best_resolution(valid_examples[0]["edited_image"].size[0], valid_examples[0]["edited_image"].size[1])
        
        # 处理所有有效样本
        original_images = [example["original_image"] for example in valid_examples]
        edited_images = [example["edited_image"] for example in valid_examples]
        tasks = [example["task"] for example in valid_examples]
        heights = [example["height"] for example in valid_examples]
        widths = [example["width"] for example in valid_examples]

        instructions = []
        for example in valid_examples:
            instruction = example["instruction"]
            instructions.append(instruction)
        
        original_pixel_values = []
        edited_pixel_values = []
        edited_resize_crop_images = []
        for org_img, edited_img in zip(original_images, edited_images):

            if  org_img is not None:
                org_img = org_img.resize(edited_img.size, Image.LANCZOS)
                ori_size = org_img.size
                ori_radio = max(target_width / ori_size[0], target_height / ori_size[1])
                ori_temp_size = (int(ori_size[1] * ori_radio), int(ori_size[0] * ori_radio))
                image_size_transforms = transforms.Compose([
                    transforms.Resize(ori_temp_size, interpolation=transforms.InterpolationMode.LANCZOS),
                    transforms.CenterCrop((target_height, target_width))
                ])
                org_img = image_size_transforms(org_img)

            edited_size = edited_img.size
            edited_radio = max(target_width / edited_size[0], target_height / edited_size[1])
            edited_temp_size = (int(edited_size[1] * edited_radio), int(edited_size[0] * edited_radio))
            image_size_transforms = transforms.Compose([
                transforms.Resize(edited_temp_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop((target_height, target_width))
            ]) 
            edited_img = image_size_transforms(edited_img)

            original_pixel_values.append(org_img)
            edited_pixel_values.append(edited_img)

        return {
            "original_image": original_pixel_values[0],
            "edit_image": edited_pixel_values[0],
            "prompt": instructions[0] if random.random() < 0.9 else " ",
            "task": tasks[0],
            "width": edited_pixel_values[0].size[0],
            "height": edited_pixel_values[0].size[1],
        }


def create_unified_dataset(
    dataset_configs: List[Dict[str, Any]],
    accelerator=None,
    seed: int = 42,
    transform: Optional[Callable] = None,
    resolution_bucketing: bool = True,
    common_resolutions: Optional[List[Tuple[int, int]]] = None,
    common_tasks=None,
    cache_dir: Optional[str] = None,
    shuffle: bool = False,
    size=1024,
    resolution_preset: Optional[str] = None,
    select_tasks: Optional[List[str]] = None,
    exclude_tasks: Optional[List[str]] = None,
    num_workers: int = None,
    use_task_prompt=False
) -> UnifiedDataset:
    """
    创建统一数据集
    
    Args:
        dataset_configs: 数据集配置列表
        accelerator: 加速器对象
        seed: 随机种子
        transform: 数据转换函数
        resolution_bucketing: 是否启用分辨率分类
        common_resolutions: 常用分辨率列表
        cache_dir: 缓存目录
        shuffle: 是否在创建数据集时打乱数据
        select_tasks: 要保留的任务列表，与exclude_tasks互斥
        exclude_tasks: 要排除的任务列表，与select_tasks互斥
        num_workers: Number of workers for parallel loading (None = auto)
        
    Returns:
        统一数据集
    """
    return UnifiedDataset(
        dataset_configs=dataset_configs,
        accelerator=accelerator,
        seed=seed,
        transform=transform,
        resolution_bucketing=resolution_bucketing,
        common_resolutions=common_resolutions,
        task_bucketing=common_tasks is not None,
        common_tasks=common_tasks,
        cache_dir=cache_dir,
        shuffle=shuffle,
        size=size,
        resolution_preset=resolution_preset,
        select_tasks=select_tasks,
        exclude_tasks=exclude_tasks,
        num_workers=num_workers
    )


def get_dataset_configs(
    dataset_name_list: List[str],
) -> List[Dict[str, Any]]:
    configs = []
    for dataset_name in dataset_name_list:
        if dataset_name == "mti_data":
            configs.append({
            "type": "mti_data",
            "path": "/cache/mti_data/I2I_Dataset_Share/20250630/",
            "dataset_name": 'mti_data',
            "original_image_column": "image",
            "edited_image_column": None,
            "instruction_column": "caption_caption",

        })  
    return configs


def tensor_to_image(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2. * 255
    x = x.cpu().numpy().astype(np.uint8)
    if x.shape[1] == 3 or x.shape[1] == 1:
        x = np.transpose(x, (0, 2, 3, 1))
    return x


if __name__ == "__main__":
    import yaml
    import warnings
    import accelerate
    from tqdm import tqdm
    accelerator = accelerate.Accelerator()
    warnings.filterwarnings('ignore')
    
    logging.basicConfig(level=logging.INFO)
    config_path = "configs/data/data_config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    dataset_configs = configs['datset_configs']
    
    print(dataset_configs)

    # Create unified dataset
    dataset = create_unified_dataset(
        dataset_configs=dataset_configs,
        accelerator=None,
        seed=42,
        resolution_bucketing=True,
        common_tasks=['t2i', 'edit']
    )

    # Get dataloader
    dataloader = dataset.get_dataloader(
        batch_size=1,
        shuffle=True,
        num_workers=8
    )
    
    # Test data loop
    print('╭────────────────────────────────────────────╮')
    print('│            Testing Data Pipeline           │')
    print('╰────────────────────────────────────────────╯')
    
    # Set number of batches to test
    num_batches_to_test = 50
    save_folder = '/home/ma-user/work/output/'
    os.makedirs(save_folder, exist_ok=True)

    import time
    try:
        
        tic = time.time()
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_batches_to_test:
                break
                
            # Print batch information
            print(f"Batch {i+1}/{num_batches_to_test}:")
            
            out_name = '_'.join(batch['prompt'].split(' ')[:20])
            
            if batch['original_image'] is not None:
                batch['original_image'].save(os.path.join(save_folder, f'{i:04d}_input.jpg'))
            batch['edit_image'].save(os.path.join(save_folder, f'{i:04d}_output.jpg'))
            
            with open(os.path.join(save_folder, f'{i:04d}.json'), 'wt') as f:
                json.dump({'instruction': batch['prompt'], 'height': batch['height'], 'width': batch['width'], 'task': batch['task']}, f)
            print('data loading time', time.time() - tic)

            tic = time.time()
        print('╭────────────────────────────────────────────╮')
        print('│        Data pipeline test successful!      │')
        print('╰────────────────────────────────────────────╯')

    except Exception as e:
        import traceback
        print(traceback.print_exc())
        print(f"❌ Data pipeline test failed: {e}")