import torch, os, accelerate
import json, sys, warnings, yaml
import math
import numpy as np
from PIL import Image
from pathlib import Path

from accelerate.utils import DistributedDataParallelKwargs, set_seed
# import torch.distributed as dist
# from torchvision import transforms

from diffsynth import load_state_dict
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, FlowMatchSFTLoss

# utils/, pipelines/ and data/ are siblings of this file. Upstream this script
# lived one level deeper (DialogEdit_qwen/train/train.py) and appended its
# grandparent; here the packages sit alongside, so append this file's own dir.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_logger import ModelLogger
from utils.utils import launch_training_task, qwen_image_parser
from pipelines.qwen_image import QwenImagePipeline, ModelConfig

from data.tar_parquet_data_v2 import create_unified_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from accelerate.logging import get_logger
# Configure the logger with INFO level
logger = get_logger(__name__, log_level="INFO")

# If you want to also configure the root logger for consistency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
    
    def get_pipeline_inputs(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["edit_image"],
            "edit_image": data["original_image"],
            "height": data["height"],
            "width": data["width"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": False,  # TODO: be careful!!!
            "zero_cond_t": self.zero_cond_t,
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        # if isinstance(data["image"], list):
        #     inputs_shared.update({
        #         "input_image": data["image"],
        #         "height": data["image"][0].size[1],
        #         "width": data["image"][0].size[0],
        #     })
        # else:
        #     inputs_shared.update({
        #         "input_image": data["image"],
        #         "height": data["image"].size[1],
        #         "width": data["image"].size[0],
        #     })
        # inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def setup_logger(args):
    
    logging_out_dir = args.output_path

    os.makedirs(logging_out_dir, exist_ok=True)
    log_file_path = os.path.join(logging_out_dir, "training_log.txt")

    # Set up logging to write to both console and file
    # First, remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Configure the basic logging format for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Create a file handler for the log file with detailed format
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # Set the overall logging level
    root_logger.setLevel(logging.INFO)
    
if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # Reproducibility. The diffusion timestep and noise are drawn from the global
    # torch RNG (diffsynth/diffusion/loss.py:9,12), so without seeding it two runs
    # of the same config diverge step to step. The dataset shuffle and batch
    # sampler are already seeded off --seed independently.
    #
    # device_specific offsets the seed by process_index: seeding every rank
    # identically would make all ranks sample the same timestep, which changes
    # training rather than just making it reproducible.
    #
    # Must come after Accelerator() -- set_seed reads AcceleratorState().
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)


    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        zero_cond_t=args.zero_cond_t,
    )
    if args.resume_from_checkpoint:
        print(f'\nload ckpt from {args.resume_from_checkpoint}')
        state_dict = load_state_dict(args.resume_from_checkpoint)
        model.pipe.dit.load_state_dict(state_dict)
        # dist.barrier(device_ids=[dist.get_rank()])
        print(f'\nload {args.resume_from_checkpoint} finish.')
    
    model.train()

    accelerator.wait_for_everyone()

    setup_logger(args)

    print(f"Using batch_size: {args.batch_size} for dataloader")
    print(f"Using gradient_accumulation_steps: {accelerator.gradient_accumulation_steps}")

    with open(args.config_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset_configs = configs['datset_configs']
    dataset = create_unified_dataset(
        dataset_configs=dataset_configs,
        accelerator=accelerator,
        seed=args.seed,
        transform=None,
        resolution_bucketing=True,
        common_resolutions=None,
        common_tasks=None if not args.task_buckets else args.task_buckets.split(','),
        cache_dir=None,
        shuffle=False,
        size=args.size,
        resolution_preset=args.resolution_preset,
    )
    # Use UnifiedDataset's get_dataloader method
    dataloader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataset_num_workers,
        global_batch_size=args.global_batch_size
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    logger.info("***** accelerator prepare  *****")
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(model, optimizer, dataloader, lr_scheduler)
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num batches each epoch = {len(dataloader)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f'  Total Trainable parameters = {(sum([np.prod(p.size()) for p in model.trainable_modules()])):,}')

    logger.info("***** Running training *****")

    launch_training_task(
        accelerator, dataloader, model, model_logger, optimizer, lr_scheduler,
        args=args,
    )
