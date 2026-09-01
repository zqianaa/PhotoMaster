import os, torch, argparse, json, math
from peft import LoraConfig, inject_adapter_in_model
import gc
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator

from diffusers.training_utils import free_memory
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, add_general_config, add_image_size_config

from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


def format_seconds(seconds):
    if seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) / 60
        return f"{hours}h{minutes:.2f}m"


class AverageMeter:
    """用于统计平均值、当前值和总和"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0        # 当前值
        self.avg = 0        # 平均值
        self.sum = 0        # 总和
        self.count = 0      # 样本数量

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_vis(accelerator, model, data_loader, save_dir, step):
    logger.info(f'run visualization in {step}')
    
    # model.pipe.scheduler.set_timesteps(40, training=False)
    for i, example in enumerate(data_loader):
        tic = time.time()
        control_image = example['original_image']
        gt_image = example['edit_image']
        image = model.pipe(
            prompt=example['prompt'],
            negative_prompt=' ',
            cfg_scale=4,
            num_inference_steps=40,
            seed=42,
            height=example['height'],
            width=example['width'],
            edit_image=control_image,
            zero_cond_t=model.zero_cond_t,
        )

        bs = 1
        j = 0
        save_path = os.path.join(save_dir, 'images', f"{step:06d}_{(i*bs + j):04d}_{accelerator.process_index:04d}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if control_image is not None:
            image = image.resize(control_image.size)
        else:
            control_image = Image.new("RGB", image.size, color=(0, 0, 0))
        array1, array2, array3 = np.array(control_image), np.array(image), np.array(gt_image)
        combined_array = np.hstack((array1, array2, array3))
        combined_image = Image.fromarray(combined_array)
        
        combined_image.save(save_path)
        with open(save_path.replace('.jpg', '.json'), 'wt') as f:
            json.dump({'prompt': example['prompt']}, f)
        
        logger.info(f'save image to {save_path}, take {time.time() - tic}')    
        break
    
    free_memory()
    gc.collect()
    
    model.pipe.scheduler.set_timesteps(1000, training=True)


def launch_training_task(
    accelerator: Accelerator,
    # 
    dataloader: torch.utils.data.DataLoader,
    # 
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    # 
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    # 
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
    # 
    gradient_accumulation_steps: int = 1,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_iter = 0

    num_batches = len(dataloader)        # Number of batches = Number of samples / Number of GPUs / batch size per GPU
    optimizer_steps = math.ceil(num_batches / gradient_accumulation_steps)
    total_iter = optimizer_steps * num_epochs

    data_meter = AverageMeter()
    iter_meter = AverageMeter()

    start_event = torch.cuda.Event(enable_timing=True)  # debug
    end_event = torch.cuda.Event(enable_timing=True)  # debug
    fwd_meter = AverageMeter()  # debug
    bwd_meter = AverageMeter()  # debug
    optim_meter = AverageMeter()  # debug
    
    for epoch_id in range(num_epochs):
        
        dataloader.batch_sampler.batch_sampler.set_epoch(epoch_id)
     
        # progress_bar = tqdm(range(0, optimizer_steps), desc=f"Rank {local_rank}; Epoch {epoch_id}", disable=not accelerator.is_local_main_process)

        tic_data = time.time()
        tic_iter_time = time.time()
        iter_idx = 0
        
        for data in dataloader:
            toc_data = time.time()
            with accelerator.accumulate(model):
                # --- Forward ---
                torch.cuda.synchronize()  # debug
                start_event.record()  # debug
                
                optimizer.zero_grad(set_to_none=True)
                loss = model(data)

                end_event.record()  # debug
                torch.cuda.synchronize()  # debug
                t_fwd = start_event.elapsed_time(end_event)  # debug, 单位是毫秒

                # --- Backward ---
                torch.cuda.synchronize()  # debug
                start_event.record()  # debug
                
                accelerator.backward(loss)

                end_event.record()  # debug
                torch.cuda.synchronize()  # debug
                t_bwd = start_event.elapsed_time(end_event)  # debug

                # --- Optimizer Step ---
                torch.cuda.synchronize()  # debug
                start_event.record()  # debug
                
                optimizer.step()
                lr_scheduler.step()

                end_event.record()  # debug
                torch.cuda.synchronize()  # debug
                t_optim = start_event.elapsed_time(end_event)  # debug

            fwd_meter.update(t_fwd)  # debug
            bwd_meter.update(t_bwd)  # debug
            optim_meter.update(t_optim)  # debug

            # torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # if global_iter == 0:
                #     model_logger.save_model(accelerator, model, os.path.join(model_logger.output_path, f"step-{global_iter}.safetensors"))
                
                model_logger.on_step_end(accelerator, model, save_steps)
                
                local_loss = loss.detach().mean()
                mean_loss = accelerator.gather(local_loss).cpu().mean().item()
                    
                # progress_bar.update(1)
                # logs = {"loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0], 'data_time': f"{(toc_data-tic_data):.3f}"}
                # progress_bar.set_postfix(**logs)
                
                if accelerator.is_main_process:
                    
                    iter_meter.update(time.time() - tic_iter_time)
                    data_meter.update(toc_data - tic_data)

                    mem_alloc = torch.cuda.memory_allocated() / 1024**3
                    gpu_memory_used = torch.cuda.memory_reserved() / 1024**3
                    speed = iter_meter.avg
                    ela_all = speed * (total_iter - global_iter)
                    ela_epoch = speed * (optimizer_steps - iter_idx)
                    # mean_loss = mean_loss

                    info_message = f"[Rank {accelerator.process_index}][Epoch:{epoch_id+1}/{num_epochs},GIter:{global_iter+1}/{total_iter}," \
                        f"EIter:{iter_idx+1}/{optimizer_steps}, {iter_meter.val:.2f}s/iter," \
                        f"ela-epoch:{format_seconds(ela_epoch)},ela-all:{format_seconds(ela_all)}]," \
                        f"[Loss:{mean_loss:.4f},LR:{(lr_scheduler.get_last_lr()[0]):.6f}]," \
                        f"[GPU_Allocated:{mem_alloc:.1f}GB,GPU_Reserved:{gpu_memory_used:.1f}GB,Data Time:{(data_meter.avg):.3f}]"
                    info_message += f"[Fwd:{(fwd_meter.avg):.2f}ms,Bwd:{(bwd_meter.avg):.2f}ms,Optim:{(optim_meter.avg):.2f}ms]"  # debug
                    logger.info(info_message)

                if global_iter == 0 or (global_iter + 1) % args.vis_steps == 0:
                    run_vis(accelerator, model, dataloader, model_logger.output_path, global_iter)
                    free_memory()
                
                iter_idx += 1
                global_iter += 1
                tic_iter_time = time.time()
            tic_data = time.time()

        free_memory()
        model_logger.on_epoch_end(accelerator, model, epoch_id+1)

    model_logger.on_training_end(accelerator, model, save_steps)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        marker_path = os.path.join(model_logger.output_path, ".photomaster_training_complete.json")
        tmp_marker_path = marker_path + ".tmp"
        os.makedirs(model_logger.output_path, exist_ok=True)
        with open(tmp_marker_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "completed",
                    "global_steps": model_logger.num_steps,
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "success_token": os.environ.get("PHOTOMASTER_SUCCESS_TOKEN"),
                },
                f,
                indent=2,
            )
        os.replace(tmp_marker_path, marker_path)
        logger.info(f"wrote training completion marker to {marker_path}")
    accelerator.end_training()


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
            break
    
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")

    # data related
    parser.add_argument("--config_path", type=str, default="", help="Path of the dataset configs.")
    parser.add_argument("--task_buckets", type=str, default='', help="task1,task2,task3")
    parser.add_argument("--size", type=int, default=1024, help="training max image size")
    parser.add_argument(
        "--resolution_preset",
        type=str,
        default=None,
        help="Named Qwen Edit training resolution preset. Overrides --size when provided.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--global_batch_size", type=int, default=-1, help="global batch size.")

    # customized
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume_from_checkpoint", type=str, help='resume training ckpt')
    parser.add_argument("--vis_steps", type=int, default=5000, help="Number of vis invervals")
    parser.add_argument("--projector_path",type=str, default='', help="vllm prjector path")
    parser.add_argument("--use_task_prompt",type=str, default='', help="use_task_prompt")

    return parser
