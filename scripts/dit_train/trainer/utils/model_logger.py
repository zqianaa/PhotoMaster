import os
from accelerate.utils import DistributedType
from accelerate.logging import get_logger

logger = get_logger(__name__)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id):
        file_name = f"epoch-{epoch_id}.safetensors"
        self.save_model(accelerator, model, file_name)

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        
        is_zero_3 = False
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            is_zero_3 = accelerator.deepspeed_config["zero_optimization"]["stage"] == 3

        if accelerator.is_main_process or is_zero_3:
            state_dict = accelerator.get_state_dict(model)
        
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            logger.info(f'save model to {path}')
            accelerator.save(state_dict, path, safe_serialization=True)
            