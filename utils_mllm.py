from transformers import MllamaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_mllama, apply_liger_kernel_to_qwen2_vl
import torch

conv_mode = "llama_3" 

def evaluate_model_config(model, model_path, use_liger_kernel=True, device="cuda"):
    if "cambrian" in model:
        raise NotImplementedError
    elif "qwen" in model.lower():
        from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_path, max_pixels = 512 * 28 * 28)
        # Load and update configuration to ensure it's not a dict.
        # config = AutoConfig.from_pretrained(model_path)
        # config.attn_implementation = "flash_attention_2"
        # if use_liger_kernel:
        #     apply_liger_kernel_to_qwen2_vl()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16, 
        )
        return processor, model
    elif "llama" in model.lower():
        processor = AutoProcessor.from_pretrained(model_path)
        
        if use_liger_kernel:
            apply_liger_kernel_to_mllama()
        
        model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
            )
        return processor, model
                