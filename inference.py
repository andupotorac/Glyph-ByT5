import argparse
import os
import json
import copy
import os.path as osp

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from transformers import PretrainedConfig

from diffusers import DPMSolverMultistepScheduler

from glyph_sdxl.utils import (
    parse_config,
    UNET_CKPT_NAME,
    huggingface_cache_dir,
    load_byt5_and_byt5_tokenizer,
    BYT5_MAPPER_CKPT_NAME,
    INSERTED_ATTN_CKPT_NAME,
    BYT5_CKPT_NAME,
    PromptFormat,
)
from glyph_sdxl.custom_diffusers import (
    StableDiffusionGlyphXLPipeline,
    CrossAttnInsertBasicTransformerBlock,
)
from glyph_sdxl.modules import T5EncoderBlockByT5Mapper

byt5_mapper_dict = [T5EncoderBlockByT5Mapper]
byt5_mapper_dict = {mapper.__name__: mapper for mapper in byt5_mapper_dict}


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder=subfolder, 
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir", type=str)
    parser.add_argument("ckpt_dir", type=str)
    parser.add_argument("ann_path", type=str, default='examples/shower.json')
    parser.add_argument("--out_folder", type=str, default='None')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--sampler", type=str, choices=['euler', 'dpm'])
    args = parser.parse_args()
    
    config = parse_config(args.config_dir)
        
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_2",
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )
    
    vae_path = (
        config.pretrained_model_name_or_path
        if config.pretrained_vae_model_name_or_path is None
        else config.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if config.pretrained_vae_model_name_or_path is None else None, 
        revision=config.revision,
        cache_dir=huggingface_cache_dir,
    )

    byt5_model, byt5_tokenizer = load_byt5_and_byt5_tokenizer(
        **config.byt5_config,
        huggingface_cache_dir=huggingface_cache_dir,
    )

    inference_dtype = torch.float32
    if config.inference_dtype == "fp16":
        inference_dtype = torch.float16
    elif config.inference_dtype == "bf16":
        inference_dtype = torch.bfloat16

    if config.pretrained_vae_model_name_or_path is None:
        vae.to(args.device, dtype=torch.float32)
    else:
        vae.to(args.device, dtype=inference_dtype)
    text_encoder_one.to(args.device, dtype=inference_dtype)
    text_encoder_two.to(args.device, dtype=inference_dtype)
    byt5_model.to(args.device)
    unet.to(args.device, dtype=inference_dtype)

    inserted_new_modules_para_set = set()
    for name, module in unet.named_modules():
        if isinstance(module, BasicTransformerBlock) and name in config.attn_block_to_modify:
            parent_module = unet
            for n in name.split(".")[:-1]:
                parent_module = getattr(parent_module, n)
            new_block = CrossAttnInsertBasicTransformerBlock.from_transformer_block(
                module,
                byt5_model.config.d_model if config.byt5_mapper_config.sdxl_channels is None else config.byt5_mapper_config.sdxl_channels,
            )
            new_block.requires_grad_(False)
            for inserted_module_name, inserted_module in zip(
                new_block.get_inserted_modules_names(), 
                new_block.get_inserted_modules()
            ):
                inserted_module.requires_grad_(True)
                for para_name, para in inserted_module.named_parameters():
                    para_key = name + '.' + inserted_module_name + '.' + para_name
                    assert para_key not in inserted_new_modules_para_set
                    inserted_new_modules_para_set.add(para_key)
            for origin_module in new_block.get_origin_modules():
                origin_module.to(args.device, dtype=inference_dtype)
            parent_module.register_module(name.split(".")[-1], new_block)
            print(f"inserted cross attn block to {name}")

    byt5_mapper = byt5_mapper_dict[config.byt5_mapper_type](
        byt5_model.config,
        **config.byt5_mapper_config,
    )

    unet_lora_target_modules = [
        "attn1.to_k", "attn1.to_q", "attn1.to_v", "attn1.to_out.0",
        "attn2.to_k", "attn2.to_q", "attn2.to_v", "attn2.to_out.0",
    ]
    unet_lora_config = LoraConfig(
        r=config.unet_lora_rank,
        lora_alpha=config.unet_lora_rank,
        init_lora_weights="gaussian",
        target_modules=unet_lora_target_modules,
    )
    unet.add_adapter(unet_lora_config)
    
    unet_lora_layers_para = torch.load(osp.join(args.ckpt_dir, UNET_CKPT_NAME), map_location='cpu')
    incompatible_keys = set_peft_model_state_dict(unet, unet_lora_layers_para, adapter_name="default")
    if getattr(incompatible_keys, 'unexpected_keys', []) == []:
        print(f"loaded unet_lora_layers_para")
    else:
        print(f"unet_lora_layers has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")
    
    inserted_attn_module_paras = torch.load(osp.join(args.ckpt_dir, INSERTED_ATTN_CKPT_NAME), map_location='cpu')
    missing_keys, unexpected_keys = unet.load_state_dict(inserted_attn_module_paras, strict=False)
    assert len(unexpected_keys) == 0, unexpected_keys
    
    byt5_mapper_para = torch.load(osp.join(args.ckpt_dir, BYT5_MAPPER_CKPT_NAME), map_location='cpu')
    byt5_mapper.load_state_dict(byt5_mapper_para)
    
    byt5_model_para = torch.load(osp.join(args.ckpt_dir, BYT5_CKPT_NAME), map_location='cpu')
    byt5_model.load_state_dict(byt5_model_para)

    pipeline = StableDiffusionGlyphXLPipeline.from_pretrained(
        config.pretrained_model_name_or_path, 
        vae=vae, 
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        byt5_text_encoder=byt5_model,
        byt5_tokenizer=byt5_tokenizer,
        byt5_mapper=byt5_mapper,
        unet=unet,
        byt5_max_length=config.byt5_max_length,
        revision=config.revision,
        torch_dtype=inference_dtype,
        safety_checker=None,
        cache_dir=huggingface_cache_dir,
    )

    if args.sampler == 'dpm':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="scheduler",
            use_karras_sigmas=True,
        )

    pipeline = pipeline.to(args.device)

    with open(args.ann_path, 'r') as f:
        ann = json.load(f)
    
    os.makedirs(args.out_folder, exist_ok=True)

    prompt_format = PromptFormat()

    texts = copy.deepcopy(ann['texts'])
    bboxes = copy.deepcopy(ann['bbox'])
    styles = copy.deepcopy(ann['styles'])

    text_prompt = prompt_format.format_prompt(texts, styles)

    seed = 0 if 'seed' not in ann else ann['seed']
    generator = torch.Generator(device=args.device).manual_seed(seed)


    with torch.cuda.amp.autocast():
        image = pipeline(
            prompt=ann['bg_prompt'],
            text_prompt=text_prompt,
            texts=texts,
            bboxes=bboxes,
            num_inference_steps=50,
            generator=generator,
            text_attn_mask=None,
        ).images[0]
    image.save(f'{args.out_folder}/result.png')
