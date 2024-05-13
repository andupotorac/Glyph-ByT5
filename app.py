
import json
import webcolors
import gradio as gr
import os.path as osp
from PIL import Image, ImageDraw, ImageFont

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

from demo.constants import MAX_TEXT_BOX


html = f"""<h1>Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering</h1>
            <h2><a href='https://glyph-byt5.github.io/'>Project Page</a> | <a href='https://arxiv.org/abs/2403.09622'>arXiv Paper</a> | <a href=''>Github</a> | <a href=''>Cite our work</a> if our ideas inspire you.</h2>
            <p><b>Try some examples at the bottom of the page to get started!</b></p>
            <p><b>Usage:</b></p>
            <p>1. <b>Select bounding boxes</b> on the canvas on the left <b>by clicking twice</b>. </p>
            <p>2. Click "Redo" if you want to cancel last point, "Undo" for clearing the canvas. </p>
            <p>3. <b>Click "I've finished my layout!"</b> to start choosing specific prompts, colors and font-types. </p>
            <p>4. Enter a <b>design prompt</b> for the background image. Optionally, you can choose to specify the design categories and tags (separated by a comma). </p>
            <p>5. For each text box, <b>enter the text prompts in the text box</b> on the left, and <b>select colors and font-types from the drop boxes</b> on the right. </p>
            <p>6. <b>Click on "I've finished my texts, colors and styles, generate!"</b> to start generating!. </p>
            <style>.btn {{flex-grow: unset !important;}} </p>
            """


css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''

state = 0
stack = []
font = ImageFont.truetype("assets/Arial.ttf", 20)

device = "cuda" if torch.cuda.is_available() else 'cpu'

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

config = parse_config('configs/glyph_sdxl_albedo.py')
ckpt_dir = 'checkpoints/glyph-sdxl'

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
    vae.to(device, dtype=torch.float32)
else:
    vae.to(device, dtype=inference_dtype)
text_encoder_one.to(device, dtype=inference_dtype)
text_encoder_two.to(device, dtype=inference_dtype)
byt5_model.to(device)
unet.to(device, dtype=inference_dtype)

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
            origin_module.to(device, dtype=inference_dtype)
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

unet_lora_layers_para = torch.load(osp.join(ckpt_dir, UNET_CKPT_NAME), map_location='cpu')
incompatible_keys = set_peft_model_state_dict(unet, unet_lora_layers_para, adapter_name="default")
if getattr(incompatible_keys, 'unexpected_keys', []) == []:
    print(f"loaded unet_lora_layers_para")
else:
    print(f"unet_lora_layers has unexpected_keys: {getattr(incompatible_keys, 'unexpected_keys', None)}")

inserted_attn_module_paras = torch.load(osp.join(ckpt_dir, INSERTED_ATTN_CKPT_NAME), map_location='cpu')
missing_keys, unexpected_keys = unet.load_state_dict(inserted_attn_module_paras, strict=False)
assert len(unexpected_keys) == 0, unexpected_keys

byt5_mapper_para = torch.load(osp.join(ckpt_dir, BYT5_MAPPER_CKPT_NAME), map_location='cpu')
byt5_mapper.load_state_dict(byt5_mapper_para)

byt5_model_para = torch.load(osp.join(ckpt_dir, BYT5_CKPT_NAME), map_location='cpu')
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

pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
    config.pretrained_model_name_or_path,
    subfolder="scheduler",
    use_karras_sigmas=True,
)

pipeline = pipeline.to(device)

prompt_format = PromptFormat()

def get_pixels(
    box_sketch_template,
    evt: gr.SelectData
):
    global state
    global stack

    text_position = evt.index

    if state == 0:
        stack.append(text_position)
        state = 1
    else:
        x, y = stack.pop()
        stack.append([x, y, text_position[0], text_position[1]])
        state = 0

    print(stack)

    box_sketch_template = Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw = ImageDraw.Draw(box_sketch_template)

    for i, text_position in enumerate(stack):
        if len(text_position) == 2:
            x, y = text_position
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)

            text_color = (255, 0, 0)  
            draw.text((x+2, y), str(i + 1), font=font, fill=text_color)

            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)

            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), str(i + 1), font=font, fill=text_color)
            
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))

    return box_sketch_template

def exe_redo(
    box_sketch_template
):
    global state
    global stack

    state = 1 - state
    if len(stack[-1]) == 2:
        stack = stack[:-1]
    else:
        x, y, _, _ = stack[-1]
        stack = stack[:-1] + [[x, y]]

    box_sketch_template = Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw = ImageDraw.Draw(box_sketch_template)

    for i, text_position in enumerate(stack):
        if len(text_position) == 2:
            x, y = text_position
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)

            text_color = (255, 0, 0)  
            draw.text((x+2, y), str(i+1), font=font, fill=text_color)

            draw.ellipse((leftUpPoint, rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)

            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), str(i+1), font=font, fill=text_color)

            draw.rectangle((x0,y0,x1,y1), outline=(255, 0, 0))

    return box_sketch_template

def exe_undo(
    box_sketch_template
):
    global state
    global stack
    
    state = 0
    stack = []
    box_sketch_template = Image.new('RGB', (1024, 1024), (255, 255, 255))

    return box_sketch_template

def process_box():
    global stack
    global state

    visibilities = []
    for _ in range(MAX_TEXT_BOX + 1):
        visibilities.append(gr.update(visible=False))
    for n in range(len(stack) + 1):
        visibilities[n] = gr.update(visible=True)
    
    # return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]
    return [gr.update(visible=True), *visibilities]

def generate_image(bg_prompt, bg_class, bg_tags, seed, *conditions):
    print(conditions)
    
    # 1. parse input
    global state
    global stack

    prompts = []
    colors = []
    font_type = []
    bboxes = []
    num_boxes = len(stack) if len(stack[-1]) == 4 else len(stack) - 1
    for i in range(num_boxes):
        prompts.append(conditions[i])
        colors.append(conditions[i + MAX_TEXT_BOX])
        font_type.append(conditions[i + MAX_TEXT_BOX * 2])

    # 2. input check
    styles = []
    if bg_prompt == "" or bg_prompt is None:
        raise gr.Error("Empty background prompt!")
    for i, (prompt, color, style) in enumerate(zip(prompts, colors, font_type)):
        if prompt == "" or prompt is None:
            raise gr.Error(f"Invalid prompt for text box {i + 1} !")
        if color is None:
            raise gr.Error(f"Invalid color for text box {i + 1} !")
        if style is None:
            raise gr.Error(f"Invalid style for text box {i + 1} !")
        bboxes.append(
            [
                stack[i][0] / 1024,
                stack[i][1] / 1024,
                (stack[i][2] - stack[i][0]) / 1024,
                (stack[i][3] - stack[i][1]) / 1024,
            ]
        )
        styles.append(
            {
                'color': webcolors.name_to_hex(color),
                'font-family': style,
            }
        )

    # 3. format input
    if bg_class != "" and bg_class is not None:
        bg_prompt = bg_class + ". " + bg_prompt
    if bg_tags != "" and bg_tags is not None:
        bg_prompt += " Tags: " + bg_tags
    text_prompt = prompt_format.format_prompt(prompts, styles)

    print(bg_prompt)
    print(text_prompt)

    # 4. inference
    if seed == -1:
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    with torch.cuda.amp.autocast():
        image = pipeline(
            prompt=bg_prompt,
            text_prompt=text_prompt,
            texts=prompts,
            bboxes=bboxes,
            num_inference_steps=50,
            generator=generator,
            text_attn_mask=None,
        ).images[0]
    return image

def process_example(bg_prompt, bg_class, bg_tags, color_str, style_str, text_str, box_str, seed):
    global stack
    global state
    
    colors = color_str.split(",")
    styles = style_str.split(",")
    boxes = box_str.split(";")
    prompts = text_str.split("**********")
    colors = [color.strip() for color in colors]
    styles = [style.strip() for style in styles]
    colors += [None] * (MAX_TEXT_BOX - len(colors))
    styles += [None] * (MAX_TEXT_BOX - len(styles))
    prompts += [""] * (MAX_TEXT_BOX - len(prompts))

    state = 0
    stack = []
    print(boxes)
    for box in boxes:
        print(box)
        box = box.strip()[1:-1]
        print(box)
        box = box.split(",")
        print(box)
        x = eval(box[0].strip()) * 1024
        y = eval(box[1].strip()) * 1024
        w = eval(box[2].strip()) * 1024
        h = eval(box[3].strip()) * 1024
        stack.append([int(x), int(y), int(x + w + 0.5), int(y + h + 0.5)])

    visibilities = []
    for _ in range(MAX_TEXT_BOX + 1):
        visibilities.append(gr.update(visible=False))
    for n in range(len(stack) + 1):
        visibilities[n] = gr.update(visible=True)

    box_sketch_template = Image.new('RGB', (1024, 1024), (255, 255, 255))
    draw = ImageDraw.Draw(box_sketch_template)

    for i, text_position in enumerate(stack):
        if len(text_position) == 2:
            x, y = text_position
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)

            text_color = (255, 0, 0)  
            draw.text((x+2, y), str(i + 1), font=font, fill=text_color)

            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)

            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), str(i + 1), font=font, fill=text_color)
            
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))

    return [
        gr.update(visible=True), box_sketch_template, seed, *visibilities, *colors, *styles, *prompts,
    ]

def main():
    # load configs
    with open('assets/color_idx.json', 'r') as f:
        color_idx_dict = json.load(f)
        color_idx_list = list(color_idx_dict)
    with open('assets/font_idx_512.json', 'r') as f:
        font_idx_dict = json.load(f)
        font_idx_list = list(font_idx_dict)
    
    with gr.Blocks(
        title="Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering",
        css=css,
    ) as demo:
        gr.HTML(html)
        with gr.Row():
            with gr.Column(elem_id="main-image"):
                box_sketch_template = gr.Image(
                    value=Image.new('RGB', (1024, 1024), (255, 255, 255)), 
                    sources=[],
                    interactive=False,
                )

                box_sketch_template.select(get_pixels, [box_sketch_template], [box_sketch_template])

                with gr.Row():
                    redo = gr.Button(value='Redo - Cancel last point') 
                    undo = gr.Button(value='Undo - Clear the canvas') 
                redo.click(exe_redo, [box_sketch_template], [box_sketch_template])
                undo.click(exe_undo, [box_sketch_template], [box_sketch_template])

                button_layout = gr.Button("(1) I've finished my layout!", elem_id="main_button", interactive=True)

                prompts = []
                colors = []
                styles = []
                color_row = [None] * (MAX_TEXT_BOX + 1)
                with gr.Column(visible=False) as post_box:
                    for n in range(MAX_TEXT_BOX + 1):
                        if n == 0 :
                            with gr.Row(visible=True) as color_row[n]:
                                bg_prompt = gr.Textbox(label="Design prompt for the background image", value="")
                                bg_class = gr.Textbox(label="Design type for the background image (optional)", value="")
                                bg_tags = gr.Textbox(label="Design type for the background image (optional)", value="")
                        else:
                            with gr.Row(visible=False) as color_row[n]:
                                prompts.append(gr.Textbox(label="Prompt for box "+str(n)))
                                colors.append(gr.Dropdown(
                                    label="Color for box "+str(n),
                                    choices=color_idx_list,
                                ))
                                styles.append(gr.Dropdown(
                                    label="Font type for box "+str(n),
                                    choices=font_idx_list,
                                ))

                    seed_ = gr.Slider(label="Seed", minimum=-1, maximum=999999999, value=-1, step=1)
                    button_generate = gr.Button("(2) I've finished my texts, colors and styles, generate!", elem_id="main_button", interactive=True)

                button_layout.click(process_box, inputs=[], outputs=[post_box, *color_row], queue=False)

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False)

            button_generate.click(generate_image, inputs=[bg_prompt, bg_class, bg_tags, seed_, *(prompts + colors + styles)], outputs=[output_image], queue=True)

        # examples
        color_str = gr.Textbox(label="Color list", value="", visible=False)
        style_str = gr.Textbox(label="Font type list", value="", visible=False)
        box_str = gr.Textbox(label="Bbox list", value="", visible=False)
        text_str = gr.Textbox(label="Text list", value="", visible=False)

        gr.Examples(
            examples=[
                [
                    'The image features a small bunny rabbit sitting in a basket filled with various flowers. The basket is placed on a yellow background, creating a vibrant and cheerful scene. The flowers surrounding the rabbit come in different sizes and colors, adding to the overall visual appeal of the image. The rabbit appears to be the main focus of the scene, and its presence among the flowers creates a sense of harmony and balance.', 
                    'Facebook Post', 
                    'green, yellow, minimalist, easter day, happy easter day, easter, happy easter, decoration, happy, egg, spring, selebration, poster, illustration, greeting, season, design, colorful, cute, template', 
                    'darkolivegreen, darkolivegreen, darkolivegreen', 
                    'Gagalin-Regular, Gagalin-Regular, Brusher-Regular',
                    'MAY ALLYOUR PRAYERS BE ANSWERED**********HAVE A HAPPY**********Easter Day',
                    '[0.08267477203647416, 0.5355623100303951, 0.42857142857142855, 0.07477203647416414]; [0.08389057750759879, 0.1951367781155015, 0.38054711246200607, 0.03768996960486322]; [0.07537993920972644, 0.2601823708206687, 0.49544072948328266, 0.14650455927051673]',
                    1,
                ],
                [
                    'The image features a large gray elephant sitting in a field of flowers, holding a smaller elephant in its arms. The scene is quite serene and picturesque, with the two elephants being the main focus of the image. The field is filled with various flowers, creating a beautiful and vibrant backdrop for the elephants.',
                    'Cards and invitations',
                    'Light green, orange, Illustration, watercolor, playful, Baby shower invitation, baby boy shower invitation, baby boy, welcoming baby boy, koala baby shower invitation, baby shower invitation for baby shower, baby boy invitation, background, playful baby shower card, baby shower, card, newborn, born, Baby Shirt Baby Shower Invitation',
                    'peru, olive, olivedrab, peru, peru, peru',
                    'LilitaOne, Sensei-Medium, Sensei-Medium, LilitaOne, LilitaOne, LilitaOne',
                    "RSVP to +123-456-7890**********Olivia Wilson**********Baby Shower**********Please Join Us For a**********In Honoring**********23 November, 2021 | 03:00 PM Fauget Hotels",
                    '[0.07112462006079028, 0.6462006079027356, 0.3373860182370821, 0.026747720364741642]; [0.07051671732522796, 0.38662613981762917, 0.37264437689969604, 0.059574468085106386]; [0.07234042553191489, 0.15623100303951368, 0.6547112462006079, 0.12401215805471125]; [0.0662613981762918, 0.06747720364741641, 0.3981762917933131, 0.035866261398176294]; [0.07051671732522796, 0.31550151975683893, 0.22006079027355624, 0.03951367781155015]; [0.06990881458966565, 0.48328267477203646, 0.39878419452887537, 0.1094224924012158]',
                    0,
                ],
                [
                    'The image features a white background with a variety of colorful flowers and decorations. There are several pink flowers scattered throughout the scene, with some positioned closer to the top and others near the bottom. A blue flower can also be seen in the middle of the image. The overall composition creates a visually appealing and vibrant display.',
                    'Instagram Posts',
                    'grey, navy, purple, pink, teal, colorful, illustration, happy, celebration, post, party, year, new, event, celebrate, happy new year, new year, countdown, sparkle, firework',
                    'purple, midnightblue, black, black',
                    'Caveat-Regular, Gagalin-Regular, Quicksand-Light, Quicksand-Light',
                    'Happy New Year**********2024**********All THE BEST**********A fresh start to start a change for the better.',
                    '[0.2936170212765957, 0.2887537993920973, 0.40303951367781155, 0.07173252279635259]; [0.24984802431610942, 0.3951367781155015, 0.46200607902735563, 0.17203647416413373]; [0.3951367781155015, 0.1094224924012158, 0.2109422492401216, 0.02796352583586626]; [0.20911854103343466, 0.6127659574468085, 0.5586626139817629, 0.08085106382978724]',
                    1,
                ],
                [
                    'The image features a stack of pancakes with syrup and strawberries on top. The pancakes are arranged in a visually appealing manner, with some pancakes placed on top of each other. The syrup is drizzled generously over the pancakes, and the strawberries are scattered around, adding a touch of color and freshness to the scene. The overall presentation of the pancakes is appetizing and inviting.',
                    'Instagram Posts',
                    'brown, peach, grey, modern, minimalist, simple, colorful, illustration, Instagram post, instagram, post, national pancake day, international pancake day, happy pancake day, pancake day, pancake, sweet, cake, discount, sale',
                    'dimgray, white, darkolivegreen',
                    'MoreSugarRegular, Chewy-Regular, Chewy-Regular',
                    'Get 75% Discount for your first order**********Order Now**********National Pancake Day',
                    '[0.043161094224924014, 0.5963525835866261, 0.2936170212765957, 0.08389057750759879]; [0.12279635258358662, 0.79209726443769, 0.26382978723404255, 0.05167173252279635]; [0.044984802431610946, 0.09787234042553192, 0.4413373860182371, 0.4158054711246201]',
                    1,
                ]
            ],
            inputs=[
                bg_prompt,
                bg_class,
                bg_tags,
                color_str,
                style_str,
                text_str,
                box_str,
                seed_,
            ],
            outputs=[post_box, box_sketch_template, seed_, *color_row, *colors, *styles, *prompts],
            fn=process_example,
            run_on_click=True,
            label='Examples',
        )

    demo.queue()
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    main()
