import time

from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, \
    EulerAncestralDiscreteScheduler, StableDiffusionLatentUpscalePipeline, DPMSolverMultistepScheduler, \
    StableDiffusionImg2ImgPipeline
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import json
from compel import Compel
from GreenScreenRemover import remove_green_screen_pil


# supress future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)


# load pipeline
pipeline = StableDiffusionControlNetPipeline.from_single_file(
    "SDCheckpoints/aingdiffusion_v13.safetensors",
    use_safetensors=True,
    local_files_only=True,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
'''
# load pipeline
pipeline = StableDiffusionPipeline.from_single_file(
    "SDCheckpoints/aingdiffusion_v13.safetensors",
    use_safetensors=True,
    local_files_only=True,
    torch_dtype=torch.float16
)
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_id = "stabilityai/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id,
                                                                torch_dtype=torch.float16)


upscaler.to("cuda")
upscaler.enable_xformers_memory_efficient_attention()

# override default scheduler with multistep scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# load embeddings
pipeline.load_textual_inversion("./SDCheckpoints/embedding/Aissist-supercharge-neg.pt", token="Aissist")
pipeline.load_textual_inversion("./SDCheckpoints/embedding/verybadimagenegative_v1.3.pt", token="verybadimagenegative_v1.3")


# load lora weights
pipeline.load_lora_weights("./SDCheckpoints/lora/", weight_name="GreenScreen_N.safetensors")
#pipeline = pipeline.to("cuda")

# Assuming 'pipeline' is your existing Stable Diffusion pipeline object
# Use a dictionary comprehension to include all components except 'controlnet'
filtered_components = {k: v for k, v in pipeline.components.items() if k != 'controlnet'}

# Initialize the Img2Img pipeline with the filtered components
img2img = StableDiffusionImg2ImgPipeline(**filtered_components)

compel = Compel(tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                truncate_long_prompts=False,
                device="cuda")


pipeline.enable_model_cpu_offload()

# enable efficient implementations using xformers for faster inference
pipeline.enable_xformers_memory_efficient_attention()

save_intermediate = True

# JSON structure
prompt_json = '''
{
  "character_base": "1girl",
  "hair": "short blond hair",
  "eyes": "red eyes",
  "face": "pretty childlike face, detailed face",
  "mood": "neutral face",
  "wearing": "black, egl, gothic lolita dress, black jacket with red crosses, black bows in hair",
  "image_quality": "intricate, visual novel style, beautiful, masterpiece, detailed eyes",
  "pose_reference": "waist_up_arms_down.png"
}
'''
prompt_data = json.loads(prompt_json)

# load control net pose reference
image_input = load_image(prompt_data["pose_reference"])
pose_reference = openpose(image_input)

if save_intermediate:
    pose_reference.save("pose_reference.png")

# Building the prompt using the specified format
character_base = prompt_data["character_base"]
hair = prompt_data["hair"]
eyes = prompt_data["eyes"]
face = prompt_data["face"]
mood = prompt_data["mood"]
wearing = prompt_data["wearing"]
image_quality = prompt_data["image_quality"]

background = "isolated on green background, solid background"
utility_instructions = "4K, high resolution, <lora:GreenScreen_N:1.5>"

prompt = f"{character_base}, {hair}, {eyes}, {face}, {background}, {mood}, {wearing}, {image_quality}, {utility_instructions}"
negative_prompt = "text, double image, (worst quality, low quality:1.4), (zombie, interlocked fingers), messed up eyes, extra arms, pornographic"

with torch.no_grad():
    prompt_embeds = compel(prompt)
    negative_prompt_embeds = compel(negative_prompt)

    [prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length([prompt_embeds,
                                                                                              negative_prompt_embeds])

g = torch.Generator(device="cuda")
g.manual_seed(1)

prompt = f"{character_base}, {hair},{eyes},{face},{background}, {mood}, {wearing}, {image_quality}, {utility_instructions}"
low_res_latents = pipeline(prompt_embeds=prompt_embeds,
                           negative_prompt_embeds=negative_prompt_embeds,
                           image=pose_reference,
                           guidance_scale=7,
                           num_inference_steps=10,
                           generator=g,
                           clip_skip=2,
                           height=768,
                           width=512,
                           output_type="latent"
                           ).images
if save_intermediate:
    with torch.no_grad():
        image = pipeline.decode_latents(low_res_latents)
        image = pipeline.numpy_to_pil(image)[0]
        image.save("SDControlNetTest0.png")

g.manual_seed(1)
upscaled_image_latents = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=1,
    generator=g,
    output_type="latent"
).images

if save_intermediate:
    with torch.no_grad():
        image = pipeline.decode_latents(upscaled_image_latents)
    image = pipeline.numpy_to_pil(image)[0]
    image.save("SDControlNetTest0-Upscaled.png")

upscaled_image_2 = img2img(prompt_embeds=prompt_embeds,
                           negative_prompt_embeds=negative_prompt_embeds,
                           image=upscaled_image_latents,
                           num_inference_steps=40,
                           guidance_scale=7,
                           generator=g,
                           clip_skip=2,
                           strength=0.5).images
upscaled_image_2[0].save("SDControlNetTest0-Upscaled_2.png")

# do chroma key (i.e., turn the green background into transparent)
green_removed_image = remove_green_screen_pil(upscaled_image_2[0], blur=4)
green_removed_image.save("SDControlNetTest0-Upscaled_2-GreenScreen.png")