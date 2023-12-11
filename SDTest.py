import time

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import transformers
import torch

# supress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

repo_id = "runwayml/stable-diffusion-v1-5"

pipeline = StableDiffusionPipeline.from_single_file(
    "SDCheckpoints/aingdiffusion_v13.safetensors",
    use_safetensors=True,
    local_files_only=True,
    #scheduler=DPMSolverMultistepScheduler,
    torch_dtype=torch.float16
)


pipeline = pipeline.to("cuda")

character_distinctives = "A cute brown haired girl with short hair. with blue eyes"
mood = "excited"
wearing = "white lab coat, white shirt, black tie"
background = "(green solid background)"
camera_angle = "(view of head, torso and arms), from the front"
camera_direction = "looking at the camera"
image_quality = "best quality, 4K, high resolution"

negative_prompt = "(worst quality, low quality:1.4), (zombie, interlocked fingers), extra arms"

g = torch.Generator(device="cuda")
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

g.manual_seed(0)
prompt = f"{character_distinctives}, {mood}, {wearing}, {background}, {camera_angle}, {camera_direction}, {image_quality}"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=9, num_inference_steps=40, generator=g).images[0]
image.save("SDTest0.png")

g.manual_seed(0)
mood = "angry"
prompt = f"{character_distinctives}, {mood}, {wearing}, {background}, {camera_angle}, {camera_direction}, {image_quality}"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=9, num_inference_steps=40, generator=g).images[0]
image.save("SDTest1.png")


