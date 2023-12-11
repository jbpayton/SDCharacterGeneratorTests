import time

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import transformers
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


# supress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

repo_id = "runwayml/stable-diffusion-v1-5"

openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)


pipeline = StableDiffusionControlNetPipeline.from_single_file(
    "SDCheckpoints/aingdiffusion_v13.safetensors",
    use_safetensors=True,
    local_files_only=True,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipeline = pipeline.to("cuda")

# enable efficient implementations using xformers for faster inference

# load control net pose reference
image_input = load_image("poseref.png")
pose_reference = openpose(image_input)
pose_reference.save("pose_reference.png")



character_distinctives = "A cute brown haired girl with short hair. with blue eyes"
mood = "excited"
wearing = "white lab coat, white shirt, black tie, black pants, black stockings, black shoes"
background = "solid color blue background"
camera_angle = "from the front"
camera_direction = "looking forward"
image_quality = "visual novel character, best quality, 4K, high resolution"

negative_prompt = "(worst quality, low quality:1.4), background, (zombie, interlocked fingers), extra arms"

g = torch.Generator(device="cuda")
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

g.manual_seed(0)
prompt = f"{character_distinctives}, {mood}, {wearing}, {background}, {camera_angle}, {camera_direction}, {image_quality}"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=pose_reference, guidance_scale=8,
                 num_inference_steps=40, generator=g).images[0]
image.save("SDControlNetTest0.png")

g.manual_seed(0)
mood = "angry"
prompt = f"{character_distinctives}, {mood}, {wearing}, {background}, {camera_angle}, {camera_direction}, {image_quality}"
image = pipeline(prompt=prompt, negative_prompt=negative_prompt, image=pose_reference, guidance_scale=8,
                 num_inference_steps=40, generator=g).images[0]
image.save("SDControlNetTest1.png")


