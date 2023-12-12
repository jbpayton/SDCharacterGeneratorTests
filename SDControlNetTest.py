import time

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionUpscalePipeline
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from realesrgan import RealESRGANer as RealESRGAN
from RealESRGANInferencer import RealESRGANInferencer


# supress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# upscaler = RealESRGANInferencer('./SDCheckpoints/upscaler/RealESRGAN_x4plus_anime_6B.pth')

# load embeddings
pipeline.load_textual_inversion("./SDCheckpoints/embedding/Aissist-supercharge-neg.pt", token="Aissist")
pipeline.load_textual_inversion("./SDCheckpoints/embedding/verybadimagenegative_v1.3.pt", token="verybadimagenegative_v1.3")

# override default scheduler with multistep scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# load lora weights
pipeline.load_lora_weights("./SDCheckpoints/lora/", weight_name="GreenScreen_N.safetensors")

pipeline = pipeline.to("cuda")

# enable efficient implementations using xformers for faster inference
pipeline.enable_xformers_memory_efficient_attention()

# load control net pose reference
image_input = load_image("poseref.png")
pose_reference = openpose(image_input)
pose_reference.save("pose_reference.png")



character_distinctives = "1girl, short brown hair. blue eyes"
background = "isolated on green background"
mood = "excited face"
wearing = "white lab coat, white shirt, black tie, black pants"
image_quality = "detailed eyes, beautiful, best quality, masterpiece, 4k, high resolution <lora:GreenScreen_N:1.5>"

negative_prompt = "verybadimagenegative_v1.3, Aissist, immodest"

g = torch.Generator(device="cuda")
#torch.use_deterministic_algorithms(True)

g.manual_seed(1)
prompt = f"{character_distinctives}, {background}, {mood}, {wearing}, {image_quality}"
image = pipeline(prompt=prompt,
                 negative_prompt=negative_prompt,
                 image=pose_reference,
                 guidance_scale=9,
                 num_inference_steps=20,
                 generator=g,
                 clip_skip=2,
                 height=768,
                 width=512
                 ).images[0]
image.save("SDControlNetTest0.png")
#upscaled_image = upscaler.run_inference_on_image(image)
#upscaled_image.save("SDControlNetTest0-Upsampled.png")

g.manual_seed(1)
mood = "angry face"
prompt = f"{character_distinctives}, {background}, {mood}, {wearing}, {image_quality}"
image = pipeline(prompt=prompt,
                 negative_prompt=negative_prompt,
                 image=pose_reference,
                 guidance_scale=9,
                 num_inference_steps=20,
                 generator=g,
                 clip_skip=2,
                 height=768,
                 width=512
                 ).images[0]
image.save("SDControlNetTest1.png")
#upscaled_image = upscaler.run_inference_on_image(image)
#upscaled_image.save("SDControlNetTest1-Upsampled.png")


