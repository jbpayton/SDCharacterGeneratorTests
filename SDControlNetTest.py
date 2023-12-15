import time

from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, ControlNetModel, \
    EulerAncestralDiscreteScheduler, StableDiffusionLatentUpscalePipeline, DPMSolverMultistepScheduler, \
    StableDiffusionImg2ImgPipeline
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image


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

pipeline.enable_model_cpu_offload()

# enable efficient implementations using xformers for faster inference
pipeline.enable_xformers_memory_efficient_attention()

# load control net pose reference
image_input = load_image("poseref.png")
pose_reference = openpose(image_input)
pose_reference.save("pose_reference.png")

character_distinctives = "1girl, short brown hair. blue eyes, detailed face"
background = "isolated on green background"
mood = "excited face"
wearing = "white lab coat, white shirt, black tie, black pants"
image_quality = "intricate, visual novel style, beautiful, masterpiece, detailed eyes, 4K, high resolution <lora:GreenScreen_N:1.5>"

negative_prompt = "text, double image, (worst quality, low quality:1.4), (zombie, interlocked fingers), messed up eyes, extra arms, slutty"
#negative_prompt = "verybadimagenegative_v1.3, immodest"


g = torch.Generator(device="cuda")

g.manual_seed(1)
prompt = f"{character_distinctives}, {background}, {mood}, {wearing}, {image_quality}"
low_res_latents = pipeline(prompt=prompt,
                           negative_prompt=negative_prompt,
                           image=pose_reference,
                           guidance_scale=7,
                           num_inference_steps=20,
                           generator=g,
                           clip_skip=2,
                           height=768,
                           width=512,
                           output_type="latent"
                           ).images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]
image.save("SDControlNetTest0.png")

g.manual_seed(1)
upscaled_image_latents = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=g,
    output_type="latent"
).images

with torch.no_grad():
    image = pipeline.decode_latents(upscaled_image_latents)
image = pipeline.numpy_to_pil(image)[0]
image.save("SDControlNetTest0-Upscaled.png")

upscaled_image_2 = img2img(prompt=prompt,
                           negative_prompt=negative_prompt,
                           image=upscaled_image_latents,
                           num_inference_steps=20,
                           guidance_scale=7,
                           generator=g,
                           clip_skip=2,
                           strength=0.5).images
upscaled_image_2[0].save("SDControlNetTest0-Upscaled_2.png")


'''
g.manual_seed(1)
mood = "angry face"
prompt = f"{character_distinctives}, {background}, {mood}, {wearing}, {image_quality}"
low_res_latents = pipeline(prompt=prompt,
                           negative_prompt=negative_prompt,
                           image=pose_reference,
                           guidance_scale=5,
                           num_inference_steps=20,
                           generator=g,
                           clip_skip=2,
                           height=768,
                           width=512,
                           output_type="latent"
                           ).images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]
image.save("SDControlNetTest1.png")

g.manual_seed(1)
upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=5,
    generator=g,
).images[0]
upscaled_image.save("SDControlNetTest1-Upscaled.png")
'''