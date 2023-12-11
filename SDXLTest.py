import time

from diffusers import StableDiffusionXLPipeline
import torch

# supress future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pipeline = StableDiffusionXLPipeline.from_single_file(
    "SDXLTCheckpoints/aingdiffusionXL_v01.safetensors", torch_dtype=torch.float16)

pipeline = pipeline.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."


image = pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=1).images[0]
image.save("SDXLTest.png")

# now, lets test the pipeline with different num_inference_steps (1-5)
for i in range(1, 3):
    # let's time it
    start_time = time.time()
    image = pipeline(prompt=prompt, guidance_scale=0.0, num_inference_steps=i).images[0]
    image.save(f"SDXLTest{i}.png")
    print(f"Time taken for {i} steps: {time.time() - start_time}")

