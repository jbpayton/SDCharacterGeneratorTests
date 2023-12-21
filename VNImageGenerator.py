import json, os
from datetime import datetime

import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionLatentUpscalePipeline, \
    DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionInpaintPipeline

from controlnet_aux.processor import Processor
from diffusers.utils import load_image
from compel import Compel
from GreenScreenRemover import remove_green_screen_pil
from PIL import Image
from FaceMasker import create_face_mask_pil

# supress future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class VNImageGenerator:
    def __init__(self, pipeline_path, upscaler_model_id, device='cuda', generate_transparent_background=True, output_folder="./output"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lora_weights = None

        # init to false, will be set to true if we load the green screen weights
        self.background_transparency = False

        # Load ControlNet and Openpose models
        self.openpose = Processor("openpose_full")
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)

        # Load the main pipeline
        self.pipeline = StableDiffusionControlNetPipeline.from_single_file(
            pipeline_path,
            use_safetensors=True,
            local_files_only=True,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        self.pipeline.safety_checker = lambda images, **kwargs: (images, False)

        # Initialize upscaler
        self.upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(upscaler_model_id,
                                                                             torch_dtype=torch.float16)

        self.upscaler.safety_checker = lambda images, **kwargs: (images, False)

        self.upscaler.to(device)
        self.upscaler.enable_xformers_memory_efficient_attention()

        # Modify the scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)

        # load green screen weights
        self.enable_tranparent_background(generate_transparent_background)

        # Initialize compel and img2img
        self.compel = Compel(tokenizer=self.pipeline.tokenizer, text_encoder=self.pipeline.text_encoder,
                             truncate_long_prompts=False, device=device)

        filtered_components = {k: v for k, v in self.pipeline.components.items() if k != 'controlnet'}
        self.img2img = StableDiffusionImg2ImgPipeline(**filtered_components)
        self.img2img.safety_checker = lambda images, **kwargs: (images, None)

        self.inpainter = StableDiffusionInpaintPipeline(**filtered_components)
        self.inpainter.safety_checker = lambda images, **kwargs: (images, False)

        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_model_cpu_offload()

        # if it doesnt exist, make an output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

    def enable_tranparent_background(self, enable):
        if enable and self.background_transparency == False:
            self.pipeline.load_lora_weights("./SDCheckpoints/lora/", weight_name="GreenScreen_N.safetensors")
            self.background_transparency = True
        elif enable == False and self.background_transparency == True:
            self.pipeline.unload_lora_weights()
            # make sure our other lora weights are loaded
            if self.lora_weights is not None:
                self.load_configurations(lora_weights=self.lora_weights)
            self.background_transparency = False
        else:
            print("Background transparency is already set to the desired value.")

    def load_configurations(self, textual_inversions=None, lora_weights=None):
        for path, token in textual_inversions:
            self.pipeline.load_textual_inversion(path, token=token)

        # save the list of lora weights
        self.lora_weights = lora_weights

        for path, weight_name in lora_weights:
            self.pipeline.load_lora_weights(path, weight_name=weight_name)

    def change_facial_expression(self, latents, prompt_data, expression, save_intermediate=False):
        with torch.no_grad():
            image = self.pipeline.decode_latents(latents)
            image = self.pipeline.numpy_to_pil(image)[0]

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prompt_data["expression"] = expression
        height = prompt_data.get("height", 768)
        width = prompt_data.get("width", 512)

        # first, get the non-prompt specific data from the prompt_data structure
        # get seed from JSON
        seed = prompt_data.get("seed", -1)

        mask = create_face_mask_pil(image, debug=save_intermediate)

        # if this json contains the character_base key, we need to create a charcter prompt
        negative_prompt, prompt = self.build_character_prompt(prompt_data)

        with torch.no_grad():
            prompt_embeds = self.compel(prompt)
            negative_prompt_embeds = self.compel(negative_prompt)

            [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds])

        g = torch.Generator(device=self.device)
        if seed == -1:
            g.seed()
            seed = g.initial_seed()
        else:
            g.manual_seed(seed)

        facial_change = self.inpainter(prompt_embeds=prompt_embeds,
                                       negative_prompt_embeds=negative_prompt_embeds,
                                       image=image,
                                       mask_image=mask,
                                       num_inference_steps=20,
                                       guidance_scale=10,
                                       generator=g,
                                       clip_skip=2,
                                       height=height,
                                       width=width,
                                       strength=0.8,
                                       output_type="latent").images

        if save_intermediate:
            with torch.no_grad():
                image = self.pipeline.decode_latents(facial_change)
                image = self.pipeline.numpy_to_pil(image)[0]
                image.save(f"{self.output_folder}/VNImageGenerator-{timestamp}-S-FaceChange-.png")
        return facial_change

    def text_to_image(self, prompt_data, save_intermediate=False):
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        pose_reference = prompt_data.get("pose_reference", None)
        if pose_reference is not None:
            image_input = load_image(prompt_data["pose_reference"])
            pose_reference = self.openpose(image_input, to_pil=True)

            if save_intermediate:
                pose_reference.save(f"{self.output_folder}/VNImageGenerator-{timestamp}-0-pose_reference.png")
        else:
            pose_reference = Image.new('RGB', (512, 512))

        # first, get the non-prompt specific data from the prompt_data structure
        # get seed from JSON
        seed = prompt_data.get("seed", -1)

        # if this json contains the character_base key, we need to create a charcter prompt
        if "character_base" in prompt_data:
            negative_prompt, prompt = self.build_character_prompt(prompt_data, save_intermediate=save_intermediate)
            height = prompt_data.get("height", 768)
            width = prompt_data.get("width", 512)
        elif "scene_base" in prompt_data:
            negative_prompt, prompt = self.build_scene_prompt(prompt_data)
            height = prompt_data.get("height", 512)
            width = prompt_data.get("width", 768)
        else:
            negative_prompt = prompt_data.get("negative_prompt", "")
            prompt = prompt_data.get("prompt", "")
            height = prompt_data.get("height", 512)
            width = prompt_data.get("width", 512)

        with torch.no_grad():
            prompt_embeds = self.compel(prompt)
            negative_prompt_embeds = self.compel(negative_prompt)

            [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds])

        g = torch.Generator(device=self.device)
        if seed == -1:
            g.seed()
            seed = g.initial_seed()
        else:
            g.manual_seed(seed)

        low_res_latents = self.pipeline(prompt_embeds=prompt_embeds,
                                        negative_prompt_embeds=negative_prompt_embeds,
                                        image=pose_reference,
                                        guidance_scale=7,
                                        num_inference_steps=10,
                                        generator=g,
                                        clip_skip=2,
                                        height=height,
                                        width=width,
                                        output_type="latent"
                                        ).images
        if save_intermediate:
            with torch.no_grad():
                image = self.pipeline.decode_latents(low_res_latents)
                image = self.pipeline.numpy_to_pil(image)[0]
                image.save(f"{self.output_folder}/VNImageGenerator-{timestamp}-1-Initial-.png")

        return low_res_latents

    def latent_upscale_and_refine(self, low_res_latents, prompt_data, expression_override=None,
                                  save_intermediate=False):
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # first, get the non-prompt specific data from the prompt_data structure
        # get seed from JSON
        seed = prompt_data.get("seed", -1)

        if expression_override is not None:
            prompt_data["expression"] = expression_override

        # if this json contains the character_base key, we need to create a charcter prompt
        if "character_base" in prompt_data:
            negative_prompt, prompt = self.build_character_prompt(prompt_data)
        elif "scene_base" in prompt_data:
            negative_prompt, prompt = self.build_scene_prompt(prompt_data)
        else:
            negative_prompt = prompt_data.get("negative_prompt", "")
            prompt = prompt_data.get("prompt", "")
        g = torch.Generator(device=self.device)
        if seed == -1:
            g.seed()
            seed = g.initial_seed()
        else:
            g.manual_seed(seed)
        upscaled_image_latents = self.upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=g,
            output_type="latent"
        ).images
        if save_intermediate:
            with torch.no_grad():
                image = self.pipeline.decode_latents(upscaled_image_latents)
            image = self.pipeline.numpy_to_pil(image)[0]
            image.save(f"{self.output_folder}/VNImageGenerator-{timestamp}-2-LatentUpscaled.png")
        with torch.no_grad():
            prompt_embeds = self.compel(prompt)
            negative_prompt_embeds = self.compel(negative_prompt)

            [prompt_embeds, negative_prompt_embeds] = self.compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds])
        upscaled_image_2 = self.img2img(prompt_embeds=prompt_embeds,
                                        negative_prompt_embeds=negative_prompt_embeds,
                                        image=upscaled_image_latents,
                                        num_inference_steps=20,
                                        guidance_scale=7,
                                        generator=g,
                                        clip_skip=2,
                                        strength=0.35).images
        if save_intermediate:
            upscaled_image_2[0].save(f"{self.output_folder}/VNImageGenerator-{timestamp}-3-Img2Img.png")
        final_image = upscaled_image_2[0]
        # Remove green screen if it was used
        if self.background_transparency:
            final_image = remove_green_screen_pil(final_image, blur=3, threshold=.1,debug=save_intermediate)
            if save_intermediate:
                final_image.save(f"{self.output_folder}/VNImageGenerator-{timestamp}-4-GreenScreen.png")
        return final_image

    def build_character_prompt(self, prompt_data, save_intermediate=False):
        # Building the prompt
        character_base = prompt_data["character_base"]
        age = prompt_data.get("age", "young adult")
        hair = prompt_data["hair"]
        eyes = prompt_data["eyes"]
        face = prompt_data["face"]
        body = prompt_data.get("body", "average")
        expression = prompt_data["expression"]
        wearing = prompt_data["wearing"]
        image_quality = prompt_data["image_quality"]
        # optional background (will be ignored if background transparency is enabled)
        background = prompt_data.get("background", "uniform background")
        negative_prompt = prompt_data.get("negative_prompt", None)
        utility_instructions = "Anime, 4K, high resolution, clean pen outline" + (
            ", <lora:GreenScreen_N:1.5>" if self.background_transparency else "")
        if self.background_transparency:
            background = "isolated on solid green background"
        prompt = f"{character_base}, {background}, {age}, {hair}, {eyes}, {face}, {body}, ((({expression}))), {wearing}, {image_quality}, {utility_instructions}"
        if negative_prompt is None:
            negative_prompt = "text, double image, (worst quality, low quality:1.4), (zombie, interlocked fingers), messed up eyes, extra arms, pornographic"
        if save_intermediate:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            #save the prompt and negative prompt to a file
            with open(f"{self.output_folder}/VNImageGenerator-{timestamp}-Prompt.txt", "w") as f:
                f.write(prompt)
            with open(f"{self.output_folder}/VNImageGenerator-{timestamp}-NegativePrompt.txt", "w") as f:
                f.write(negative_prompt)
        return negative_prompt, prompt

    def build_scene_prompt(self, prompt_data):
        # building the prompt
        scene_base = prompt_data["scene_base"]
        scene = prompt_data["scene_description"]
        image_quality = prompt_data["image_quality"]
        negative_prompt = prompt_data.get("negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = "(worst quality, low quality:1.4)"
        utility_instructions = "4K, high resolution"
        prompt = f"{scene_base}, {scene}, {image_quality}, {utility_instructions}"
        return negative_prompt, prompt

    def generate_character_images(self, character_prompt_json, save_intermediate=False):
        # Generate image
        # JSON structure
        base_prompt_json = '''
                {
                  "character_base": "girl",
                  "age": "18",
                  "hair": "short blond hair",
                  "eyes": "blue eyes",
                  "face": "pretty childlike face, detailed face",
                  "body": "slim, slender, petite, small chest",
                  "expression": "neutral",
                  "wearing": "traditional japanese clothing, pink kimono, yukata",
                  "image_quality": "intricate, beautiful, masterpiece, detailed eyes",
                  "pose_reference": "pose_references/waist_up_arms_down.png",
                  "seed": 1
                }
                '''
        prompt_data = json.loads(base_prompt_json)

        # override the default values with the ones from the json
        prompt_data.update(character_prompt_json)

        character_name = prompt_data.get("name", "character")

        # change spaces to underscores
        character_name = character_name.replace(" ", "_")

        # create a folder for the character
        os.makedirs(f"{self.output_folder}/{character_name}", exist_ok=True)
        character_image_folder = f"{self.output_folder}/{character_name}"

        # save the prompt data to the character folder
        with open(f"{character_image_folder}/prompt.json", "w") as f:
            json.dump(prompt_data, f)

        low_res_latents = self.text_to_image(prompt_data,
                                             save_intermediate=save_intermediate)
        character_img = self.latent_upscale_and_refine(low_res_latents,
                                                       prompt_data,
                                                       save_intermediate=save_intermediate)
        #character_img.save(f"{character_image_folder}/dialogue-happy.png")

        facial_expressions = ["crying", "furious, angry", "smiling", "neutral", "disappointed, let down",
                              "blushing, very embarrassed", "terrified, scared", "laughing", "yelling with open mouth"]

        short_names = ["cry", "angry", "smile", "neutral", "disappointed",
                              "blush", "scared", "laugh", "yell"]

        for facial_expression, short_name in zip(facial_expressions, short_names):
            changed_latents = self.change_facial_expression(low_res_latents, prompt_data, facial_expression, save_intermediate=save_intermediate)
            character_img = self.latent_upscale_and_refine(changed_latents,
                                                           prompt_data,
                                                           expression_override=facial_expression)
            character_img.save(f"{character_image_folder}/dialogue-{short_name}.png")


if __name__ == '__main__':

    chara_test = True
    CG_test = True
    scene_test = True
    composition_test = False

    # Usage
    pipeline_path = "SDCheckpoints/aingdiffusion_v13.safetensors"
    upscaler_model_id = "stabilityai/sd-x2-latent-upscaler"
    generator = VNImageGenerator(pipeline_path, upscaler_model_id)

    character_img = None
    scene_img = None

    if chara_test:
        # Generate image
        # JSON structure
        prompt_json = '''
        {
          "character_base": "girl",
          "age": "17",
          "name": "Sakura Testcharacter",
          "hair": "Short blonde hair with red streak",
          "eyes": "very large blue eyes",
          "face": "pretty face, detailed face",
          "body": "normal build, medium chest",
          "wearing": "red fantasy clothing",
          "seed": -1
        }
        '''
        prompt_data = json.loads(prompt_json)
        generator.generate_character_images(prompt_data, save_intermediate=True)

    if CG_test:
        prompt_json = '''
            {
              "character_base": "1girl",
              "hair": "short blond hair",
              "eyes": "red wide eyes, crazy eyes, scary eyes",
              "face": "pretty childlike face, detailed face",
              "expression": "crazed yandere smile, open mouth",
              "background": "Haunted Vampire Castle",
              "wearing": "black, egl, gothic lolita dress, black jacket with red crosses, black bows in hair",
              "image_quality": "intricate, visual novel style, beautiful, masterpiece, detailed eyes",
              "width": 768,
              "height": 512
            }
            '''
        prompt_data = json.loads(prompt_json)
        generator.enable_tranparent_background(False)
        low_res_latents = generator.text_to_image(prompt_data)
        image = generator.latent_upscale_and_refine(low_res_latents, prompt_data, save_intermediate=False)
        image.save(f"{generator.output_folder}/VNImageGenerator-CG-Final.png")

    if scene_test:
        prompt_json = '''
                {
                  "scene_base": "outdoor scene",
                  "scene_description": "by the seaside, in modern japan",
                  "image_quality": "intricate, visual novel style, beautiful, masterpiece"
                }
                '''
        prompt_data = json.loads(prompt_json)
        generator.enable_tranparent_background(False)
        low_res_latents = generator.text_to_image(prompt_data)
        scene_img = generator.latent_upscale_and_refine(low_res_latents, prompt_data, save_intermediate=True)
        scene_img.save(f"{generator.output_folder}/VNImageGenerator-background-Final.png")

    if composition_test:
        if character_img is not None and scene_img is not None:
            print("Compositing")
            # put the character on the scene, in the middle, with a bit of scaling and padding
            # scale down the character image to 1/4th of the scene image
            scene_img.paste(character_img, (
                scene_img.width // 2 - character_img.width // 2, scene_img.height // 2 - character_img.height // 2),
                            character_img)
            scene_img = scene_img.resize((scene_img.width * 2, scene_img.height * 2), resample=Image.LANCZOS)
            scene_img.save(f"{generator.output_folder}/VNImageGenerator-Composition-Final.png")
            scene_img.show()
