from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer  # Import the GFPGAN model

class RealESRGANInferencer:
    def __init__(self, model_path='RealESRGAN_x4plus_anime_6B.pth', gfpgan_model_path=None, tile_size=0):
        # Initialize the Real-ESRGAN model
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(scale=4, model_path=model_path, model=self.model, tile=tile_size, half=False)

        # Initialize GFPGAN for face enhancement if a model path is provided
        self.face_enhancer = None
        if gfpgan_model_path:
            self.face_enhancer = GFPGANer(model_path=gfpgan_model_path, upscale=1, arch='clean', channel_multiplier=2)

    def run_inference_on_image(self, pil_img, enhance_faces=False):
        # Convert PIL Image to OpenCV format
        img = np.array(pil_img)
        img = img[:, :, ::-1].copy()  # Convert RGB to BGR

        # Enhance image using Real-ESRGAN
        output, _ = self.upsampler.enhance(img, outscale=4)

        # Enhance faces using GFPGAN
        if enhance_faces and self.face_enhancer:
            output, _, _ = self.face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)

        # Convert back to PIL Image
        output_pil = Image.fromarray(output[:, :, ::-1])  # Convert BGR to RGB
        return output_pil

