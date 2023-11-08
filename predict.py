import json
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor
from cryptography.fernet import Fernet
from face_painter import FacePainter

import json
import requests
from io import BytesIO
import tarfile
import torch
from PIL import Image
import shutil
import math


from dataset_and_utils import TokenEmbeddingsHandler



SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-fix-1.0.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)

# Currently using symmetric encryption
# But we should use asymmetric encryption
# This would be NODE_PRIVATE_KEY
NODE_KEY = b'xOBP_J1N6J1y7a0I9MHJ7VbVbV_a7BzI2s4O5pLKZKU='

class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_output(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights_url, pipe):
        # Get the TAR archive content
        weights_tar_data = requests.get(weights_url).content

        with tarfile.open(fileobj=BytesIO(weights_tar_data), mode='r') as tar_ref:
            tar_ref.extractall("trained-model")

        local_weights_cache = "./trained-model"

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. Assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            sd = pipe.unet.state_dict()
            sd.update(new_unet_params)
            pipe.unet.load_state_dict(sd)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet = pipe.unet
            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]

                module = LoRAAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=name_rank_map[name],
                )
                unet_lora_attn_procs[name] = module.to("cuda")

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True            
            
                        
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_model = False

        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)
        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        # TODO(anna) Could we do more in this setup step to make load times faster?

        print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        # Copy the image to a temporary location
        shutil.copyfile(path, "/tmp/image.png")
        
        # Open the copied image
        img = Image.open("/tmp/image.png")
        
        # Calculate the new dimensions while maintaining aspect ratio
        width, height = img.size
        new_width = math.ceil(width / 64) * 64
        new_height = math.ceil(height / 64) * 64
        
        # Resize the image if needed
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Convert the image to RGB mode
        img = img.convert("RGB")
        
        return img

    @torch.inference_mode()
    def predict(
        self,
        Lora_url: str = Input(
            description="Load Lora model",
        ),
        prompt: str = Input(
            description="Input prompt. If encryptedInput is true, this should be encrypted",
            default="An TOK riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        enable_face_inpainting: bool = Input(
            description="Inpaint small faces to improve resolution. Will slow down inference.",
            default=False,
        ),
        max_face_inpaint_size: int = Input(
            description="Max size of face to inpaint. Recommended: 135-400. If it's too high, may get weird portraits.",
            default=300,
        ),
        encryptedInput: bool = Input(
            description="Whether prompt is encrypted",
            default=False,
        ),
        encryptedOutput: bool = Input(
            description="Whether image output should be encrypted",
            default=False,
        ),
        userPublicKey: str = Input(
            description="The public key of the user, used to encrypt image. Only used if encryptedOutput is on",
            default='4KRWKwyJCi5RyDQ10YmTUL4yS0XkyBFpr_BeB0XGQlM=',
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=768,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=10,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=35
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=False,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
    ) -> List[Path]:
        lora = True
        if lora == True :
            self.is_lora = True
            print("Loading sdxl txt2img pipeline...")
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                SDXL_MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        
            self.load_trained_weights(Lora_url, self.txt2img_pipe)
            self.txt2img_pipe.to("cuda")
            self.is_lora = True

            print("Loading SDXL img2img pipeline...")
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            self.img2img_pipe.to("cuda")

            print("Loading SDXL inpaint pipeline...")
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            self.inpaint_pipe.to("cuda")

            print("Initializing face painter...")
            self.face_painter = FacePainter(self.inpaint_pipe)

            print("Loading SDXL refiner pipeline...")

            print("Loading refiner pipeline...")
            self.refiner = DiffusionPipeline.from_pretrained(
                "refiner-cache",
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                vae=self.txt2img_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner.to("cuda")


        else :
            print("Loading sdxl txt2img pipeline...")
            self.txt2img_pipe = DiffusionPipeline.from_pretrained(
                SDXL_MODEL_CACHE,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.is_lora = False

            self.txt2img_pipe.to("cuda")

            print("Loading SDXL img2img pipeline...")
            self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            self.img2img_pipe.to("cuda")

            print("Loading SDXL inpaint pipeline...")
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
                vae=self.txt2img_pipe.vae,
                text_encoder=self.txt2img_pipe.text_encoder,
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                tokenizer=self.txt2img_pipe.tokenizer,
                tokenizer_2=self.txt2img_pipe.tokenizer_2,
                unet=self.txt2img_pipe.unet,
                scheduler=self.txt2img_pipe.scheduler,
            )
            self.inpaint_pipe.to("cuda")
            print("Loading refiner pipeline...")
            self.refiner = DiffusionPipeline.from_pretrained(
                "refiner-cache",
                text_encoder_2=self.txt2img_pipe.text_encoder_2,
                vae=self.txt2img_pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner.to("cuda")

        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")

        # Decrypt the encrypted prompt
        if encryptedInput:
            cipher_suite = Fernet(NODE_KEY)
            prompt = cipher_suite.decrypt(prompt).decode()
            print(f"Decrypted prompt: {prompt}")


        if image and mask:
            print("inpainting mode")
            loaded_image = self.load_image(image)
            sdxl_kwargs["image"] = loaded_image
            sdxl_kwargs["mask_image"] = self.load_image(mask)
            sdxl_kwargs["strength"] = prompt_strength

            # Get the dimensions (height and width) of the loaded image
            image_width, image_height = loaded_image.size

            sdxl_kwargs["target_size"] = (image_width, image_height)
            sdxl_kwargs["original_size"] = (image_width, image_height)

            pipe = self.inpaint_pipe
        elif image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
            pipe = self.txt2img_pipe

        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs)

        if refine in ["expert_ensemble_refiner", "base_image_refiner"]:
            refiner_kwargs = {
                "image": output.images,
            }

            if refine == "expert_ensemble_refiner":
                refiner_kwargs["denoising_start"] = high_noise_frac
            if refine == "base_image_refiner" and refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        if enable_face_inpainting:
            print("inpainting mode")
            # May need to download lora
            inpainted_images = [self.face_painter.paint_faces(
                    image, 
                    guidance_scale=guidance_scale, 
                    lora_paths=[Lora_url],
                    prompt='a bright green square', # prompt[i],
                    save_working_images=False,
                    max_face_size = max_face_inpaint_size)
                                            for i, image in enumerate(output.images)]
            output = inpainted_images

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        output_paths = []
        for i, image in enumerate(output):
            if encryptedOutput:
                # TODO need to switch to asymmetric encryption 
                user_cipher_suite = Fernet(userPublicKey.encode())
                # TODO need to check security assumptions of writing file temporarily
                # Save the image in a standard format (e.g., PNG) to a temporary file
                temp_image_path = f"/tmp/temp_image-{i}.png"
                image.save(temp_image_path, 'PNG')  # Assuming 'image' is a PIL Image object

                # Read the saved image file in binary mode
                with open(temp_image_path, 'rb') as file:
                    image_data = file.read()

                # Encrypt the binary data of the whole file
                encryptedImage = user_cipher_suite.encrypt(image_data)

                # Save the encrypted data to the final output file
                output_path = f"/tmp/encrypted_out-{i}.enc"
                with open(output_path, 'wb') as file_out:
                    file_out.write(encryptedImage)

                # Append the path for record-keeping
                output_paths.append(Path(output_path))

                # Delete the temporary image file
                os.remove(temp_image_path)
            else: 
                output_path = f"/tmp/out-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

        return output_paths
