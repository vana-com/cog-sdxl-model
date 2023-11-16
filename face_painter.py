import mediapipe
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import time
import os
class ImageArea:
    def __init__(self, x: int, y:int , w:int, h :int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class FacePainter:
    class Mask:
        def __init__(self, face_area: ImageArea, max_width: int, max_height: int, zoom: int=640):
            self.face_area = face_area
            self.max_width = max_width
            self.max_height = max_height
            self.zoom = zoom

        @property
        def mask_area(self):
            mask_dimension = int(self.face_area.h * 1.5)
            mask_dimension = int( mask_dimension + ( 48 - mask_dimension % 8))
            if mask_dimension > self.zoom:
                mask_dimension = self.zoom

            return ImageArea(
                int(min(max(self.face_area.x - ((mask_dimension - self.face_area.w) / 2), 0), self.max_height - mask_dimension)),
                int(min(max(self.face_area.y - ((mask_dimension - self.face_area.h) / 2), 0), self.max_height - mask_dimension)),
                int(min(mask_dimension, self.max_width)),
                int(min(mask_dimension, self.max_height)),
            )
        
        @property
        def zoomed_size(self):
            if self.mask_area.w > self.zoom:
                return self.mask_area.w
            return self.mask_area.w * int(self.zoom / self.mask_area.w)
        
        @property
        def mask(self):
            # Create a black and white mask
            mask_img = np.ones((self.max_width, self.max_height), dtype=np.uint8)
            mask_img[self.face_area.y : self.face_area.y + self.face_area.h, self.face_area.x : self.face_area.x + self.face_area.w] = 255
            mask_img = Image.fromarray(mask_img)
            mask_img = mask_img.crop((self.mask_area.x, self.mask_area.y, self.mask_area.x + self.mask_area.w, self.mask_area.y + self.mask_area.h))
            if self.mask_area.w < self.zoom:
                mask_img = mask_img.resize([self.zoomed_size, self.zoomed_size])
            return mask_img
        
        def image(self, base_image: Image.Image):
            print('image: ', base_image)
            base_image = base_image.copy()
            base_image = base_image.crop((self.mask_area.x, self.mask_area.y, self.mask_area.x + self.mask_area.w, self.mask_area.y + self.mask_area.h))
            if self.mask_area.w < self.zoom:
                base_image = base_image.resize([self.zoomed_size, self.zoomed_size])
            return base_image

    def __init__(self, inpaint_pipe) -> None:
        self.face_detection = mediapipe.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.10
        )
        self.inpaint_pipe = inpaint_pipe
        self.masks = []

    def _create_masks(self, face_detect_image: Image.Image, max_face_mask_size : int=256):
        self.masks = []
        image_arr = np.asarray(face_detect_image)
        results = self.face_detection.process(image_arr)
        
        if results.detections:
            for detection in results.detections:
                x_min = int(
                    detection.location_data.relative_bounding_box.xmin
                    * image_arr.shape[1]
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin
                    * image_arr.shape[0]
                )
                width = int(
                    detection.location_data.relative_bounding_box.width
                    * image_arr.shape[1]
                )
                width = width - width % 2
                
                height = int(
                    detection.location_data.relative_bounding_box.height
                    * image_arr.shape[0]
                )
                height = height - height % 2

                # Add padding
                padding = 0
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                width = min(width + 2 * padding, image_arr.shape[1] - x_min)
                height = min(height + 2 * padding, image_arr.shape[0] - y_min)

                self.masks.append(
                    self.Mask(
                    face_area=ImageArea(x_min, y_min, width, height),
                    max_width=image_arr.shape[0],
                    max_height=image_arr.shape[1],
                    )
                )

    def _paste(self, new_image, area: ImageArea):

        new_image = new_image.resize([area.w, area.h])
        paste_mask = Image.new('L', new_image.size, 255)
        gradient_size = 15
        for i in range(gradient_size):
            alpha = 255 * (i + 1) // (gradient_size + 1)
            ImageDraw.Draw(paste_mask).rectangle(
                [i, i, paste_mask.size[0] - 1 - i, paste_mask.size[1] - 1 - i], outline=alpha, width=1
            )
        self.image.paste(new_image, (area.x, area.y), paste_mask)

    def paint_faces(
        self, 
        image, 
        prompt,
        negative_prompt,
        guidance_scale=4.5, 
        lora_paths=[], 
        face_detect_image=None,
        save_working_images=False,
        max_face_size=None
    ):
        
        # self.inpaint_pipe.enable_vae_slicing = False
        self.image = image.copy()
        if face_detect_image is None:
            face_detect_image = self.image
        self._create_masks(face_detect_image)

        if lora_paths:
            print('got lora paths', lora_paths)
            # prompt = 'a bright green square, bright green square'
            # prompt = f'"<1>" face. {prompt.replace("<1>", "").replace("<2>", "")}'
        else:
            print('no lora paths', lora_paths)
            pass
            # prompt = 'a bright green square, bright green square'
            # prompt = f' face. {prompt.replace("<1>", "").replace("<2>", "")}'
        for i, mask in enumerate(
            sorted(self.masks, key=lambda x: x.face_area.w * x.face_area.h, reverse=True)
        ):
            if max_face_size is not None and mask.face_area.w > max_face_size:
                continue
            if lora_paths and i < len(lora_paths):
                # inpaint_pipe
                # self.pipeline_manager.lora_paths = [lora_paths[i % len(lora_paths)]]
                print("lora_paths and i < len(lora_paths)")
            else:
                print("else")
                # inpaint_pipe
                # self.pipeline_manager.lora_paths = None
            if save_working_images:
                os.makedirs("./tmp", exist_ok=True)
                self.image.save(f"./tmp/{time.time()}_working_{i}.png")
                mask.mask.save(f"./tmp/{time.time()}_mask_{i}.png")
                mask.image(self.image).save(f"./tmp/{time.time()}_image_{i}.png")

            # Generate with new model
            
            # In-paint the image
            output = self.inpaint_pipe(
                # TODO(anna) these args may be off
                # inpaint_pipe(**inpainting_args, **sdxl_kwargs)
                prompt=prompt,
                negative_prompt=negative_prompt, # "frame, mask, surgical, ui, ugly, distorted eyes, deformed iris, toothless, squint, deformed iris, deformed pupils, low quality, jpeg artifacts, ugly, mutilated",
                image=mask.image(self.image),
                mask_image=mask.mask,
                guidance_scale=guidance_scale,
                num_inference_steps=10 + int(4 * guidance_scale),
                width=mask.zoomed_size,
                height=mask.zoomed_size,
            )

            self._paste(output.images[0], mask.mask_area)
            #self._paste(output["images"][0], mask.mask_area)
        return self.image