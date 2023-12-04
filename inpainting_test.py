from face_painter import FacePainter

from PIL import Image, ImageDraw, ImageFont
# Import the FacePlsainter class and other dependencies here

def mock_inpaint_pipe(cross_attention_kwargs, seed, prompt, negative_prompt, image, mask_image, guidance_scale, num_inference_steps, width, height):
    # Create a red image of the specified size
    green_image = Image.new("RGB", (width, height), (0, 255, 0))

    # Return the output image in a dictionary
    return {"images": [green_image]}


def main():
    # Create a FacePainter instance with a mock or real inpaint_pipe
    face_painter = FacePainter(inpaint_pipe=mock_inpaint_pipe)

    # Load the image
    image_path = "out-1.png"
    image = Image.open(image_path)

    # Process the image
    processed_image = face_painter.paint_faces(image, 'a bright green square positive prompt', 'some negative prompt', save_working_images=True, gradient_percent_size=0.05)
    processed_image.save("processed_image-1.jpg")
    processed_image.show()

    # Load the image
    image_path = "out-2.png"
    image = Image.open(image_path)

    # Process the image
    processed_image = face_painter.paint_faces(image, 'a bright green square positive prompt', 'some negative prompt', save_working_images=True, gradient_percent_size=0.05)
    
    # Save or display the result
    processed_image.save("processed_image-2.jpg")
    processed_image.show()

if __name__ == "__main__":
    main()
