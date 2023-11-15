from face_painter import FacePainter

from PIL import Image, ImageDraw, ImageFont
# Import the FacePainter class and other dependencies here

def mock_inpaint_pipe(prompt, negative_prompt, image, mask_image, guidance_scale, num_inference_steps, width, height):
    # Create a red image of the specified size
    red_image = Image.new("RGB", (width, height), (255, 0, 0))

    # Create a new image from the mask_image to ensure it's the right format
    output_mask = mask_image.copy().convert("RGB")

    # Overlay the red image onto the mask
    output_mask.paste(red_image, (0, 0), mask_image.convert("L"))

    # Return the output image in a dictionary
    return {"images": [output_mask]}

def main():
    # Load the image
    image_path = "out-2.png"
    image = Image.open(image_path)

    # Create a FacePainter instance with a mock or real inpaint_pipe
    face_painter = FacePainter(inpaint_pipe=mock_inpaint_pipe)

    # Process the image
    processed_image = face_painter.paint_faces(image, 'a sample prompt', save_working_images=True)
    print(face_painter.masks)

    # Save or display the result
    processed_image.save("processed_image.jpg")
    processed_image.show()

if __name__ == "__main__":
    main()
