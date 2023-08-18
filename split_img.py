from PIL import Image


def split_image(image_path):
    # Load the image and convert it to RGB mode
    original_image = Image.open(image_path).convert("RGB")

    # Get the width and height of the original image
    width, height = original_image.size

    # Calculate the split point (middle of the image)
    split_point = width // 2

    # Split the image into left and right halves
    left_half = original_image.crop((0, 0, split_point, height))
    right_half = original_image.crop((split_point, 0, width, height))

    # Save the two new images
    left_half.save("images/l_ws_uint16.jpg")
    right_half.save("images/r_ws_uint16.jpg")


if __name__ == "__main__":
    image_path = "images/ws_uint16.png"  # Replace this with the actual path of your RGB image
    split_image(image_path)