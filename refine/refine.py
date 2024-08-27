import argparse
import os

import cv2
import numpy as np


def get_black_border(img):
    """
    Cut the black border of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x, center_y = width // 2, height // 2

    black_pixels_horizon = np.where(img[center_y] > 0)[0]
    black_pixels_vertical = np.where(img[:, center_x] > 0)[0]

    print(f"black_pixels_horizon: {black_pixels_horizon}")
    print(f"black_pixels_vertical: {black_pixels_vertical}")

    if len(black_pixels_horizon) == 0:
        print("No horizontal black pixels found")
        return img

    if len(black_pixels_vertical) == 0:
        print("No vertical black pixels found")
        return img

    left_x = black_pixels_horizon[0]
    right_x = black_pixels_horizon[-1]

    top_y = black_pixels_vertical[0]
    bottom_y = black_pixels_vertical[-1]

    return left_x, right_x, top_y, bottom_y


def remove_black_border(image_dir, output_dir="cropped"):
    sample = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    left_x, right_x, top_y, bottom_y = get_black_border(sample)
    crop_rect = (left_x, right_x, top_y, bottom_y)
    crop_images(image_dir, crop_rect, output_dir)


def crop_images(image_dir, crop_rect, output_dir="cropped"):
    """
    Crop images to remove the black border.
    """
    if not os.path.exists(os.path.join("output", output_dir)):
        os.makedirs(os.path.join("output", output_dir))

    for image_name in os.listdir(image_dir):
        print(f"Processing image {image_name} from {image_dir}")
        img = cv2.imread(os.path.join(image_dir, image_name))
        cropped_img = img[crop_rect[2]:crop_rect[3], crop_rect[0]:crop_rect[1]]
        output_path = os.path.join("output", output_dir, image_name)
        cv2.imwrite(output_path, cropped_img)
        print(f"Images saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Directory with images to crop")
    parser.add_argument("--output", type=str, help="Output directory for cropped images", default="cropped")
    args = parser.parse_args()

    remove_black_border(args.input, args.output)
