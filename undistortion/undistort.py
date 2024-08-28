import argparse

import cv2
import numpy as np
import os
import time

import pandas as pd


def parse_file_to_dataframe(file_path):
    """
    파일을 읽어 데이터프레임으로 변환 : read the file and convert it to a dataframe
    :param file_path: 파일 경로 : file path
    :return: 데이터프레임 : dataframe
    """
    # 빈 리스트를 생성하여 데이터를 저장
    data = []

    # 파일을 읽기 모드로 열기
    with open(file_path, 'r', encoding='utf-8') as file:
        # 한 줄씩 읽기
        for line in file:
            # 각 줄을 탭을 기준으로 분리
            parts = line.strip().split('\t')
            # 필요한 정보만 추출하여 딕셔너리로 변환
            data.append({
                'file_name': parts[0],
                'prediction_focal': float(parts[1]),
                'prediction_dist': float(parts[2])
            })

    # 리스트를 데이터프레임으로 변환
    df = pd.DataFrame(data)
    return df


def undistSphIm(Idis, Paramsd, Paramsund):
    Paramsund['W'] = Paramsd['W'] * 3  # size of output (undist)
    Paramsund['H'] = Paramsd['H'] * 3

    # Parameters of the camera to generate
    f_dist = Paramsd['f']
    u0_dist = Paramsd['W'] / 2
    v0_dist = Paramsd['H'] / 2

    f_undist = Paramsund['f']
    u0_undist = Paramsund['W'] / 2
    v0_undist = Paramsund['H'] / 2
    xi = Paramsd['xi']  # distortion parameters (spherical model)
    Imd_H, Imd_W, _ = Idis.shape

    # 1. Projection on the image
    grid_x, grid_y = np.meshgrid(np.arange(1, Paramsund['W'] + 1), np.arange(1, Paramsund['H'] + 1))
    X_Cam = grid_x / f_undist - u0_undist / f_undist
    Y_Cam = grid_y / f_undist - v0_undist / f_undist
    Z_Cam = np.ones((Paramsund['H'], Paramsund['W']))

    # 2. Image to sphere cart
    xi1 = 0
    alpha_cam = (xi1 * Z_Cam + np.sqrt(Z_Cam ** 2 + ((1 - xi1 ** 2) * (X_Cam ** 2 + Y_Cam ** 2)))) / (
            X_Cam ** 2 + Y_Cam ** 2 + Z_Cam ** 2)

    X_Sph = X_Cam * alpha_cam
    Y_Sph = Y_Cam * alpha_cam
    Z_Sph = (Z_Cam * alpha_cam) - xi1

    # 3. Reprojection on distorted
    den = xi * (np.sqrt(X_Sph ** 2 + Y_Sph ** 2 + Z_Sph ** 2)) + Z_Sph
    X_d = ((X_Sph * f_dist) / den) + u0_dist
    Y_d = ((Y_Sph * f_dist) / den) + v0_dist

    # 4. Final step interpolation and mapping
    Image_und = np.zeros((Paramsund['H'], Paramsund['W'], 3))
    for c in range(3):
        Image_und[:, :, c] = cv2.remap(Idis[:, :, c].astype(np.float32), X_d.astype(np.float32), Y_d.astype(np.float32),
                                       interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    return Image_und


def undistort_image(image, focal, distortion):
    Idis = image

    xi = distortion  # distortion
    ImH, ImW, _ = Idis.shape
    f_dist = focal * (ImW / ImH) * (ImH / 299)  # focal length
    u0_dist = int(ImW / 2)
    v0_dist = int(ImH / 2)
    Paramsd = {'f': f_dist, 'W': u0_dist * 2, 'H': v0_dist * 2, 'xi': xi}

    Paramsund = {'f': f_dist, 'W': u0_dist * 2, 'H': v0_dist * 2}
    print(f"Undistorting image with focal length {focal} and distortion {distortion}")
    print(f"Image shape: {Idis.shape}")
    print(f"Distortion parameters: {Paramsd}")
    print(f"Undistortion parameters: {Paramsund}")

    start_time = time.time()
    Image_und = undistSphIm(Idis, Paramsd, Paramsund)
    print("Undistortion time: {:.2f} seconds".format(time.time() - start_time))

    return Image_und


def undistort_images(image_paths, focals, distortions, output_dir="undistorted"):
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists(os.path.join("output", output_dir)):
        os.makedirs(os.path.join("output", output_dir))

    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        image_name = image_paths[i].split(os.sep)[-1]
        print(f"Undistorting image {image_name}")
        undistorted = undistort_image(img, focals[i], distortions[i])
        print(f"Saving undistorted image to {os.path.join('output', output_dir, image_name)}")
        cv2.imwrite(os.path.join("output", output_dir, f"{image_name}"), undistorted)


def undistort_images_from_file(result_file, output_dir="undistorted"):
    df = parse_file_to_dataframe(result_file)
    image_paths = df['file_name'].tolist()
    focals = df['prediction_focal'].tolist()
    distortions = df['prediction_dist'].tolist()

    return undistort_images(image_paths, focals, distortions, output_dir)


def get_mean_params_from_file(result_file):
    df = parse_file_to_dataframe(result_file)
    image_paths = df['file_name'].tolist()
    focal_mean = df['prediction_focal'].mean()
    distortion_mean = df['prediction_dist'].mean()

    return image_paths, focal_mean, distortion_mean


def undistort_with_mean_params(result_file, output_dir="undistorted"):
    image_paths, focal_mean, distortion_mean = get_mean_params_from_file(result_file)
    return undistort_images(image_paths, [focal_mean] * len(image_paths), [distortion_mean] * len(image_paths), output_dir)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input directory name', nargs='?', default="input")
    parser.add_argument('-o', '--output', type=str, help='output directory name', nargs='?', default="undistorted")
    parser.add_argument('-p', '--option', type=str, help='option', nargs='?', default="mean")
    if parser.parse_args().option == "mean":
        undistort_with_mean_params(parser.parse_args().input, output_dir=parser.parse_args().output)
    elif parser.parse_args().option == "manual":
        parser.parse_args()
        input_paths = os.listdir(os.path.join('input', parser.parse_args().input))
        input_paths = [os.path.join('input', parser.parse_args().input, path) for path in input_paths]
        undistort_images(input_paths, [330] * len(input_paths), [0.30] * len(input_paths), output_dir=parser.parse_args().output)
    else:
        undistort_images_from_file(parser.parse_args().input, output_dir=parser.parse_args().output)
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
