
import json
from pathlib import Path
from colmapUtils.read_write_model import *
import numpy as np
import struct


def convert_to_native_type(value):
    if isinstance(value, np.generic):
        return value.item()  # 转换为原生Python类型
    return value


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# Function to read binary cameras.bin file and extract intrinsic matrix
def read_cameras_binary(path_to_model_file):
    cameras = {}

    CAMERA_MODEL_IDS = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE",
        6: "FULL_OPENCV",
        7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE",
        9: "RADIAL_FISHEYE",
        10: "THIN_PRISM_FISHEYE"
    }

    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            # Read camera properties
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id]
            width = camera_properties[2]
            height = camera_properties[3]

            # Handle different camera models based on model_name
            if model_name == "SIMPLE_PINHOLE":
                num_params = 3  # fx, cx, cy
            elif model_name == "PINHOLE":
                num_params = 4  # fx, fy, cx, cy
            elif model_name == "SIMPLE_RADIAL":
                num_params = 4  # fx, cx, cy, k1 (radial distortion)
            else:
                raise NotImplementedError(f"Camera model {model_name} not implemented.")

            # Read camera parameters
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)

            # Construct the intrinsic matrix K and handle radial distortion
            if model_name == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            elif model_name == "PINHOLE":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            elif model_name == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                k1 = params[3]  # radial distortion parameter
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                # You will need to handle radial distortion in projection functions separately
                print(f"Radial distortion k1: {k1} for camera {camera_id}")

            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
                "K": K  # Store the intrinsic matrix
            }

        assert len(cameras) == num_cameras
    return cameras


def load_colmap_coords(basedir):
    output_list = {}  # 使用字典来存储每个 3D 点和其对应的多个图像信息
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')
    cameras = read_cameras_binary(Path(basedir) / 'sparse' / '0' / 'cameras.bin')

    # 遍历所有图像，找到每个 3D 点在不同图像中的 2D 坐标
    for id_im in range(1, len(images) + 1):
        image_info = images[id_im]
        R = image_info.qvec2rotmat()  # 旋转矩阵
        t = image_info.tvec.reshape([3, 1])  # 平移向量
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

        ###########至关重要#############
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)  # 世界坐标系到相机坐标系的变换矩阵
        K = cameras[id_im]["K"]  # 内参矩阵
        ##############################

        for i in range(len(image_info.xys)):
            point2D = image_info.xys[i]
            id_3D = image_info.point3D_ids[i]

            # 如果没有对应的3D点，跳过该2D点
            if id_3D == -1:
                continue

            # 获取该3D点的坐标
            point3D = points[id_3D].xyz

            # 如果该 3D 点还没有存入 output_list，则创建一条记录
            if id_3D not in output_list:
                output_list[id_3D] = {
                    "point_id": convert_to_native_type(id_3D),
                    "3D_coord": [convert_to_native_type(coord) for coord in point3D],
                    "image_projections": []  # 存储该点在不同图像中的投影
                }

            # 将当前图像的信息和对应的2D坐标（直接搜索结果）存入字典
            output_list[id_3D]["image_projections"].append({
                "image_id": convert_to_native_type(id_im),
                "2D_coord_direct_search": [convert_to_native_type(coord) for coord in point2D]
            })

            # 通过相机矩阵计算 2D 投影坐标
            point3D_homogeneous = np.append(point3D, 1)  # 转换为齐次坐标
            proj_2D = K @ (w2c @ point3D_homogeneous)[:3]  # 投影到相机平面
            proj_2D = proj_2D[:2] / proj_2D[2]  # 齐次坐标转为2D坐标

            # 将计算得到的投影结果添加到字典
            output_list[id_3D]["image_projections"][-1]["2D_coord_projected"] = [convert_to_native_type(coord) for coord
                                                                                 in proj_2D]

    # 将结果转换为列表并保存为 JSON 文件
    output_list_values = list(output_list.values())
    output_file = Path(basedir) / 'colmap_output_with_projection.json'
    with open(output_file, 'w') as f:
        json.dump(output_list_values, f, indent=4)
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    ###########改一下dataset的地址##########
    basedir = r"test_dataset"
    load_colmap_coords(basedir)
