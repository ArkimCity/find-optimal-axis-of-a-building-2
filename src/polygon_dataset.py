import os
import json
import cv2
import numpy as np

import shapely.affinity
from shapely.geometry import Polygon

import torch
from torch.utils.data import Dataset, DataLoader

CURR_DIR = os.path.dirname(__file__)


class EachData:
    def __init__(self, parcel_img_tensor, building_img_tensor, vec) -> None:
        self.parcel_img_tensor = parcel_img_tensor
        self.building_img_tensor = building_img_tensor
        self.vec = vec


class PolygonDataset(Dataset):
    def __init__(self, num_samples, num_test_samples, img_size):
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples
        self.img_size = img_size

        with open(os.path.join(CURR_DIR, "..", "data/buildings_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json"), "r") as f:
            self.buildings_data_json: dict = json.load(f)
        with open(os.path.join(CURR_DIR, "..", "data/parcels_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json"), "r") as f:
            self.parcels_data_json: dict = json.load(f)

        buffer_num = 2217  # FIXME:

        # 학습에 사용할 데이터
        self.parcel_img_tensor_dataset, self.building_img_tensor_dataset, self.vec_dataset = self.make_datasets(0, self.num_samples + buffer_num)  # FIXME: 41 개가 적합하지 않음. 기본 데이터를 수정

        # 평가에 사용할 데이터
        self.test_parcel_img_tensor_dataset, self.test_building_img_tensor_dataset, self.test_vec_dataset  = self.make_datasets(self.num_samples + buffer_num, self.num_samples + buffer_num + self.num_test_samples)

    def __len__(self):
        return len(self.parcel_img_tensor_dataset)

    def __getitem__(self, index):
        return self.parcel_img_tensor_dataset[index]

    def get_vec(self, idx):
        return self.vec_dataset[idx]

    def normalize_coordinates(self, vertices):
        # 다각형의 좌표를 [0, 1] 범위로 정규화
        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)

        # 이미지 크기에 맞게 좌표 조정 (img_size 기준)
        scaled_vertices = normalized_vertices * (self.img_size - 1)

        return scaled_vertices

    def make_each_img_tensor(self, vertices_raw):
        # input 점들을 img_size 크기에 맞게 변환 준비
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        polygon_translated = shapely.affinity.translate(polygon_normalized, -polygon_normalized.bounds[0] + 0.1, -polygon_normalized.bounds[1] + 0.1)
        vertices_translated = np.array(polygon_translated.exterior.coords)

        img = np.zeros((self.img_size, self.img_size))
        for i in range(len(vertices_translated) - 1):
            pt1 = tuple((vertices_translated[i]).astype(int))
            pt2 = tuple((vertices_translated[i + 1]).astype(int))
            cv2.line(img, pt1, pt2, color=1, thickness=1)

        # PyTorch Tensor로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # 채널 차원 추가
        return img_tensor

    def get_label_vector(self, vertices_raw):
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        polygon_translated = shapely.affinity.translate(polygon_normalized, -polygon_normalized.bounds[0] + 0.1, -polygon_normalized.bounds[1] + 0.1)

        coords = polygon_translated.minimum_rotated_rectangle.exterior.coords
        vecs = [(coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1]) for i, coord in enumerate(coords[:-1])]
        vec_raw = [vec for vec in vecs if vec[0] >= 0 and vec[1] > 0][0]
        vec = vec_raw

        return vec

    def make_datasets(self, start_index, end_index):
        parcel_img_tensor_dataset = []
        building_img_tensor_dataset = []
        labels = []
        for i in range(start_index, end_index):
            parcel_vertices = self.parcels_data_json["features"][i]["geometry"]["coordinates"][0]
            pnu = self.parcels_data_json["features"][i]["properties"]["A1"]

            matching_buildings = [parcel for parcel in self.buildings_data_json["features"] if parcel["properties"]["A2"] == pnu]

            if len(matching_buildings) > 0:
                matching_building = matching_buildings[0]
                building_vertices = matching_building["geometry"]["coordinates"][0]

                building_img_tensor = self.make_each_img_tensor(building_vertices)
                parcel_img_tensor = self.make_each_img_tensor(parcel_vertices)
                vec = self.get_label_vector(building_vertices)

                parcel_img_tensor_dataset.append(parcel_img_tensor)
                building_img_tensor_dataset.append(building_img_tensor)
                labels.append(vec)

        return parcel_img_tensor_dataset, building_img_tensor_dataset, labels
