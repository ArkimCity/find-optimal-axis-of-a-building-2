import os
import json
import cv2
import numpy as np

import shapely.affinity
from shapely import Polygon

import torch
from torch.utils.data import DataLoader, Dataset

CURR_DIR = os.path.dirname(__file__)


class PolygonDataset(Dataset):
    def __init__(self, num_samples, num_test_samples, img_size):
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples
        self.img_size = img_size

        with open(os.path.join(CURR_DIR, "..", "data/buildings_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json"), "r") as f:
            self.buildings_data_json = json.load(f)
        with open(os.path.join(CURR_DIR, "..", "data/parcels_data_divided/196164.22754000025_449303.8666800002_196905.28352000023_451480.8424600002.json"), "r") as f:
            self.parcels_data_json = json.load(f)

        # 학습에 사용할 데이터
        (
            parcels_img_tensors,
            buildings_img_tensors,
            vecs
        ) = self.make_datasets(0, self.num_samples + 41)  # FIXME: 41 개가 적합하지 않음. 기본 데이터를 수정
        self.parcels_img_tensors = parcels_img_tensors
        self.buildings_img_tensors = buildings_img_tensors
        self.vecs = vecs

        # 평가에 사용할 데이터
        (
            test_parcels_img_tensors,
            test_buildings_img_tensors,
            test_vecs
        ) = self.make_datasets(self.num_samples + 41, self.num_samples + 41 + self.num_test_samples)
        self.test_parcels_img_tensors = test_parcels_img_tensors
        self.test_buildings_img_tensors = test_buildings_img_tensors
        self.test_vecs = test_vecs

    def __len__(self):
        return self.num_samples

    def get_vec(self, idx):
        return self.vecs[idx]

    def normalize_coordinates(self, vertices):
        # 다각형의 좌표를 [0, 1] 범위로 정규화
        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)

        # 이미지 크기에 맞게 좌표 조정 (32x32 이미지 기준)
        scaled_vertices = normalized_vertices * (self.img_size - 1)

        return scaled_vertices

    def make_each_img_tensor(self, vertices_raw):
        # input 점들을 32 32 tensor 크기에 맞게 변환 준비
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


    def make_datasets(self, start_index, end_index):
        parcels_img_tensors, buildings_img_tensors, vecs = [], []
        for i in range(start_index, end_index):
            building_vertices = self.buildings_data_json["features"][i]["geometry"]["coordinates"][0]
            pnu = self.buildings_data_json["features"][i]["properties"]["A2"]

            matching_parcels = [parcel for parcel in self.parcels_data_json["features"] if parcel["properties"]["A1"] == pnu]

            if len(matching_parcels) > 0:
                matching_parcel = matching_parcels[0]
                parcel_vertices = matching_parcel["geometry"]["coordinates"][0]

                parcel_img_tensor = self.make_each_img_tensor(parcel_vertices)
                building_img_tensor = self.make_each_img_tensor(building_vertices)
                vec = None  # FIXME

                parcels_img_tensors.append(parcel_img_tensor)
                buildings_img_tensors.append(building_img_tensor)
                vecs.append(vec)

        return parcels_img_tensors, buildings_img_tensors, vecs
