import os
import json
import cv2
import numpy as np

import shapely.affinity
from shapely.geometry import Polygon

import torch
from torch.utils.data import Dataset

CURR_DIR = os.path.dirname(__file__)

PARCELS_DATA_FILE_NAME = "AL_11_D002_20230506.json"
PARCELS_DATA_FILE_PATH = os.path.join(CURR_DIR, "..", "data", PARCELS_DATA_FILE_NAME)
BUILDINGS_DATA_FILE_NAME = "AL_11_D010_20230506.json"
BUILDINGS_DATA_FILE_PATH = os.path.join(CURR_DIR, "..", "data", BUILDINGS_DATA_FILE_NAME)

PARCELS_DATA_FILE_NAME_FOR_TEST = "parcels_data_for_test.json"
PARCELS_DATA_FILE_PATH_FOR_TEST = os.path.join(CURR_DIR, "..", "data", PARCELS_DATA_FILE_NAME_FOR_TEST)
BUILDINGS_DATA_FILE_NAME_FOR_TEST = "buildings_data_for_test.json"
BUILDINGS_DATA_FILE_PATH_FOR_TEST = os.path.join(CURR_DIR, "..", "data", BUILDINGS_DATA_FILE_NAME_FOR_TEST)


class EachData:
    def __init__(self, parcel_img_tensor, building_img_tensor, vec) -> None:
        self.parcel_img_tensor = parcel_img_tensor
        self.building_img_tensor = building_img_tensor
        self.vec = vec


class PolygonDatasetBase(Dataset):
    def __init__(self, num_samples, num_test_samples, img_size, is_test=False):
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples
        self.img_size = img_size

        print("loading parcels data ...")
        with open(PARCELS_DATA_FILE_PATH_FOR_TEST if is_test else PARCELS_DATA_FILE_PATH, "r", encoding="utf-8") as f:
            parcels_data_json = json.load(f)
        print("loading buildings data ...")
        with open(BUILDINGS_DATA_FILE_PATH_FOR_TEST if is_test else BUILDINGS_DATA_FILE_PATH, "r", encoding="utf-8") as f:
            buildings_data_json = json.load(f)
        print("loading data done")

        # 필지의 pnu를 기준으로 building 을 조회할 예정이기 때문에 미리 index 생성
        print("making buildings data index ...")
        self.buildings_data_index = self.make_buildings_data_index(buildings_data_json)

        print("making train dataset ...")
        (
            all_parcel_img_tensor_dataset,
            all_building_img_tensor_dataset,
            all_vec_dataset,
        ) = self.make_datasets(self.num_samples + self.num_test_samples, parcels_data_json)

        # 학습에 사용할 데이터
        self.parcel_img_tensor_dataset = all_parcel_img_tensor_dataset[:self.num_samples]
        self.building_img_tensor_dataset = all_building_img_tensor_dataset[:self.num_samples]
        self.vec_dataset = all_vec_dataset[:self.num_samples]
        # 평가에 사용할 데이터
        self.test_parcel_img_tensor_dataset = all_parcel_img_tensor_dataset[self.num_samples:]
        self.test_building_img_tensor_dataset = all_building_img_tensor_dataset[self.num_samples:]
        self.test_vec_dataset = all_vec_dataset[self.num_samples:]

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

        if any(np.isclose(max_coords - min_coords, 0, atol=1e-8)):
            raise Exception("정상적인 나눔이 아닙니다.")

        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords)
        # 이미지 크기에 맞게 좌표 조정 (img_size 기준)
        scaled_vertices = normalized_vertices * (self.img_size - 1)

        return scaled_vertices

    def make_each_data(self, vertices_raw):
        # input 점들을 img_size 크기에 맞게 변환 준비
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        polygon_translated = shapely.affinity.translate(
            polygon_normalized,
            -polygon_normalized.bounds[0] + 0.1,
            -polygon_normalized.bounds[1] + 0.1,
        )
        vertices_translated = np.array(polygon_translated.exterior.coords)

        return vertices_translated

    def get_label_vector(self, vertices_raw):
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        polygon_translated = shapely.affinity.translate(
            polygon_normalized,
            -polygon_normalized.bounds[0] + 0.1,
            -polygon_normalized.bounds[1] + 0.1,
        )

        coords = polygon_translated.minimum_rotated_rectangle.exterior.coords
        vecs = [
            (coords[i + 1][0] - coord[0], coords[i + 1][1] - coord[1])
            for i, coord in enumerate(coords[:-1])
        ]
        vec_raw = [vec for vec in vecs if vec[0] >= 0 and vec[1] > 0][0]
        vec = vec_raw

        return vec

    def make_buildings_data_index(self, buildings_data_json):
        buildings_index = {}
        for building in buildings_data_json["features"]:
            a2 = building["properties"]["A2"]
            if a2 not in buildings_index:
                buildings_index[a2] = []
            buildings_index[a2].append(building)
        return buildings_index

    def make_datasets(self, needed_num_samples, parcels_data_json):
        parcel_img_tensor_dataset = []
        building_img_tensor_dataset = []
        labels = []
        checking_index = 0
        for checking_index, parcel_data in enumerate(parcels_data_json["features"]):
            try:
                parcel_vertices = parcel_data["geometry"]["coordinates"][0]
                pnu = parcels_data_json["features"][checking_index]["properties"]["A1"]

                matching_buildings = self.buildings_data_index.get(pnu, [])

                if len(matching_buildings) > 0:
                    matching_building = matching_buildings[0]
                    building_vertices = matching_building["geometry"]["coordinates"][0]

                    building_img_tensor = self.make_each_data(building_vertices)
                    parcel_img_tensor = self.make_each_data(parcel_vertices)
                    vec = self.get_label_vector(building_vertices)

                    parcel_img_tensor_dataset.append(parcel_img_tensor)
                    building_img_tensor_dataset.append(building_img_tensor)
                    labels.append(vec)
                else:
                    # print("no matching buildins. pnu: ", pnu)
                    pass
            except Exception:
                print("error occured during processing, pnu: ", pnu)

            if len(parcel_img_tensor_dataset) == needed_num_samples:
                # 필요한 개수 모이면 stop
                break

        return parcel_img_tensor_dataset, building_img_tensor_dataset, labels

class PolygonDatasetForCNN(PolygonDatasetBase):
    def __init__(self, num_samples, num_test_samples, img_size, is_test=False):
        super().__init__(num_samples, num_test_samples, img_size, is_test)

    def make_each_img_tensor(self, vertices_translated):
        img = np.zeros((self.img_size, self.img_size))
        for i in range(len(vertices_translated) - 1):
            pt1 = tuple((vertices_translated[i]).astype(int))
            pt2 = tuple((vertices_translated[i + 1]).astype(int))
            cv2.line(img, pt1, pt2, color=1, thickness=1)

        # PyTorch Tensor로 변환
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.unsqueeze(0)  # 채널 차원 추가

        return img_tensor

    def make_each_data(self, vertices_raw):
        # input 점들을 img_size 크기에 맞게 변환 준비
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        polygon_translated = shapely.affinity.translate(
            polygon_normalized,
            -polygon_normalized.bounds[0] + 0.1,
            -polygon_normalized.bounds[1] + 0.1,
        )
        vertices_translated = np.array(polygon_translated.exterior.coords)

        img_tensor = self.make_each_img_tensor(vertices_translated)

        return img_tensor

class PolygonDatasetForSeriesData(PolygonDatasetBase):

    def make_each_vertices_data(self, vertices_translated):
        return vertices_translated

    def make_each_data(self, vertices_raw):
        # input 점들을 img_size 크기에 맞게 변환 준비
        vertices_normalized = self.normalize_coordinates(vertices_raw)
        polygon_normalized = Polygon(vertices_normalized)
        vertices_translated = np.array(polygon_normalized.exterior.coords)

        vertices_data = self.make_each_vertices_data(vertices_translated)

        return vertices_data

if __name__ == "__main__":
    # 데이터 체크
    dataset_test = PolygonDatasetForSeriesData(0, 128, 32, True)

    # 데이터 체크
    dataset_test = PolygonDatasetForCNN(2 ** 16, 128, 32)
    assert dataset_test.num_samples == len(dataset_test.parcel_img_tensor_dataset)
    assert dataset_test.num_samples == len(dataset_test.building_img_tensor_dataset)
    assert dataset_test.num_samples == len(dataset_test.vec_dataset)

    assert dataset_test.num_test_samples == len(dataset_test.test_parcel_img_tensor_dataset)
    assert dataset_test.num_test_samples == len(dataset_test.test_building_img_tensor_dataset)
    assert dataset_test.num_test_samples == len(dataset_test.test_vec_dataset)
