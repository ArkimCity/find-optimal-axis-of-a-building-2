import os
import json
import torch

from model import CNN
from model import EncodeTensor
from polygon_dataset import PolygonDataset
from debug import visualize_polygon_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models/trained_model.pth")

NUM_SAMPLES = 2048
NUM_TEST_SAMPLES = 64
IMG_SIZE = 32  # 32 * 32 픽셀 처럼 표현 해상도 결정


if __name__ == "__main__":
    # 테스트
    dataset = PolygonDataset(NUM_SAMPLES, NUM_TEST_SAMPLES, IMG_SIZE)

    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    result_vecs = []
    for test_data in dataset.test_parcel_img_tensor_dataset:
        result_vec = model(test_data.unsqueeze(0))
        result_vecs.append((float(result_vec[0][0]), float(result_vec[0][1])))

    with open("test_result.json", "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "test_datsets": dataset.test_parcel_img_tensor_dataset,
                "test_vecs": dataset.test_vec_dataset,
                "result_vecs": result_vecs,
            },
            json_file,
            cls=EncodeTensor,
        )

    visualize_polygon_dataset(
        dataset.test_parcel_img_tensor_dataset,
        result_vecs,
        dataset.test_vec_dataset,
        num_images=32,
    )
