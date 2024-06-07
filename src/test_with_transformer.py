import os
import json
import pickle
import torch

from model import CNN, DirectionPredictionModelWithTransformer
from model import EncodeTensor
from polygon_dataset import PolygonDatasetForCNN, PolygonDatasetForSeries
from debug import visualize_results
from train_with_transformer import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM

CURR_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURR_DIR, "..", "models/trained_model_6000.pth")

NUM_SAMPLES = 2**15
NUM_TEST_SAMPLES = 64
IMG_SIZE = 32  # 32 * 32 픽셀 처럼 표현 해상도 결정


if __name__ == "__main__":
    # 테스트
    pickle_path = os.path.join(
        CURR_DIR,
        "..",
        f"data/dataset_{NUM_SAMPLES}_{NUM_TEST_SAMPLES}_{IMG_SIZE}_series.pickle",
    )
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            dataset = pickle.load(f)
        print("dataset loaded from pickle.")
    else:
        dataset = PolygonDatasetForSeries(NUM_SAMPLES, NUM_TEST_SAMPLES, IMG_SIZE, dataset_for="series")
        with open(pickle_path, "wb") as f:
            pickle.dump(dataset, f)
        print("dataset created and pickled.")

    model = DirectionPredictionModelWithTransformer(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

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

    visualize_results(
        dataset.test_parcel_img_tensor_dataset,
        result_vecs,
        dataset
    )
