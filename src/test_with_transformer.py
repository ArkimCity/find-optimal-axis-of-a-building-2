import os
import json
import torch

from model import DirectionPredictionModelWithTransformer
from model import EncodeTensor
from polygon_dataset import load_dataset
from debug import visualize_results
from train_with_transformer import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM
from train_with_transformer import NUM_SAMPLES, NUM_TESTS, POLYGON_SCALE

CURR_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(CURR_DIR, "..", "models/transformer/trained_model_2000.pth")


if __name__ == "__main__":
    # 테스트
    pickle_path = os.path.join(
        CURR_DIR,
        "..",
        f"data/dataset_{NUM_SAMPLES}_{NUM_TESTS}_{POLYGON_SCALE}_series.pickle",
    )
    dataset = load_dataset(NUM_SAMPLES, NUM_TESTS, POLYGON_SCALE, "series")

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
        dataset.test_building_img_tensor_dataset,
        dataset.test_vec_dataset,
        result_vecs,
    )
