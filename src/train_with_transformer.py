import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DirectionPredictionModelWithTransformer
from polygon_dataset import load_dataset
from debug import visualize_results


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')  # Use MPS if available on macOS with M1/M2 chips
    elif torch.cuda.is_available():
        return torch.device('cuda')  # Use CUDA if NVIDIA GPU is available
    else:
        return torch.device('cpu')  # Fallback to CPU if none of the above


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    NUM_SAMPLES = 2 ** 10
    NUM_TESTS = 64
    batch_size = NUM_SAMPLES
    batch_counts = 1

    train_dataset = load_dataset(NUM_SAMPLES, NUM_TESTS, 32, "series")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    all_labels = [
        torch.tensor(
            [train_dataset.get_vec(i * batch_size + j) for j in range(batch_size)]
        ).to(device)
        for i in range(batch_counts)
    ]  # FIXME: dataloader 자체에 적용

    INPUT_DIM = 2
    HIDDEN_DIM = 16
    OUTPUT_DIM = 2
    model = DirectionPredictionModelWithTransformer(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 2000
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for i, train_data in enumerate(dataloader, 0):
            inputs = train_data.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, all_labels[i])  # 실제 레이블을 사용하여 손실 계산

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss / len(train_dataset):.4f}')

    predictions = []
    for points in train_dataset.test_parcel_img_tensor_dataset:
        points = points.to(device)  # Ensure test data is on the same device
        pred_direction = model(points.unsqueeze(0).float()).detach().cpu().numpy()
        predictions.append(pred_direction)

    visualize_results(train_dataset.test_parcel_img_tensor_dataset, predictions)
