import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DirectionPredictionModelWithTransformer
from polygon_dataset import load_dataset
from debug import visualize_results
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # Separate the data and labels in the batch
    data = [item[0] for item in batch]
    labels_in_batch = [torch.tensor(item[1]) for item in batch]

    # Pad the sequences so they are all the same size in the batch
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    labels_in_batch = torch.stack(labels_in_batch)

    return data_padded, labels_in_batch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    NUM_SAMPLES = 2 ** 10
    NUM_TESTS = 64
    batch_size = 32  # batch size 단위로 폴리곤 점 개수를 맞춰서 한번에 학습에 박아넣음 - collate_fn

    train_dataset = load_dataset(NUM_SAMPLES, NUM_TESTS, 32, "series")
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

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
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {total_loss / len(dataloader):.4f}')

    predictions = []
    for points in train_dataset.test_parcel_img_tensor_dataset:
        points = points.to(device)
        pred_direction = model(points.unsqueeze(0).float()).detach().cpu().numpy()
        predictions.append(pred_direction)

    visualize_results(train_dataset.test_parcel_img_tensor_dataset, predictions)
