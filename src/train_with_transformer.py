import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import shapely.affinity
from shapely.geometry import Polygon
from model import DirectionPredictionModelWithTransformer
from polygon_dataset import PolygonDatasetForSeriesData
from debug import visualize_results

if __name__ == "__main__":
    num_samples = 2 ** 10
    num_tests = 64

    train_dataset = PolygonDatasetForSeriesData(num_samples, num_tests, 32)

    input_dim = 2
    hidden_dim = 16
    output_dim = 2
    model = DirectionPredictionModelWithTransformer(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2000
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, train_data in enumerate(train_dataset):
            optimizer.zero_grad()

            output = model(train_data.unsqueeze(0).float())  # Add .float() for compatibility
            label = train_dataset.vec_dataset[i].unsqueeze(0)

            loss = criterion(output, label.float())  # Add .float() for compatibility
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataset):.4f}')

    predictions = []
    for points in train_dataset.test_parcel_img_tensor_dataset:
        pred_direction = model(points.unsqueeze(0).float()).detach().numpy()  # Add .float() for compatibility
        predictions.append(pred_direction)

    visualize_results(train_dataset.test_parcel_img_tensor_dataset, predictions)
