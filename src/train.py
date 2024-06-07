import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import CNN
from polygon_dataset import PolygonDatasetForCNN, load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import debugvisualizer as dv

CURR_DIR = os.path.dirname(__file__)

NUM_SAMPLES = 2 ** 15
NUM_TEST_SAMPLES = 64
IMG_SIZE = 32  # 32 * 32 픽셀 처럼 표현 해상도 결정

if __name__ == "__main__":
    # CUDA, MPS, CPU 순으로 디바이스 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 모델 초기화 및 선택된 디바이스로 이동
    model = CNN().to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()  # 회귀 문제 사용할 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 데이터셋 인스턴스 생성
    dataset = load_dataset(NUM_SAMPLES, NUM_TEST_SAMPLES, IMG_SIZE, "cnn")

    # 데이터 및 라벨 불러오기
    batch_size = NUM_SAMPLES
    batch_counts = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels = [
        torch.tensor(
            [dataset.get_vec(i * batch_size + j) for j in range(batch_size)]
        ).to(device)
        for i in range(batch_counts)
    ]  # FIXME: dataloader 자체에 적용

    print("train start")
    num_epochs = 10000

    def update(epoch):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs = data.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, all_labels[i])  # 실제 레이블을 사용하여 손실 계산

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / 100

        if (epoch + 1) % 100 == 0:
            print("[%d] loss: %.3f" % (epoch + 1, epoch_loss))

            # Save the model
            model_file_name = f"trained_model_{epoch + 1}.pth"
            model_save_path = os.path.join(CURR_DIR, "..", f"models/{model_file_name}")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    for i in range(num_epochs):
        update(epoch=i)

    print("Finished Training")
