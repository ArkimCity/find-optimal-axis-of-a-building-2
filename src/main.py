import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import CNN
from model import EncodeTensor
from polygon_dataset import PolygonDataset
from debug import visualize_polygon_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import debugvisualizer as dv


if __name__ == "__main__":
    # 모델 초기화
    model = CNN()

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()  # 회귀 문제 사용할 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 데이터셋 인스턴스 생성
    num_samples = 2048
    num_test_samples = 64
    img_size = 32  # 32 * 32 픽셀 처럼 표현 해상도 결정
    dataset = PolygonDataset(num_samples=num_samples, num_test_samples=num_test_samples, img_size=img_size)
    batch_size = num_samples
    batch_counts = 1

    # 데이터 및 라벨 불러오기
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels = [
        torch.tensor([dataset.get_vec(i * batch_size + j) for j in range(batch_size)]) for i in range(batch_counts)
    ]  # FIXME: dataloader 자체에 적용

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line, = ax.plot([], [], color='blue')  # 초기 라인 객체 생성

    num_epochs = 2000
    losses = []

    def init():
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, 5)  # 손실의 예상 범위를 설정, 필요에 따라 조정하세요
        return line,

    def update(epoch):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs = data
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, all_labels[i])  # 실제 레이블을 사용하여 손실 계산

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / 100
        losses.append(epoch_loss)
        line.set_data(range(epoch+1), losses)

        if (epoch+1) % 100 == 0:
            print('[%d] loss: %.3f' % (epoch + 1, epoch_loss))

        return line,

    ani = animation.FuncAnimation(fig, update, frames=num_epochs, init_func=init, blit=True, interval=50)

    plt.show()

    print('Finished Training')

    # FIXME: save model


def test():
    # 테스트
    result_vecs = []
    for test_data in dataset.test_parcel_img_tensor_dataset:
        result_vec = model(test_data.unsqueeze(0))
        result_vecs.append((float(result_vec[0][0]), float(result_vec[0][1])))

    with open('test_result.json', 'w', encoding="utf-8") as json_file:
        json.dump({
            "test_datsets": dataset.test_datasets,
            "test_vecs": dataset.test_vecs,
            "result_vecs": result_vecs,
        }, json_file, cls=EncodeTensor)

    visualize_polygon_dataset(dataset.test_datasets, result_vecs, dataset.test_vecs, num_images=num_test_samples)
