from matplotlib import pyplot as plt


def visualize_polygon_dataset(img_tensors, vecs, comparison_vecs, num_images=64):
    num_rows = num_images // 8  # 8개씩 8줄로 나누기
    _, axes = plt.subplots(num_rows, 8, figsize=(16, 2*num_rows))  # 그림판 생성

    for i in range(num_images):
        row = i // 8
        col = i % 8

        img_tensor = img_tensors[i].squeeze().numpy()  # 이미지 텐서 가져오기
        axes[row, col].imshow(img_tensor, cmap='gray')  # 이미지 표시

        vec_multiplier = 1
        # 이미지 위에 벡터들을 선분으로 표시
        vec = vecs[i]
        axes[row, col].plot([0, vec[0] * vec_multiplier], [0, vec[1] * vec_multiplier], color='red')  # 빨간색 벡터

        comparison_vec = comparison_vecs[i]
        axes[row, col].plot([0, comparison_vec[0] * vec_multiplier], [0, comparison_vec[1] * vec_multiplier], color='green')  # 초록색 벡터

        axes[row, col].axis('off')  # 축 레이블 제거

    plt.show()
