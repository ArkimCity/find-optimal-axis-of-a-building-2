from matplotlib import pyplot as plt


def visualize_polygon_dataset(img_tensors, vecs, comparison_vecs, num_images=64):
    num_rows = num_images // 8  # 8개씩 8줄로 나누기
    _, axes = plt.subplots(num_rows, 8, figsize=(16, 2 * num_rows))  # 그림판 생성

    for i in range(num_images):
        row = i // 8
        col = i % 8

        img_tensor = img_tensors[i].squeeze().numpy()  # 이미지 텐서 가져오기
        axes[row, col].imshow(img_tensor, cmap="gray")  # 이미지 표시

        vec_multiplier = 1
        # 이미지 위에 벡터들을 선분으로 표시
        vec = vecs[i]
        axes[row, col].plot(
            [0, vec[0] * vec_multiplier], [0, vec[1] * vec_multiplier], color="red"
        )  # 빨간색 벡터

        comparison_vec = comparison_vecs[i]
        axes[row, col].plot(
            [0, comparison_vec[0] * vec_multiplier],
            [0, comparison_vec[1] * vec_multiplier],
            color="green",
        )  # 초록색 벡터

        axes[row, col].axis("off")  # 축 레이블 제거

    plt.show()

def visualize_results(test_cases, predictions):
    num_test_cases = len(test_cases)
    num_cols = 4
    num_rows = (num_test_cases + num_cols - 1) // num_cols

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for i in range(num_test_cases):
        points = test_cases[i]
        pred_direction = predictions[i]

        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(points[:, 0], points[:, 1], 'b-')
        plt.scatter(points[:, 0], points[:, 1], color='r')

        calculated_direction = calculate_main_direction(points)
        plt.arrow(points[0][0], points[0][1], calculated_direction[0], calculated_direction[1], head_width=0.5, head_length=0.5, fc='r', ec='r')

        plt.arrow(points[0][0], points[0][1], pred_direction[0], pred_direction[1], head_width=0.5, head_length=0.5, fc='g', ec='g')

        plt.title(f"Test Case {i+1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
