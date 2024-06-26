from matplotlib import pyplot as plt

from polygon_dataset import PolygonDatasetForSeries


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


def visualize_results(test_cases, test_buildings, test_vecs, predictions):
    num_test_cases = len(test_cases)
    num_cols = 8
    num_rows = (num_test_cases + num_cols - 1) // num_cols

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for i in range(num_test_cases):
        points = test_cases[i]
        pred_direction = predictions[i]
        building = test_buildings[i]  # Get the building for the current test case

        plt.subplot(num_rows, num_cols, i + 1)
        plt.plot(points[:, 0], points[:, 1], "b-")
        plt.scatter(points[:, 0], points[:, 1], color="r")
        plt.plot(building[:, 0], building[:, 1], "g-")  # Plot the building

        plt.arrow(
            points[0][0],
            points[0][1],
            test_vecs[i][0],
            test_vecs[i][1],
            head_width=0.5,
            head_length=0.5,
            fc="g",
            ec="g",
        )

        plt.arrow(
            points[0][0],
            points[0][1],
            pred_direction[0],
            pred_direction[1],
            head_width=0.5,
            head_length=0.5,
            fc="r",
            ec="r",
        )

        plt.title(f"Test Case {i+1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

    plt.axis('equal')
    plt.show()
