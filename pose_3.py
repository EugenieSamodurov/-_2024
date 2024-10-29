import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# Устройство для выполнения вычислений (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Загрузка весов модели
weights = torch.load('yolov7-w6-pose.pt')
model = weights['model']
model = model.half().to(device)
_ = model.eval()

# Путь к видеофайлу
video_path = 'draka2.mp4'

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Ошибка при попытке чтения видео. Пожалуйста, проверьте путь еще раз')

# Получение ширины и высоты кадров
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Подготовка первого кадра для получения размеров для записи видео
vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]

save_name = f"{video_path.split('/')[-1].split('.')[0]}"
# Определение кодека и создание объекта VideoWriter
out = cv2.VideoWriter(f"{save_name}_keypoint.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (resize_width, resize_height))

frame_count = 0  # Счетчик кадров
total_fps = 0  # Общая частота кадров

keypoint_positions = []  # Список для хранения координат ключевых точек

# Укажите индексы ключевых точек, которые хотите отслеживать (запястья рук)
keypoint_indices = [9, 10]  # Правое и левое запястья

# Порог для определения драки (на основе скорости движения ключевых точек)
movement_threshold = 50.0
previous_keypoints = None
fight_frames = []

while cap.isOpened():
    # Захват каждого кадра видео
    ret, frame = cap.read()
    if ret:
        orig_image = frame
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(device)
        image = image.half()

        # Начало измерения времени
        start_time = time.time()
        with torch.no_grad():
            output, _ = model(image)
        # Конец измерения времени
        end_time = time.time()
        # Вычисление FPS
        fps = 1 / (end_time - start_time)
        # Добавление FPS к общему количеству
        total_fps += fps
        # Увеличение счетчика кадров
        frame_count += 1

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

            # Отображение ограничивающих рамок вокруг людей (опционально)
            xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
            xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
            cv2.rectangle(
                nimg,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        # Отслеживание ключевых точек (запястий)
        if output.shape[0] > 0 and all([keypoint_index * 3 + 7 < output.shape[1] for keypoint_index in keypoint_indices]):
            current_keypoints = []
            for keypoint_index in keypoint_indices:
                keypoint_x, keypoint_y = output[0, 7 + keypoint_index * 3], output[0, 8 + keypoint_index * 3]
                current_keypoints.append((keypoint_x, keypoint_y))
                cv2.circle(nimg, (int(keypoint_x), int(keypoint_y)), 5, (0, 0, 255), -1)
            
            keypoint_positions.append(current_keypoints)

            if previous_keypoints is not None:
                # Вычисление движения ключевых точек
                movements = [np.sqrt((current_keypoints[i][0] - previous_keypoints[i][0])**2 +
                                     (current_keypoints[i][1] - previous_keypoints[i][1])**2)
                             for i in range(len(keypoint_indices))]
                average_movement = np.mean(movements)
                if average_movement > movement_threshold:
                    fight_frames.append(1)
                else:
                    fight_frames.append(0)
            else:
                fight_frames.append(0)
            previous_keypoints = current_keypoints
        else:
            fight_frames.append(0)

        # Вывод FPS на текущий кадр
        cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # Отображение изображения
        cv2.imshow('image', nimg)
        out.write(nimg)
        # Нажмите `q` для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождение захвата видео
cap.release()
# Закрытие всех окон
cv2.destroyAllWindows()
# Вычисление и вывод средней частоты кадров
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

# Сохранение информации о кадрах в файл
np.savetxt(f"{save_name}_fight_frames1.txt", fight_frames, fmt="%d", header="fight_frame")

print(f"Fight frames information has been saved to {save_name}_fight_frames.txt")
