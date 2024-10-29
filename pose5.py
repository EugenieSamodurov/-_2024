import torch
import cv2
import numpy as np
import time

from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Устройство для выполнения вычислений (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Загрузка весов модели
weights = torch.load('yolov7-w6-pose.pt')
model = weights['model']
model = model.half().to(device)
_ = model.eval()

# Путь к видеофайлу
video_path = 'draka21.mp4'

# Открытие видеофайла
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Ошибка при попытке чтения видео. Пожалуйста, проверьте путь еще раз')

# Получение ширины и высоты кадров
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Определение целевого размера, кратного 64
target_size = (frame_width // 64) * 64, (frame_height // 64) * 64

# Подготовка первого кадра для получения размеров для записи видео
vid_write_image = letterbox(cap.read()[1], target_size, stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]

save_name = f"{video_path.split('/')[-1].split('.')[0]}"

# Определение кодека и создание объекта VideoWriter
out = cv2.VideoWriter(f"{save_name}_last.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (resize_width, resize_height))

# Переменные для отслеживания координат и идентификаторов людей
people_tracks = []  # Список для хранения координат людей на каждом кадре
person_id_counter = 0  # Счётчик для присвоения уникальных идентификаторов
trackers = []  # Список трекеров Калмана

movement_threshold = 50.0  # Порог для определения драки (на основе скорости движения ключевых точек)

# Переменные для подсчета FPS
total_fps = 0
frame_count = 0

# Функция создания Калманова фильтра для отслеживания
def create_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.zeros((4, 1))
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R = np.eye(2) * 10
    kf.Q = np.eye(4)
    return kf

while cap.isOpened():
    # Захват каждого кадра видео
    ret, frame = cap.read()
    if ret:
        orig_image = frame
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        # Использование letterbox для изменения размера изображения до ближайшего кратного 64
        image = letterbox(image, target_size, stride=64, auto=True)[0]
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
        total_fps += fps
        frame_count += 1

        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        current_people = []  # Список для хранения текущих координат людей на кадре
        keypoints_list = []  # Список всех ключевых точек текущего кадра

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

            # Считывание всех ключевых точек
            keypoints = []
            for keypoint_index in range(0, output.shape[1] - 7, 3):
                keypoint_x, keypoint_y = output[idx, 7 + keypoint_index], output[idx, 8 + keypoint_index]
                keypoints.append((keypoint_x, keypoint_y))
                cv2.circle(nimg, (int(keypoint_x), int(keypoint_y)), 5, (0, 0, 255), -1)

            keypoints_list.append(keypoints)

            current_people.append({
                'keypoints': keypoints,
                'bbox': (int(xmin), int(ymin), int(xmax), int(ymax))  # Приведение к int
            })

        # Прогнозирование следующего состояния трекеров
        for tracker in trackers:
            tracker.predict()

        # Создание матрицы стоимости для сопоставления
        if len(trackers) > 0 and len(keypoints_list) > 0:
            cost_matrix = np.zeros((len(trackers), len(keypoints_list)), dtype=np.float32)
            for t, tracker in enumerate(trackers):
                for d, keypoints in enumerate(keypoints_list):
                    predicted_state = tracker.x[:2].T[0]
                    keypoint_center = np.mean(keypoints, axis=0)
                    cost_matrix[t, d] = np.linalg.norm(predicted_state - keypoint_center)

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_detections = set()

            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] < movement_threshold:
                    trackers[r].update(np.mean(keypoints_list[c], axis=0))
                    current_people[c]['id'] = r
                    assigned_tracks.add(r)
                    assigned_detections.add(c)

            # Обновление трекеров, которым не назначены детекции
            unassigned_tracks = set(range(len(trackers))) - assigned_tracks
            for t in unassigned_tracks:
                trackers[t].update(tracker.x[:2].T[0])

            # Добавление новых трекеров для неназначенных детекций
            unassigned_detections = set(range(len(keypoints_list))) - assigned_detections
            for d in unassigned_detections:
                kf = create_kalman_filter()
                kf.x[:2] = np.mean(keypoints_list[d], axis=0).reshape(2, 1)
                trackers.append(kf)
                current_people[d]['id'] = len(trackers) - 1

        else:
            # Создание трекеров, если их нет или нет детекций
            for d in range(len(keypoints_list)):
                kf = create_kalman_filter()
                kf.x[:2] = np.mean(keypoints_list[d], axis=0).reshape(2, 1)
                trackers.append(kf)
                current_people[d]['id'] = len(trackers) - 1

        # Добавление уникальных ID к текущим людям и отображение их на изображении
        for person in current_people:
            person_id = person['id']
            cv2.putText(nimg, f"ID: {person_id}", (int(person['bbox'][0]), int(person['bbox'][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Сохранение списка текущих людей для текущего кадра
        people_tracks.append(current_people)

        # Вывод FPS на текущий кадр
        # cv2.putText(nimg, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    # 1, (0, 255, 0), 2)
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
