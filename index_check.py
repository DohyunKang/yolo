import cv2
from ultralytics import YOLO

# 클래스 이름 설정
class_names = ["car", "parkingspace", "plate"]

# YOLO 모델로 감지된 결과에서 번호판 및 주차공간 위치를 사각형으로 표시하는 함수
def draw_positions(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 박스 좌표와 라벨 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            if cls_idx < len(class_names):
                label = class_names[cls_idx]

                if label == "plate":
                    # plate 박스 그리기 및 터미널 출력
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    position_text = f"plate: ({x1}, {y1}), ({x2}, {y2})"
                    cv2.putText(image, position_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print(position_text)

                elif label == "parkingspace":
                    # parkingspace 박스 그리기 및 터미널 출력
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    position_text = f"parkingspace: ({x1}, {y1}), ({x2}, {y2})"
                    cv2.putText(image, position_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    print(position_text)

    return image

def predict_webcam_positions(model_path):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 웹캠 열기
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 예측
        results = model.predict(source=frame, save=False, imgsz=640)

        # 인식된 plate 및 parkingspace 위치에 사각형 표시
        frame = draw_positions(frame, results)

        # 이미지 저장
        filename = f'position_frame_{frame_count}.png'
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'best.pt'  # YOLO 모델의 경로
    predict_webcam_positions(model_path)
