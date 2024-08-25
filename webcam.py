import cv2
import json
import difflib
import pytesseract
from ultralytics import YOLO
import math
import os
import requests

# Tesseract OCR 경로 설정 (Windows의 경우)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 클래스 이름 설정
class_names = ["car", "parkingspace", "plate"]

# 파킹스페이스 위치와 인덱스 (제공된 좌표)
parking_space_positions = [
    ((373, 247), (487, 368)),  # 1
    ((260, 249), (341, 368)),  # 2
    ((110, 252), (237, 372)),  # 3
    ((359, 176), (432, 238)),  # 4
    ((277, 178), (333, 241)),  # 5
    ((181, 179), (257, 243))   # 6
]

# plate의 위치와 인덱스 (y축 범위를 여유롭게 조정한 좌표들)
plate_positions = [
    ((382, 306), (484, 340)),  # 1: 오른쪽 아래 (68오 8269)
    ((242, 328), (343, 356)),  # 2: 가운데 아래 (35우 9235)
    ((104, 331), (210, 360)),  # 3: 왼쪽 아래 (56가 9683)
    ((362, 201), (433, 224)),  # 4: 오른쪽 위 (52다 3682)
    ((258, 200), (329, 221)),  # 5: 가운데 위 (23마 3562)
    ((172, 201), (245, 223))   # 6: 왼쪽 위 (33라 6538)
]

# 실제 번호판 데이터 (추가적으로 입력해줌)
actual_plate_texts = [
    "52다 3682",
    "23마 3562",
    "35우 9235",
    "68오 8269",
    "33라 6538",
    "56가 9683"
]

# 이전 상태를 저장할 전역 변수
previous_json_data = {"free": [], "plates": [], "outsider": []}

# OCR 결과와 실제 번호판 텍스트를 비교하여 유사하면 실제 번호판 텍스트로 대체
def match_plate_text(ocr_text, actual_texts, threshold=0.6):
    best_match = None
    best_ratio = 0.0
    for actual in actual_texts:
        ratio = difflib.SequenceMatcher(None, ocr_text, actual).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = actual
    if best_ratio >= threshold:
        return best_match
    else:
        return ocr_text

# 특정 위치와 비슷한지 확인하는 함수
def is_similar_position(pos1, pos2, tolerance=50):  # tolerance 값을 늘림
    (x1_1, y1_1), (x2_1, y2_1) = pos1
    (x1_2, y1_2), (x2_2, y2_2) = pos2

    # 중심점 계산
    center_1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
    center_2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)

    # 중심점 간 거리 계산
    distance = math.sqrt((center_1[0] - center_2[0]) ** 2 + (center_1[1] - center_2[1]) ** 2)

    return distance <= tolerance

# YOLO 모델로 감지된 결과에서 번호판 및 주차공간을 인식하고 JSON 데이터를 생성하는 함수
def draw_predictions_and_create_json(image, results):
    global previous_json_data
    json_data = {"free": [], "plates": [], "outsider": []}
    status_changed = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 박스 좌표와 라벨 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            if cls_idx < len(class_names):
                label = class_names[cls_idx]

                if label == "plate":
                    for idx, pos in enumerate(plate_positions, start=1):
                        if is_similar_position(((x1, y1), (x2, y2)), pos):
                            # 번호판 영역 추출
                            plate_img = image[y1:y2, x1:x2]

                            # OCR 수행
                            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8', lang='kor').strip()

                            # 유사한 실제 번호판 텍스트와 비교
                            matched_text = match_plate_text(plate_text, actual_plate_texts)

                            # 외부 차량 여부 확인
                            if matched_text not in actual_plate_texts:
                                print(f"OCR result: {plate_text} > Recognized as outsider: {matched_text} (Index: {idx})")
                                json_data["outsider"].append({"outsider": matched_text, "index": idx})
                            else:
                                print(f"OCR result: {plate_text} > Matched with: {matched_text} (Index: {idx})")
                                json_data["plates"].append({"plate_text": matched_text, "index": idx})

                            # OCR 결과를 이미지에 표시
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image, matched_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            break

                elif label == "parkingspace":
                    for idx, pos in enumerate(parking_space_positions, start=1):
                        if is_similar_position(((x1, y1), (x2, y2)), pos):
                            # 파킹스페이스 인식 및 JSON 데이터 생성
                            print(f"Parkingspace {idx}: ({x1}, {y1}), ({x2, y2})")
                            json_data["free"].append({"free": idx})

                            # 파킹스페이스 결과를 이미지에 표시
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(image, f"Parkingspace {idx}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            break

    # JSON 데이터 변화를 확인하여 상태가 변경되었는지 확인
    if json_data != previous_json_data:
        status_changed = True
        previous_json_data = json_data

    return image, json_data, status_changed

def send_data_to_server(json_data, image_path):
    url = "http://127.0.0.1:8000/parking_status/update_parking_data/"  # 서버 URL로 변경
    files = {
        'json_data': (None, json.dumps(json_data), 'application/json'),
        'image': open(image_path, 'rb')
    }
    response = requests.post(url, files=files)
    print(f"Server response: {response.status_code}, {response.text}")

def predict_webcam_ocr(model_path):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 웹캠 열기
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0

    # 이미지 저장을 위한 디렉토리 경로 설정
    save_dir = os.path.join(os.getcwd(), 'image')
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 이미지를 복사하여 보관
        original_frame = frame.copy()

        # YOLO 예측
        results = model.predict(source=frame, save=False, imgsz=640)

        # OCR 결과 추출 및 JSON 데이터 생성
        processed_frame, json_data, status_changed = draw_predictions_and_create_json(frame, results)

        if status_changed:
            # YOLO 처리 전의 원본 이미지 저장
            original_image_path = os.path.join(save_dir, f'original_image_{frame_count}.png')
            cv2.imwrite(original_image_path, original_frame)
            print(f"Original image saved as {original_image_path}")

            # YOLO 처리된 이미지 저장
            processed_image_path = os.path.join(save_dir, f'processed_image_{frame_count}.png')
            cv2.imwrite(processed_image_path, processed_frame)
            print(f"Processed image saved as {processed_image_path}")

            # 서버로 JSON 데이터 및 이미지를 전송
            send_data_to_server(json_data, original_image_path)

            frame_count += 1  # 상태 변경이 있을 때만 증가

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'best.pt'  # YOLO 모델의 경로
    predict_webcam_ocr(model_path)
