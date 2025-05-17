import os
import cv2
from ultralytics import YOLO

DATA_PATH = 'D:\Coding\python\Alexandria_Sign\Alexandria_Sign.v1i.yolov8\data.yaml'
MODEL_PATH = 'runs/detect/train2/weights/best.pt'
VIDEO_SOURCE = 0  #IF you want video

print("Loading YOLOv8 model")
model = YOLO('yolov8n.pt')

def train_model():
    print("Starting Training...")
    model.train(data=DATA_PATH, epochs=50, imgsz=640)
    print("Training Completed")

def export_model():
    print("Exporting the model to ONNX format...")
    model.export(format='onnx')
    print("Model exported successfully.")

def real_time_detection():
    print("Starting Real-Time Detection...")

    trained_model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        results = trained_model.predict(source=frame, imgsz=640, conf=0.5)

        annotated_frame = results[0].plot()
        cv2.imshow("Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose operation:")
    print("1️⃣ -> Model training")
    print("2️⃣ -> Export model")
    print("3️⃣ -> Turn On Real-Time Detection")
    choice = input("Enter the Operation number:")

    if choice == '1':
        train_model()
    elif choice == '2':
        export_model()
    elif choice == '3':
        real_time_detection()
    else:
        print("[ERROR]!")
