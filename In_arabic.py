import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np

DATA_PATH = 'D:/Coding/python/Alexandria_Sign/Alexandria_Sign.v1i.yolov8/data.yaml'
MODEL_PATH = 'runs/detect/train2/weights/best.pt'
VIDEO_SOURCE = 0
ARABIC_FONT_PATH = "arial.ttf"

print("تحميل نموذج YOLOv8...")
model = YOLO('yolov8n.pt')

arabic_names = {
    0: "أبو تلاتة",
    1: "المنشية - محطة الرمل - سيدي بشر",
    2: "الساعة",
    3: "بحري - المندرة - ميامي",
    4: "خمسة وأربعين",
    5: "موقف",
    6: "واحد وعشرين",
    7: "فيكتوريا"
}

fontpath = "arial.ttf"
font = ImageFont.truetype(fontpath, 32)

colors = {
    0: (110, 150, 200),
    1: (150, 100, 150),
    2: (200, 20, 50),
    3: (255, 165, 0),
    4: (170, 120, 40),
    5: (128, 50, 200),
    6: (70, 120, 90),
    7: (200, 20, 100)
}

def draw_text(image, text, position, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    draw.text(position, bidi_text, font=font, fill=color)

    image = np.array(image_pil)
    return image

def train_model():
    print("بدء تدريب النموذج...")
    model.train(data=DATA_PATH, epochs=50, imgsz=640)
    print("التدريب انتهى بنجاح.")

def export_model():
    print("تصدير النموذج إلى صيغة ONNX...")
    model.export(format='onnx')
    print("تم تصدير النموذج بنجاح.")

def real_time_detection():
    print("بدء الكشف في الوقت الحقيقي...")
    trained_model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("فشل في قراءة الصورة من الكاميرا.")
            break

        results = trained_model.predict(source=frame, imgsz=640, conf=0.5)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, class_id in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                color = colors.get(class_id, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if class_id in arabic_names:
                    frame = draw_text(frame, arabic_names[class_id], (x1, y1 - 30), color)

        cv2.imshow("الكشف في الوقت الحقيقي", frame)
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
        print("[ERROR] خيار غير صحيح!")
