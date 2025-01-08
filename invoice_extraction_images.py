import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO(r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\best.pt")

# Path to the image
image_path = r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\test\images\2890-Invoice-pdf_jpg.rf.72c00715231442900ca3ca51646b3ed5.jpg"

# Perform inference
results = model(image_path)

# Get the bounding boxes, confidences, and class labels
boxes = results[0].boxes.xyxy  # x1, y1, x2, y2
confidences = results[0].boxes.conf
class_labels = results[0].boxes.cls

# Load the image using OpenCV
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Plot the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

# Define a list of class names (for COCO dataset, modify if custom classes)
class_names = model.names

# Loop through all detections
for box, conf, cls in zip(boxes, confidences, class_labels):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    class_name = class_names[int(cls)]
    label = f"{class_name}: {conf:.2f}"

    # Create a rectangle patch
    rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add the label
    plt.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='red', edgecolor='red', alpha=0.5))

# Show the plot with bounding boxes
plt.axis('off')
plt.show()








import pytesseract
import cv2
from ultralytics import YOLO
import numpy as np
import easyocr

# Set the path for Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
#reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU and want to use it

def preprocess_image(img):
    #img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    #ret, thresh1= cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return img

# Function to extract text from a given ROI using Tesseract OCR
def extract_text_from_roi(roi):
    # Perform OCR on the image
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text.strip()

# Function to extract text from a given ROI using easyocr
#def extract_text_from_roi(roi):
    # Perform OCR on the image
   # result = reader.readtext(roi)
   # extracted_text = ' '.join([text[1] for text in result])
   # return extracted_text.strip()

# Function to get text from bounding boxes given by YOLO model
def get_text(image):
    results = model(image)  # Perform inference with YOLO model
    extracted_text = {}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes and move to CPU
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Extract class IDs and move to CPU
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]  # Get class names
        
        for box, class_name in zip(boxes, class_names):
            x_min, y_min, x_max, y_max = map(int, box[:4])  # Extract bounding box coordinates
            roi = image[y_min:y_max, x_min:x_max]  # Crop the region of interest (ROI) from the image

            # Preprocess the ROI
            preprocessed_roi = preprocess_image(roi)
            
            # Convert the ROI to text
            extracted_text_for_class = extract_text_from_roi(preprocessed_roi)
            
            # Append the extracted text to the corresponding class in the dictionary
            if class_name in extracted_text:
                extracted_text[class_name] += " " + extracted_text_for_class
            else:
                extracted_text[class_name] = extracted_text_for_class

    return extracted_text

# Load the image using OpenCV
image_path = r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\test\images\2890-Invoice-pdf_jpg.rf.72c00715231442900ca3ca51646b3ed5.jpg"
image = cv2.imread(image_path)

# Get the text from the image
extracted_text = get_text(image)

# Print the extracted text for each class
for class_name, text in extracted_text.items():
    print(f"Class: {class_name}\nText: {text}\n{'-' * 20}")