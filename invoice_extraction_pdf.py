from ultralytics import YOLO
import fitz  # PyMuPDF
import torch
from PIL import Image

# Load a pre-trained YOLOv8 model
model = YOLO(r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\best.pt")
pdf_path = r"C:\Users\jagdi\Downloads\invoice_845.pdf"
image_path = r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\test.png"

def pdf_to_jpg(pdf_path, jpg_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Select the first page of the PDF
    page = pdf_document.load_page(0)

    # Convert the PDF page to an image (PNG) and save it as JPG
    pix = page.get_pixmap()
    
    # Convert the pixmap to a PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the PIL Image as JPG
    img.save(jpg_path, 'JPEG')

    # Get page width and height in points
    width_pt = page.rect.width
    height_pt = page.rect.height

    # Close the PDF file
    pdf_document.close()

    return width_pt, height_pt, img.size

pdf_width, pdf_height, img_size= pdf_to_jpg(pdf_path,image_path)
# Perform inference
results = model(image_path)

dict = {
    0:'Discount_Percentage',
    1:'Due_Date',
    2:'Email_Client',
    3:'Name_Client',
    4:'Products',
    5:'Remise',
    6:'Subtotal',
    7:'Tax',
    8:'Tax_Percentage',
    9:'Tel_Client',
    10:'Billing address',
    11:'Header',
    12:'Invoice date',
    13:'Invoice number',
    14:'Shipping address',
    15:'Total'
}

def yolo_to_pdf_coords(yolo_boxes, img_size, pdf_width, pdf_height):
    img_width, img_height = img_size
    
    pdf_boxes = []
    for box in yolo_boxes:
        x1, y1, x2, y2 = box

        # Convert image coordinates to PDF coordinates
        x1_pdf = x1 * pdf_width / img_width
        y1_pdf = (y1 * pdf_height / img_height)
        x2_pdf = x2 * pdf_width / img_width
        y2_pdf = (y2 * pdf_height / img_height)

        pdf_boxes.append((x1_pdf, y1_pdf, x2_pdf, y2_pdf))
    
    return pdf_boxes

tensor=results[0].boxes.xyxy
# Convert to list of lists
list_of_lists = tensor.tolist()

# Convert to list of tuples
yolo_boxes = [tuple(box) for box in list_of_lists]

#original_dim = (596, 842)  # Original image size
pdf_boxes = yolo_to_pdf_coords(yolo_boxes, img_size, pdf_width, pdf_height)
#print(pdf_boxes)

# Function to extract text from bounding boxes
def extract_text_from_bounding_box(pdf_path, bounding_boxes):
    document = fitz.open(pdf_path)
    page = document[0]
    extracted_texts = []

    for box in bounding_boxes:
        x1, y1, x2, y2 = box  # Assuming the box is in [x1, y1, x2, y2] format
        rect = fitz.Rect(x1, y1, x2, y2)
        text = page.get_text("text", clip=rect)
        extracted_texts.append(text.strip())

    return extracted_texts

# Example usage
texts = extract_text_from_bounding_box(pdf_path, pdf_boxes)

tens=results[0].boxes.cls
# Convert tensor elements to integers
tensor_int = tens.int()

# Convert to list of lists
classes = tensor_int.tolist()

result_dict = {}
for i, (cls, text) in enumerate(zip(classes, texts)):
    result_dict[dict[cls]] = text
    print(f"{dict[cls]}: {text}")
    print("---------")
