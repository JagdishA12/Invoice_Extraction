# import matplotlib.pyplot as plt
# import cv2
# from ultralytics import YOLO

# # Load a pre-trained YOLOv8 model
# model = YOLO(r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\best.pt")

# # Path to the image
# image_path = r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\test.png"

# # Perform inference
# results = model(image_path)

# # Get the bounding boxes, confidences, and class labels
# boxes = results[0].boxes.xyxy  # x1, y1, x2, y2
# confidences = results[0].boxes.conf
# class_labels = results[0].boxes.cls

# # Load the image using OpenCV
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# # Plot the image with bounding boxes
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# ax = plt.gca()

# # Define a list of class names (for COCO dataset, modify if custom classes)
# class_names = model.names

# # Loop through all detections
# for box, conf, cls in zip(boxes, confidences, class_labels):
#     x1, y1, x2, y2 = box
#     width, height = x2 - x1, y2 - y1
#     class_name = class_names[int(cls)]
#     label = f"{class_name}: {conf:.2f}"

#     # Create a rectangle patch
#     rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)

#     # Add the label
#     plt.text(x1, y1, label, color='white', fontsize=12, bbox=dict(facecolor='red', edgecolor='red', alpha=0.5))

# # Show the plot with bounding boxes
# plt.axis('off')
# plt.show()




import fitz  # PyMuPDF

def visualize_boxes_on_pdf(pdf_path, page_number, bounding_boxes, output_path):
    # Open the PDF document
    document = fitz.open(pdf_path)
    
    # Select the specific page
    page = document[page_number - 1]
    
    # Iterate through bounding boxes and draw rectangles on the page
    for box in bounding_boxes:
        x1, y1, x2, y2 = box  # Assuming the box is in [x1, y1, x2, y2] format

        # Convert PDF coordinates to points
        #x1_pt, y1_pt, x2_pt, y2_pt = x1, page.rect.y1 - y2, x2, page.rect.y1 - y1
        
        # Draw a rectangle on the page
        rect = fitz.Rect(x1, y1, x2, y2)
        page.draw_rect(rect)
    
    # Save the modified PDF
    if output_path:
        document.save(output_path)
        print(f"Annotated PDF saved at: {output_path}")
    else:
        print("No output path provided. Annotations will not be saved.")
    
    # Close the document
    document.close()

# Example usage
pdf_path = r"C:\Users\jagdi\Downloads\Order_Invoice5935218061.pdf"
output_path= r"C:\Users\jagdi\OneDrive\Desktop\Invoice_Project\TestInv.pdf"
page_number = 1
bounding_boxes = [
    (35.61538977107508, 327.96923873233754, 582.8851874018426, 397.0046839842135), (534.309074786023, 437.60966983568943, 582.6350643656173, 457.58792533497206), (32.36849953839868, 300.89553567768473, 581.8477528830803, 326.05136794699996), (295.94675910539127, 438.2608298846784, 343.668228799846, 456.80339037622997), (390.9793699159843, 438.3350389399333, 441.3486812088901, 457.12085378530594), (109.44643084105142, 200.93029323475093, 176.61690211418278, 213.48024366252488), (105.26237933582551, 190.6874754828934, 218.16887345179077, 201.77457380623528), (136.60136166702168, 224.24015999780283, 216.95518540427238, 234.59398619662483), (442.09149115195737, 418.9940292917794, 488.1440007137635, 436.3919029005863), (539.1532197598393, 417.930783185546, 579.8089118042224, 436.67909682659774) # Example coordinates [x_center, y_center, width, height]
]

# Visualize bounding boxes on the PDF and display it
visualize_boxes_on_pdf(pdf_path, page_number, bounding_boxes, output_path)

