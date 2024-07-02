from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

# Define colors and labels for each class
colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}  # Red for head, Green for helmet, Blue for person
labels = {0: "Head", 1: "Helmet", 2: "Person"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Load the image
        image = cv2.imread(filepath)
        
        # Detect objects in the image
        results = model(image)[0]
        
        # Draw bounding boxes and labels
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            color = colors[int(class_id)]
            label = labels[int(class_id)]
            
            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save the result image
        output_filepath = os.path.join('static', filename)
        cv2.imwrite(output_filepath, image)
        
        return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
