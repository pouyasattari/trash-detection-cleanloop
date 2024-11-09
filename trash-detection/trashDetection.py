from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='web')
CORS(app)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Initialize YOLO model
model = YOLO("yolov8x.pt")

def process_image_with_yolo(image_file):
    """Process image with YOLOv8 and return detections and annotated image"""
    try:
        # Read image file
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Multi-scale detection
        scales = [1.0, 0.8, 1.2]  # Different scales for detection
        all_detections = []
        
        for scale in scales:
            # Resize image for different scales
            if scale != 1.0:
                width = int(image.shape[1] * scale)
                height = int(image.shape[0] * scale)
                scaled_image = cv2.resize(image, (width, height))
            else:
                scaled_image = image

            # Run YOLOv8 inference with lower confidence threshold
            results = model(scaled_image, conf=0.15, iou=0.3)  # Lower confidence threshold, higher IOU
            
            # Get detections
            for r in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = r
                class_name = model.names[int(cls)]
                
                # Adjust coordinates back to original scale
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                
                # Add detection
                all_detections.append({
                    'name': class_name,
                    'confidence': float(conf) * 100,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'area': float((x2 - x1) * (y2 - y1))
                })

        # Remove overlapping detections
        filtered_detections = []
        all_detections.sort(key=lambda x: x['area'], reverse=True)  # Sort by area, largest first
        
        def calculate_iou(box1, box2):
            x1 = max(box1['x1'], box2['x1'])
            y1 = max(box1['y1'], box2['y1'])
            x2 = min(box1['x2'], box2['x2'])
            y2 = min(box1['y2'], box2['y2'])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
            box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
            
            return intersection / (box1_area + box2_area - intersection)

        # Filter overlapping boxes
        for detection in all_detections:
            should_keep = True
            for filtered in filtered_detections:
                if calculate_iou(detection['bbox'], filtered['bbox']) > 0.5:
                    should_keep = False
                    break
            if should_keep:
                filtered_detections.append(detection)

        # Draw boxes on image
        for det in filtered_detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Color based on confidence
            color = (0, int(255 * (confidence/100)), int(255 * (1-confidence/100)))
            
            # Draw rectangle with thickness based on confidence
            thickness = max(1, int((confidence/100) * 4))
            cv2.rectangle(image, 
                        (int(bbox['x1']), int(bbox['y1'])), 
                        (int(bbox['x2']), int(bbox['y2'])), 
                        color, thickness)
            
            # Add label with background
            label = f"{det['name']}"
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Draw label background
            cv2.rectangle(image,
                        (int(bbox['x1']), int(bbox['y1'] - text_height - 10)),
                        (int(bbox['x1'] + text_width), int(bbox['y1'])),
                        color, -1)
            
            # Draw text
            cv2.putText(image, label,
                       (int(bbox['x1']), int(bbox['y1'] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), font_thickness)

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', image)
        annotated_image = base64.b64encode(buffer).decode('utf-8')
        
        return filtered_detections, f"data:image/jpeg;base64,{annotated_image}"
    
    except Exception as e:
        print(f"Error in YOLO processing: {str(e)}")
        return [], None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_trash():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(image_file)
        image_url = upload_result['secure_url']
        
        # Reset file pointer
        image_file.seek(0)
        
        # Process with YOLO
        detections, annotated_image = process_image_with_yolo(image_file)
        
        # Calculate cleanup estimates
        item_count = len(detections) or 1
        # Adjust cleanup estimates based on detection areas
        total_area = sum(det['area'] for det in detections)
        avg_area = total_area / item_count
        
        # Scale people needed based on total area and item count
        area_threshold = 640 * 480  # Adjust this threshold as needed
        if total_area > area_threshold and item_count > 10:
            people_needed = max(1, int((total_area / area_threshold) * 2))
        else:
            people_needed = 1
        
        # Adjust hours needed based on item count
        min_items_per_person_per_hour = 40  # Adjust this value as needed
        hours_needed = max(round(item_count / (min_items_per_person_per_hour * people_needed), 2), 0.05)

        result = {
            'status': 'success',
            'trash_items': detections,
            'cleanup_estimate': {
                'people_needed': people_needed,
                'hours_needed': hours_needed,
                'total_items': item_count,
                'total_area': round(total_area, 2),
                'average_item_area': round(avg_area, 2)
            },
            'original_image_url': image_url,
            'annotated_image': annotated_image or image_url
        }
        
        return jsonify(result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)