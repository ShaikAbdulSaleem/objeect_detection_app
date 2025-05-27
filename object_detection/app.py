from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename
import time
import shutil
import numpy as np
from PIL import Image # Added for image compression

app = Flask(__name__)

# Define upload and annotated folders
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_IMAGE_FOLDER = 'static/annotated_images'
ANNOTATED_VIDEO_FRAMES_FOLDER = 'static/annotated_video_frames'
COMPRESSED_IMAGE_FOLDER = 'static/compressed_images' # New folder for compressed images
LIVE_STREAM_FOLDER = 'static/live_stream'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_IMAGE_FOLDER'] = ANNOTATED_IMAGE_FOLDER
app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'] = ANNOTATED_VIDEO_FRAMES_FOLDER
app.config['COMPRESSED_IMAGE_FOLDER'] = COMPRESSED_IMAGE_FOLDER # Register new folder
app.config['LIVE_STREAM_FOLDER'] = LIVE_STREAM_FOLDER

# Ensure folders exist at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_IMAGE_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_VIDEO_FRAMES_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_IMAGE_FOLDER, exist_ok=True) # Create new folder
os.makedirs(LIVE_STREAM_FOLDER, exist_ok=True)


# --- Model Loading ---
yolo_model = YOLO('yolov8n.pt')

FACE_PROTO = 'models/deploy.prototxt'
FACE_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

GENDER_PROTO = 'models/deploy_gender.prototxt'
GENDER_MODEL = 'models/gender_net.caffemodel'
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.426377603, 87.768914374, 114.895847746)
# --- End Model Loading ---


# Allowed extensions for file uploads
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
# For compression, we will only allow images
ALLOWED_COMPRESSION_EXTENSIONS = {'png', 'jpg', 'jpeg'} # GIF might be tricky to compress this way

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def resize_image_for_inference(image, max_dim=640):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def detect_gender_and_annotate(image_cv2):
    frame_copy = image_cv2.copy()
    h, w = frame_copy.shape[:2]
    
    face_blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), (104, 117, 123), swapRB=False, crop=False)
    face_net.setInput(face_blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, h) # Fixed: endY should be min(h, endY) not min(h,h)

            face_roi = frame_copy[startY:endY, startX:endX]
            if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                continue

            gender_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            label = f"Gender: {gender}"
            cv2.rectangle(frame_copy, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame_copy, label, (startX, startY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
    return frame_copy

def detect_objects_on_image(input_path, output_path):
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")

        resized_image_yolo = resize_image_for_inference(image.copy())
        
        results = yolo_model(resized_image_yolo)

        annotated_image_yolo = resized_image_yolo.copy()
        for result in results:
            annotated_image_yolo = result.plot()
            
        final_annotated_image = detect_gender_and_annotate(annotated_image_yolo)
        
        cv2.imwrite(output_path, final_annotated_image)
        return True
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return False

def detect_objects_on_video(input_path, output_folder):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {input_path}")

        frame_count = 0
        annotated_frame_urls = []

        shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame_yolo = resize_image_for_inference(frame.copy())

            results = yolo_model(resized_frame_yolo)
            annotated_frame_yolo = resized_frame_yolo.copy()
            for result in results:
                annotated_frame_yolo = result.plot()
            
            final_annotated_frame = detect_gender_and_annotate(annotated_frame_yolo)

            frame_filename = f"frame_{frame_count:05d}.jpg"
            output_frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(output_frame_path, final_annotated_frame)
            annotated_frame_urls.append(url_for('static', filename=f'annotated_video_frames/{os.path.basename(output_folder)}/{frame_filename}'))
            frame_count += 1

        cap.release()
        return annotated_frame_urls
    except Exception as e:
        print(f"Error processing video {input_path}: {e}")
        return []

def compress_image(input_path, output_path, quality=80):
    """Compresses an image using Pillow and saves it."""
    try:
        img = Image.open(input_path)
        # Convert to RGB if not already (important for JPEG compression)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        img.save(output_path, quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Error compressing image {input_path}: {e}")
        return False


@app.route('/')
def index():
    # Clean up previous uploads and annotated files to manage space
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['ANNOTATED_IMAGE_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['COMPRESSED_IMAGE_FOLDER'], ignore_errors=True) # Clear compressed images too
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANNOTATED_IMAGE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['COMPRESSED_IMAGE_FOLDER'], exist_ok=True) # Recreate compressed images folder

    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['ANNOTATED_IMAGE_FOLDER'], filename)
        file.save(input_path)

        if detect_objects_on_image(input_path, output_path):
            return render_template('result.html', image_path=url_for('static', filename=f'annotated_images/{filename}'))
        else:
            return "Error processing image.", 500
    return f"Invalid file type. Please upload a valid image file (allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}).", 400

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        video_output_subfolder = os.path.join(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], os.path.splitext(filename)[0] + '_' + str(int(time.time())))
        os.makedirs(video_output_subfolder, exist_ok=True)

        annotated_frame_urls = detect_objects_on_video(input_path, video_output_subfolder)
        if annotated_frame_urls:
            return render_template('video_result.html', frame_urls=annotated_frame_urls)
        else:
            return "Error processing video.", 500
    return f"Invalid file type. Please upload a valid video file (allowed types: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}).", 400


@app.route('/compress-image', methods=['POST'])
def compress_image_route():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename, ALLOWED_COMPRESSION_EXTENSIONS):
        filename = secure_filename(file.filename)
        # Ensure filename has a common image extension like .jpg for compression quality setting
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_compressed.jpg" # Always save as JPG for best compression
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['COMPRESSED_IMAGE_FOLDER'], output_filename)
        file.save(input_path) # Save original to process
        
        # You can adjust the quality (0-100). 80 is a good balance.
        if compress_image(input_path, output_path, quality=80):
            return render_template('compress_result.html', 
                                   compressed_image_path=url_for('static', filename=f'compressed_images/{output_filename}'),
                                   original_filename=filename,
                                   compressed_filename=output_filename)
        else:
            return "Error compressing image.", 500
    return f"Invalid file type for compression. Please upload a valid image file (allowed types: {', '.join(ALLOWED_COMPRESSION_EXTENSIONS)}).", 400


@app.route('/live')
def live_detection():
    return render_template('live_unsupported.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
