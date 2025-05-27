from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename
import time
import shutil

app = Flask(__name__)

# Define upload and annotated folders
UPLOAD_FOLDER = 'static/uploads'
ANNOTATED_IMAGE_FOLDER = 'static/annotated_images'
ANNOTATED_VIDEO_FRAMES_FOLDER = 'static/annotated_video_frames'
LIVE_STREAM_FOLDER = 'static/live_stream' # Not directly used for server-side live, but good for structure

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_IMAGE_FOLDER'] = ANNOTATED_IMAGE_FOLDER
app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'] = ANNOTATED_VIDEO_FRAMES_FOLDER
app.config['LIVE_STREAM_FOLDER'] = LIVE_STREAM_FOLDER

# Ensure folders exist at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_IMAGE_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_VIDEO_FRAMES_FOLDER, exist_ok=True)
os.makedirs(LIVE_STREAM_FOLDER, exist_ok=True)


# Load the YOLOv8 model
# This will download yolov8n.pt if it's not already present
model = YOLO('yolov8n.pt')

# Allowed extensions for file uploads
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def resize_image_for_inference(image, max_dim=640):
    """Resizes an image to have its longest side be max_dim, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def detect_objects_on_image(input_path, output_path):
    """Performs object detection on an image and saves the annotated image."""
    try:
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image from {input_path}")

        # Resize image before inference to save memory
        resized_image = resize_image_for_inference(image)

        # Perform inference
        results = model(resized_image) # Use resized_image for inference

        # Plot results on the image (YOLO's plot method handles annotation)
        # It's important to plot on the original image if you want original dimensions,
        # but for memory reasons, plotting on the resized image and then scaling back
        # or accepting smaller output is necessary. For simplicity, we'll plot on resized.
        for result in results:
            annotated_image = result.plot()

        cv2.imwrite(output_path, annotated_image)
        return True
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")
        return False

def detect_objects_on_video(input_path, output_folder):
    """
    Performs object detection on a video and saves annotated frames.
    Resizes each frame before inference to reduce memory usage.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {input_path}")

        frame_count = 0
        annotated_frame_paths = []

        # Clear previous frames in the output folder for this session
        shutil.rmtree(output_folder, ignore_errors=True)
        os.makedirs(output_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame before inference to save memory
            resized_frame = resize_image_for_inference(frame)

            results = model(resized_frame) # Use resized_frame for inference
            for result in results:
                annotated_frame = result.plot()

            frame_filename = f"frame_{frame_count:05d}.jpg"
            output_frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(output_frame_path, annotated_frame)
            annotated_frame_paths.append(url_for('static', filename=f'annotated_video_frames/{os.path.basename(output_folder)}/{frame_filename}'))
            frame_count += 1

        cap.release()
        return annotated_frame_paths
    except Exception as e:
        print(f"Error processing video {input_path}: {e}")
        return []

@app.route('/')
def index():
    # Clean up previous uploads and annotated files to manage space
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['ANNOTATED_IMAGE_FOLDER'], ignore_errors=True)
    shutil.rmtree(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], ignore_errors=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANNOTATED_IMAGE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], exist_ok=True)

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
    return "Invalid file type.", 400

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

        # Generate a unique folder for video frames to avoid clashes if multiple users
        # upload videos concurrently. This is still ephemeral storage.
        video_output_subfolder = os.path.join(app.config['ANNOTATED_VIDEO_FRAMES_FOLDER'], os.path.splitext(filename)[0] + '_' + str(int(time.time())))
        os.makedirs(video_output_subfolder, exist_ok=True)

        annotated_frame_urls = detect_objects_on_video(input_path, video_output_subfolder)
        if annotated_frame_urls:
            # Pass a list of URLs to the template for display
            return render_template('video_result.html', frame_urls=annotated_frame_urls)
        else:
            return "Error processing video.", 500
    return "Invalid file type.", 400

@app.route('/live')
def live_detection():
    # As previously stated, direct live webcam access for server-side processing
    # in a hosted environment like Render is complex and generally not feasible
    # without advanced streaming setups (e.g., WebSockets, RTMP).
    # For a simple Flask app on Render, this functionality is best implemented
    # client-side using JavaScript and then sending images/frames to the server
    # for processing (which would be a separate API endpoint).
    return render_template('live_unsupported.html')

if __name__ == '__main__':
    # Use 0.0.0.0 to bind to all available network interfaces
    # Render provides the port via the $PORT environment variable.
    # For local development, you can use a fixed port like 5000 or 10000.
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
