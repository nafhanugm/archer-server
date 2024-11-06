from flask import Flask, render_template, request, jsonify, send_file,send_from_directory
import cv2
import numpy as np
import os
import uuid
from pprint import pp
from services.velocity.velocity import process_velocity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = os.path.join(os.getcwd(), 'static', 'output')
processing_status = {}

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Check if a file is uploaded
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files['video']

    threshold = request.form.get("y_threshold", 60)

    # Generate a unique code for this video
    code = str(uuid.uuid4())
    processing_status[code] = "processing"

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the video asynchronously and update the status
    output_image_path = process_video(filepath, code, y_difference_threshold=int(threshold))

    # Return the processing status and code
    return jsonify({"status": "processing", "code": code}), 202

@app.route('/api/result', methods=['POST'])
def api_result():
    if request.is_json:
        data = request.get_json()
        code = data.get("code")
        
        if code not in processing_status:
            return jsonify({"status": "error", "message": "Invalid code"}), 404

    # Check if processing is complete
   # Check if processing is complete
        if processing_status[code] == "complete":
            output_image_url = f"/static/output/{code}_result.jpg"
            return jsonify({"status": "complete", "url": output_image_url}), 200
        elif processing_status[code] == "processing":
            return jsonify({"status": "processing"}), 202
        else:
            return jsonify({"status": "error", "message": "Failed to process video"}), 500
    else:
        return jsonify({"status": "error", "message": "Unsupported Media Type"}), 415

@app.route('/static/output/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/velocity', methods=['POST'])
def process_video_route():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files['video']
    distance = float(request.form.get('distance', 0))

    if not video_file or not distance:
        return jsonify({"error": "Video file and distance are required"}), 400

    # Save the video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    try:
        elapsed_time, velocity = process_velocity(video_path, distance)
        return jsonify({
            "elapsed_time": elapsed_time,
            "velocity": velocity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # finally:
        # if os.path.exists(video_path):
            # os.remove(video_path)  # Clean up the video file after processing


def process_video(video_path, code, y_difference_threshold=60):
    # Extract the filename without extension
    output_filename = f"{code}_result.jpg"
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Load the video and process
    cap = cv2.VideoCapture(video_path)
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
    kernel = np.ones((3, 3), np.uint8)
    start = "00:00:00"
    end = "00:03:00"

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    first_frame = cap.read()[1]

    found, _ = hog.detectMultiScale(first_frame)
    roi_bottom = max([y + h for (_, y, w, h) in found])
    roi_bottom = int(roi_bottom - 0.1 * roi_bottom)  # Extend the ROI by 10%

    start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start.split(":"))))
    end_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end.split(":"))))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    history_centroid = []
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_height, frame_width = frame.shape[:2]
        roi = frame[int(0.0 * frame_height):roi_bottom, 0:frame_width]

        fgmask = backgroundObject.apply(roi)
        _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=6)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_centroids = []

        for cnt in contours:
            x, y, width, height = cv2.boundingRect(cnt)
            if height > 0 and (width / height) > 3:
                if width > 20 or height > 20:
                    y += int(0.0 * frame_height)
                    centroid_x = int(x + width / 2)
                    centroid_y = int(y + height / 2)
                    current_centroids.append((centroid_x, centroid_y))
                    history_centroid.append({'x': centroid_x, 'y': centroid_y})

        last_frame = frame

    cap.release()

    # Fit and draw the parabolic curve
    history_centroid.sort(key=lambda x: x['y'])
    median_y = history_centroid[len(history_centroid) // 2]['y']
    history_centroid = [pt for pt in history_centroid if median_y - y_difference_threshold < pt['y'] < median_y + y_difference_threshold]
    history_centroid.sort(key=lambda pt: pt['x'])

    x_coords = [pt['x'] for pt in history_centroid]
    y_coords = [pt['y'] for pt in history_centroid]
    parabola_coeffs = np.polyfit(x_coords, y_coords, 2)
    parabola = np.poly1d(parabola_coeffs)

    curve_points = [(x, int(parabola(x))) for x in range(min(x_coords), max(x_coords))]

    for i in range(1, len(curve_points)):
        cv2.line(last_frame, curve_points[i - 1], curve_points[i], (0, 255, 0), 2)

    for pt in history_centroid:
        cv2.circle(last_frame, (pt['x'], pt['y']), 5, (255, 0, 0), -1)

    cv2.imwrite(output_image_path, last_frame)

    # Update status to complete
    processing_status[code] = "complete"

    return output_image_path

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)

