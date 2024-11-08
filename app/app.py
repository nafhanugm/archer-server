from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is uploaded
    if 'video' not in request.files:
        return 'No file uploaded', 400
    file = request.files['video']


    
    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the video and generate the output image
    output_image_path = process_video(filepath)

    return send_file(output_image_path, mimetype='image/jpeg')

def process_video(video_path):
    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{filename}_result.jpg"
    output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Load the video and process similar to your code above
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

    history_centroid.sort(key=lambda x: x['y'])
    y_difference_threshold = 30
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

    return output_image_path

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
