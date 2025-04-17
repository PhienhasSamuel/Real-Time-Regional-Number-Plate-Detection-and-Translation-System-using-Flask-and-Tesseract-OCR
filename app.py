from flask import Flask, render_template, Response, request
import cv2
import pytesseract
from datetime import datetime
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Tamil to English literal mapping
tamil_to_eng = {
    'த': 'T', 'ந': 'N', 'ா': '',  # தநா = TN
    'அ': 'A', 'ஆ': 'AA', 'இ': 'I', 'ஈ': 'II',
    'உ': 'U', 'ஊ': 'UU', 'எ': 'E', 'ஏ': 'EE',
    'ஐ': 'AI', 'ஒ': 'O', 'ஓ': 'OO', 'ஔ': 'AU',
    'ஜ': 'J', 'ஷ': 'SH', 'ஸ': 'S', 'ஹ': 'H',
    'க': 'K', 'ங': 'NG', 'ச': 'C', 'ஞ': 'NJ',
    'ட': 'D', 'ண': 'N', 'த': 'T', 'ந': 'N',
    'ப': 'B', 'ம': 'M', 'ய': 'Y', 'ர': 'R',
    'ல': 'L', 'வ': 'V', 'ழ': 'ZH', 'ள': 'L',
    'ற': 'R', 'ன': 'N', 'ம்': 'M', 'க்': 'K',
    'ட்': 'T', 'ச்': 'C', 'ப்': 'B', 'ண்': 'N'
}

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def translate_text(text):
    words = text.split()
    translated = []
    for word in words:
        t_word = ''.join(tamil_to_eng.get(char, char) for char in word)
        translated.append(t_word)
    return ' '.join(translated)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if not success:
        return "Failed to capture image"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = f"static/captures/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, frame)

    processed = preprocess_image(frame)

    # OCR using Tamil + English
    text = pytesseract.image_to_string(processed, lang='tam+eng')
    print("[OCR RAW]:", text)

    translated = translate_text(text)
    print("[Translated]:", translated)

    return render_template('result.html',
                           image_path=filename,
                           timestamp=timestamp,
                           detected_text=text.strip(),
                           translated_text=translated.strip())

if __name__ == '__main__':
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')
    app.run(debug=True)
