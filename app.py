import os
import pytesseract
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradeAnswer import grade_answer
import cv2
import pytesseract
from pytesseract import Output
from ocrModel import ocr_handwritten_text_keras

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "/Users/rahuljestadi/Downloads/Automatic_answer_Evaluator-main/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/model_executor', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        keywords = request.form.get("keywords").split(",")
        model_answer = request.form.get("model_answer")
        student_answer = request.form.get("student_answer")
        if "student_answer_image" in request.files:
            file = request.files["student_answer_image"]
            if file and allowed_file(file.filename):
                print(file)
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                print(filepath)
                file.save(filepath)
                student_answer = ocr_handwritten_text_keras(filepath)
                # Load image and convert to grayscale
                # img = cv2.imread(filepath)
                # print(img)
                # print("we are after img")

                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print("we are after gray")
                # # Threshold image to binarize
                # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # # Find contours and sort from top to bottom
                # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

                # # Set configuration for Tesseract OCR
                # custom_config = r'--oem 3 --psm 6'

                # # Loop over contours to extract words
                # for cnt in contours:
                #     print("we are inside contours")
                #     x, y, w, h = cv2.boundingRect(cnt)
                #     word = img[y:y + h, x:x + w]
                #     word_gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)

                #     # Perform OCR on the word using Tesseract
                #     student_answer += pytesseract.image_to_string(word_gray, config=custom_config)
                
                print("-----------------")
                print(student_answer)
                print("this is the student answer ------")
                score = grade_answer(keywords, model_answer, student_answer)

                return jsonify({"score": score})

    return render_template("NLPModel.html")

if __name__ == "__main__":
    app.run(debug=True)
