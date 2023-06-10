import keras_ocr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def ocr_handwritten_text_keras(image_path):
    pipeline = keras_ocr.pipeline.Pipeline()
    image = keras_ocr.tools.read(image_path)
    predictions = pipeline.recognize([image])[0]
    text = " ".join([text for _, text in predictions])
    return text

ocr_handwritten_text_keras("/Users/rahuljestadi/Desktop/1.jpeg")
