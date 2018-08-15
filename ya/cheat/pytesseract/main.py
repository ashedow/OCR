from tesseract import image_to_string
import io
from PIL import Image
import pytesseract
from wand.image import Image as wi

def imageReader():
    im = Image.open("sample1.jpg")

    text = pytesseract.image_to_string(im, lang = 'eng')

    print(text)


def pdfReader():
    pdf = wi(filename = "sample2.pdf", resolution = 300)
    pdfImage = pdf.convert('jpeg')

    imageBlobs = []

    for img in pdfImage.sequence:
        imgPage = wi(image = img)
        imageBlobs.append(imgPage.make_blob('jpeg'))

    recognized_text = []

    for imgBlob in imageBlobs:
        im = Image.open(io.BytesIO(imgBlob))
        text = pytesseract.image_to_string(im, lang = 'eng')
        recognized_text.append(text)

    print(recognized_text)
