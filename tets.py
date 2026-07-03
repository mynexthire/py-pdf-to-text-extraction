import base64
from src.pdfExtractor import extract

def test_custom_path():
    input_path = "/home/nikhil/Downloads/36698.pdf"
    result = extract(input_path)
    with open(input_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
    with open("test_input.txt", "w") as f:
        f.write(str(base64_data))



test_custom_path()