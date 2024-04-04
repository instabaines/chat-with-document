from fpdf import FPDF
from pypdf import PdfReader
def read_uploaded_file(uploaded_file):
    # Read the content of the uploaded file
    if uploaded_file.type == "text/plain":
        return [uploaded_file.getvalue().decode("utf-8")]
    elif uploaded_file.type == "application/pdf":
       text = []
       reader = PdfReader(uploaded_file)
       for page in reader.pages:
           text.append(page.extract_text())
       return text
    
