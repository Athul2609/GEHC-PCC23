import os
from fpdf import FPDF

# Define the list of file patterns to match (e.g., ending with "explanation.txt" or "features.txt")
file_patterns = ["explanation.txt"]

# Function to create a PDF from a text file
def create_pdf(input_file, output_folder):
    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Read the content of the text file
    with open(input_file, "r") as f:
        content = f.read()

    # Add the content to the PDF
    pdf.multi_cell(0, 10, content)

    # Define the output PDF file name
    output_pdf = os.path.splitext(os.path.basename(input_file))[0] + ".pdf"

    # Define the full path to the output PDF file
    output_path = os.path.join(output_folder, output_pdf)

    # Output the PDF to the file
    pdf.output(output_path)

    print(f"PDF created: {output_path}")

# Define the folder where PDFs will be created
output_folder = os.path.join(os.getcwd(), "Hacakathon-Chatbot-main", "data")

# Get a list of PDF files in the output folder that end with "explanation"
pdf_files_to_delete = [f for f in os.listdir(output_folder) if f.endswith("explanation.pdf")]

# Delete the existing PDF files
for pdf_file in pdf_files_to_delete:
    pdf_path = os.path.join(output_folder, pdf_file)
    os.remove(pdf_path)
    print("pdfdeleted")

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Iterate through each file pattern and create a PDF for each matching file
for pattern in file_patterns:
    for filename in os.listdir(current_directory):
        if filename.endswith(pattern):
            file_path = os.path.join(current_directory, filename)
            create_pdf(file_path, output_folder)
