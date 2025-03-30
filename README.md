# PDF Size Chart Converter

A web-based application that allows users to upload PDFs containing size charts, select specific regions, and automatically convert inch measurements to centimeters directly in the PDF document.

## Features

- Upload multiple PDF files
- Interactive PDF viewing
- Area selection for size charts (similar to screenshot tools)
- Automatic extraction of size chart data using OpenAI Vision
- Conversion of inch measurements to centimeters
- Background processing with queue system
- Generation of new PDFs with the converted tables
- Table styling with header/row highlighting

## Requirements

- Python 3.7+
- Flask
- PyMuPDF (fitz)
- OpenAI API
- Pillow (PIL)
- numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abrarulhoque/pdf-editor.git
   cd pdf-editor
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements_pdf_editor.txt
   ```

3. Set up your OpenAI API key in `app.py`.

4. Run the application:

   ```bash
   python run_pdf_editor.py
   ```

5. Open a web browser and navigate to http://localhost:5001

## Usage

1. Upload one or more PDF documents containing size charts
2. Navigate to the page containing the size chart (defaults to page 4 if available)
3. Click "Select Size Chart Area" and draw a rectangle around the size chart
4. Click "Process Selection" to extract and convert the measurements
5. The file moves to the processing queue and the next file is displayed (if you uploaded multiple)
6. When processing is complete, you can download the modified PDF from the "Completed Files" section

## Technical Implementation

The application uses:

- **PyMuPDF** for PDF manipulation
- **Flask** for the web application backend
- **OpenAI Vision API** for size chart recognition and data extraction
- **HTML5 Canvas** for interactive PDF viewing and selection
- **Bootstrap** for responsive UI
- **Background processing** with Python threading for handling multiple files

## Queue System

The application implements two queue systems:

1. **Selection Queue**: Files waiting for the user to select size chart areas
2. **Processing Queue**: Files being processed by the OpenAI Vision API

This allows for efficient batch processing of multiple files without waiting for each conversion to complete.

## License

This project is available under the MIT License.
