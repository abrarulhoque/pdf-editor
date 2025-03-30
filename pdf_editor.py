import os
import io
import base64
import json
import uuid
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify, send_file, url_for, session
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import numpy as np
import sys
import traceback
import threading
import time
from collections import deque
import copy

# Attempt to import from app.py, but provide fallback if unavailable
try:
    from app import process_with_openai_vision, create_table_image

    print("Successfully imported OpenAI functions from app.py")
except ImportError:
    # Provide basic implementations if imports fail
    def process_with_openai_vision(base64_image):
        """Fallback implementation if app.py import fails"""
        print("WARNING: Using fallback implementation of process_with_openai_vision")
        return {
            "is_valid_size_chart": False,
            "explanation": "Could not import OpenAI processing function from app.py",
        }

    def create_table_image(table_data):
        """Fallback implementation if app.py import fails"""
        print("WARNING: Using fallback implementation of create_table_image")
        # Create a simple error image
        img = Image.new("RGB", (800, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text(
            (50, 50),
            "Error: Could not import table creation function from app.py",
            fill=(255, 0, 0),
            font=font,
        )
        return img


# Create the Flask app
app = Flask(__name__, static_folder="static")
# Set a secret key for session management
app.secret_key = os.urandom(24)

# Ensure directories exist
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Queue and processing status tracking
# Global variables to track queues and processing status
selection_queue = {}  # Maps session_id to a list of files waiting for area selection
processing_queue = (
    {}
)  # Maps session_id to a list of files waiting for OpenAI processing
completed_files = {}  # Maps session_id to a list of completed files
processing_status = {}  # Maps file_id to status (queued, processing, completed, failed)
processing_results = {}  # Maps file_id to processing results
processing_lock = threading.Lock()  # Lock for thread-safe queue operations


# Processing worker thread
def processing_worker():
    """Worker thread to process files in the processing queue"""
    while True:
        # Check each session's processing queue
        with processing_lock:
            for session_id, queue in list(processing_queue.items()):
                if queue:  # If there's a file to process
                    # Get the next file from this session's queue
                    file_info = queue[0]

                    # Only process if it's not already processing
                    if processing_status.get(file_info["file_id"]) == "queued":
                        # Set status to processing
                        processing_status[file_info["file_id"]] = "processing"
                        # Remove from queue since we're processing it now
                        processing_queue[session_id].pop(0)

                        # Process file in a separate thread to not block the worker
                        threading.Thread(
                            target=process_file_background, args=(session_id, file_info)
                        ).start()

        # Sleep to avoid excessive CPU usage
        time.sleep(1)


# Start the worker thread
worker_thread = threading.Thread(target=processing_worker, daemon=True)
worker_thread.start()


def process_file_background(session_id, file_info):
    """Process a file in the background"""
    try:
        # Extract file information
        file_id = file_info["file_id"]
        internal_filename = file_info["internal_filename"]
        original_filename = file_info["original_filename"]
        page_num = file_info["page_num"]
        selection = file_info["selection"]

        filepath = os.path.join(UPLOAD_FOLDER, internal_filename)
        if not os.path.exists(filepath):
            with processing_lock:
                processing_status[file_id] = "failed"
                processing_results[file_id] = {"error": "PDF file not found"}
            return

        # Create output filename
        unique_id = str(uuid.uuid4())[:8]
        name, ext = os.path.splitext(original_filename)
        output_filename = f"{name}_{unique_id}{ext}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        # Open the PDF
        doc = fitz.open(filepath)
        if page_num >= doc.page_count:
            with processing_lock:
                processing_status[file_id] = "failed"
                processing_results[file_id] = {
                    "error": f"Page {page_num} does not exist"
                }
            doc.close()
            return

        # Get the page
        page = doc[page_num]

        # Convert selection coordinates to PDF coordinates
        zoom = 1.5  # Same zoom used in get_pdf_page
        x = selection["x"] / zoom
        y = selection["y"] / zoom
        width = selection["width"] / zoom
        height = selection["height"] / zoom

        # Create rect from selection
        rect = fitz.Rect(x, y, x + width, y + height)

        # Extract selected area as image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        img_bytes = pix.tobytes("png")

        # Convert to base64 for OpenAI processing
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Process with OpenAI
        result = process_with_openai_vision(img_base64)

        # Check if valid size chart
        if result.get("is_valid_size_chart", False) and "converted_table" in result:
            # Create a completely new document instead of modifying the original
            new_doc = fitz.open()

            # Copy all pages from original document
            new_doc.insert_pdf(doc)

            # Close original document
            doc.close()

            # Now work with the new document
            # Get the page we need to modify
            page = new_doc[page_num]

            # Clear the area with white rectangle
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)

            # Now draw the table with the converted data
            table_data = result["converted_table"]
            header_row = table_data.get("header_row", [])
            data_rows = table_data.get("data_rows", [])

            if header_row and data_rows:
                draw_table_on_pdf(new_doc, page_num, rect, header_row, data_rows)

            # Save the new document
            new_doc.save(output_path)
            new_doc.close()

            # Update status and result
            with processing_lock:
                processing_status[file_id] = "completed"
                processing_results[file_id] = {
                    "success": True,
                    "message": "Table processed successfully",
                    "table_data": table_data,
                    "output_filename": output_filename,
                    "original_filename": original_filename,
                }

                # Add to completed files list
                if session_id not in completed_files:
                    completed_files[session_id] = []
                completed_files[session_id].append(
                    {
                        "file_id": file_id,
                        "internal_filename": internal_filename,
                        "original_filename": original_filename,
                        "output_filename": output_filename,
                    }
                )
        else:
            doc.close()
            # Update status and result with failure
            with processing_lock:
                processing_status[file_id] = "failed"
                processing_results[file_id] = {
                    "success": False,
                    "error": result.get("explanation", "Failed to process size chart"),
                }

    except Exception as e:
        traceback.print_exc()
        # Update status with error
        with processing_lock:
            processing_status[file_id] = "failed"
            processing_results[file_id] = {
                "success": False,
                "error": f"Error: {str(e)}",
            }


# Routes
@app.route("/")
def index():
    # Initialize session queues if they don't exist
    session_id = (
        str(uuid.uuid4()) if "session_id" not in session else session["session_id"]
    )
    session["session_id"] = session_id

    # Initialize queues for this session
    if session_id not in selection_queue:
        selection_queue[session_id] = []
    if session_id not in processing_queue:
        processing_queue[session_id] = []
    if session_id not in completed_files:
        completed_files[session_id] = []

    return render_template("pdf_editor.html")


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    """Handle PDF file upload(s)"""
    if "pdfFile" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    # Get the session ID
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        selection_queue[session_id] = []
        processing_queue[session_id] = []
        completed_files[session_id] = []

    # Handle multiple files
    files = request.files.getlist("pdfFile")
    uploaded_files = []

    for file in files:
        if file.filename == "":
            continue  # Skip files with no name

        # Store original filename
        original_filename = file.filename

        # Generate a unique internal filename for processing
        file_id = str(uuid.uuid4())
        internal_filename = f"{file_id}.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, internal_filename)
        file.save(filepath)

        # Extract first page as preview
        try:
            doc = fitz.open(filepath)
            if doc.page_count > 0:
                # Render page as image (use page 4 if available, otherwise first page)
                preview_page = (
                    3 if doc.page_count > 3 else 0
                )  # 0-indexed, so page 4 is index 3
                page = doc[preview_page]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                preview_path = os.path.join(
                    UPLOAD_FOLDER,
                    f"{os.path.splitext(internal_filename)[0]}_preview.png",
                )
                pix.save(preview_path)

                # Convert preview to base64
                with open(preview_path, "rb") as img_file:
                    preview_base64 = base64.b64encode(img_file.read()).decode("utf-8")

                # Clean up preview file
                os.remove(preview_path)

                # Create file info
                file_info = {
                    "file_id": file_id,
                    "internal_filename": internal_filename,
                    "original_filename": original_filename,
                    "pageCount": doc.page_count,
                    "default_page": preview_page,
                    "preview": f"data:image/png;base64,{preview_base64}",
                }

                # Add to selection queue
                selection_queue[session_id].append(file_info)
                uploaded_files.append(file_info)
            else:
                # Skip files with no pages
                continue
        except Exception as e:
            # Skip files that can't be processed
            print(f"Error processing {original_filename}: {str(e)}")
            continue

    # Return the first file in queue if any were uploaded
    if uploaded_files:
        response = {
            "success": True,
            "files_queued": len(uploaded_files),
            "queue_position": 1,
            "total_in_queue": len(selection_queue[session_id]),
            "next_file": (
                selection_queue[session_id][0] if selection_queue[session_id] else None
            ),
        }
        return jsonify(response)
    else:
        return jsonify({"error": "No valid PDF files were uploaded"}), 400


@app.route("/get_pdf_page", methods=["GET"])
def get_pdf_page():
    """Get a specific page from the PDF as an image"""
    filename = request.args.get("filename")
    page_num = int(request.args.get("page", 0))

    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "PDF file not found"}), 404

    try:
        doc = fitz.open(filepath)
        if page_num >= doc.page_count:
            return (
                jsonify({"error": f"Page {page_num} does not exist in this PDF"}),
                400,
            )

        # Render page as image
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")

        # Convert to base64 for return
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return jsonify(
            {
                "success": True,
                "page": page_num,
                "total_pages": doc.page_count,
                "image": f"data:image/png;base64,{img_base64}",
            }
        )
    except Exception as e:
        return jsonify({"error": f"Error rendering PDF page: {str(e)}"}), 500


@app.route("/process_selection", methods=["POST"])
def process_selection():
    """Process a selected area in the PDF and advance to next file"""
    try:
        session_id = session.get("session_id")
        if not session_id:
            return jsonify({"error": "Invalid session"}), 400

        data = request.json
        file_id = data.get("file_id")
        internal_filename = data.get("filename")
        original_filename = data.get("original_filename")
        page_num = int(data.get("page", 0))
        selection = data.get("selection")

        if not internal_filename or not selection:
            return jsonify({"error": "Missing required parameters"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, internal_filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "PDF file not found"}), 404

        # Add to processing queue
        file_info = {
            "file_id": file_id,
            "internal_filename": internal_filename,
            "original_filename": original_filename,
            "page_num": page_num,
            "selection": selection,
        }

        with processing_lock:
            # Add to processing queue
            if session_id not in processing_queue:
                processing_queue[session_id] = []
            processing_queue[session_id].append(file_info)

            # Set initial status
            processing_status[file_id] = "queued"

            # Remove from selection queue
            if session_id in selection_queue and selection_queue[session_id]:
                selection_queue[session_id] = [
                    f for f in selection_queue[session_id] if f["file_id"] != file_id
                ]

            # Get next file in queue
            next_file = (
                selection_queue[session_id][0] if selection_queue[session_id] else None
            )

        # Return status and next file information
        return jsonify(
            {
                "success": True,
                "message": "Selection added to processing queue",
                "file_id": file_id,
                "status": "queued",
                "queue_position": len(processing_queue[session_id]),
                "next_file": next_file,
                "files_remaining": len(selection_queue[session_id]),
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Error: {str(e)}"}), 500


@app.route("/get_queue_status", methods=["GET"])
def get_queue_status():
    """Get status of all queues for the current session"""
    session_id = session.get("session_id")
    if not session_id:
        return jsonify({"error": "Invalid session"}), 400

    with processing_lock:
        # Get selection queue
        selection_files = selection_queue.get(session_id, [])

        # Get processing queue with statuses
        processing_files = []
        for file_info in processing_queue.get(session_id, []):
            file_id = file_info["file_id"]
            status_info = copy.deepcopy(file_info)
            status_info["status"] = processing_status.get(file_id, "unknown")
            processing_files.append(status_info)

        # Add currently processing files that aren't in the queue anymore
        for file_id, status in processing_status.items():
            if status == "processing":
                # Check if this file belongs to this session and is not already in processing_files
                if not any(p["file_id"] == file_id for p in processing_files):
                    # Find the file info from results
                    if file_id in processing_results:
                        result = processing_results[file_id]
                        if "original_filename" in result:
                            processing_files.append(
                                {
                                    "file_id": file_id,
                                    "original_filename": result["original_filename"],
                                    "status": "processing",
                                }
                            )

        # Get completed files
        completed = completed_files.get(session_id, [])

        # Gather result information for completed and failed files
        results = {}
        for file_id, status in processing_status.items():
            if status in ["completed", "failed"] and file_id in processing_results:
                results[file_id] = processing_results[file_id]

    # Return comprehensive queue status
    return jsonify(
        {
            "success": True,
            "selection_queue": selection_files,
            "processing_queue": processing_files,
            "completed_files": completed,
            "results": results,
        }
    )


@app.route("/download_pdf/<filename>", methods=["GET"])
def download_pdf(filename):
    """Download a processed PDF file"""
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(filepath):
        # Determine proper filename for the downloaded file
        download_name = filename

        # Check if this is a UUID-based filename and we have a better name
        if (
            len(filename.split("_")) > 1
            and len(filename.split("_")[-1].split(".")[0]) == 8
        ):
            # This looks like our naming scheme with UUID, use a cleaner name
            download_name = (
                "_".join(filename.split("_")[:-1]) + os.path.splitext(filename)[1]
            )

        return send_file(
            filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype="application/pdf",
        )
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/clear_queues", methods=["POST"])
def clear_queues():
    """Clear all queues for the current session"""
    session_id = session.get("session_id")
    if not session_id:
        return jsonify({"error": "Invalid session"}), 400

    with processing_lock:
        # Clear selection queue
        if session_id in selection_queue:
            selection_queue[session_id] = []

        # Clear processing queue
        if session_id in processing_queue:
            processing_queue[session_id] = []

        # Don't clear completed files as user might want to download them

    return jsonify({"success": True, "message": "Queues cleared"})


# Original helper functions
def draw_table_on_pdf(doc, page_num, rect, header_row, data_rows):
    """Draw a table on the PDF at the specified location"""
    page = doc[page_num]

    # Table dimensions
    num_rows = len(data_rows) + 1  # Add 1 for header
    num_cols = len(header_row)

    # Calculate cell dimensions
    table_width = rect.width
    table_height = rect.height
    row_height = table_height / num_rows
    col_width = table_width / num_cols

    # Define styling
    line_color = (0, 0, 0)  # Black
    line_width = 0.7
    header_bg_color = (0.98, 0.77, 0.64)  # Light peach/orange
    first_col_bg_color = (0.98, 0.77, 0.64)  # Same as header

    # Draw background for header row
    y_pos = rect.y0
    for col_idx in range(num_cols):
        x_pos = rect.x0 + (col_idx * col_width)
        cell_rect = fitz.Rect(x_pos, y_pos, x_pos + col_width, y_pos + row_height)
        page.draw_rect(cell_rect, color=None, fill=header_bg_color, width=0)

    # Draw background for first column
    for row_idx in range(1, num_rows):
        y_pos = rect.y0 + (row_idx * row_height)
        cell_rect = fitz.Rect(rect.x0, y_pos, rect.x0 + col_width, y_pos + row_height)
        page.draw_rect(cell_rect, color=None, fill=first_col_bg_color, width=0)

    # Draw horizontal lines
    for row_idx in range(num_rows + 1):
        y_pos = rect.y0 + (row_idx * row_height)
        page.draw_line(
            fitz.Point(rect.x0, y_pos),
            fitz.Point(rect.x1, y_pos),
            color=line_color,
            width=line_width,
        )

    # Draw vertical lines
    for col_idx in range(num_cols + 1):
        x_pos = rect.x0 + (col_idx * col_width)
        page.draw_line(
            fitz.Point(x_pos, rect.y0),
            fitz.Point(x_pos, rect.y1),
            color=line_color,
            width=line_width,
        )

    # Draw header text - Handle text that might not fit
    header_fontsize = 10  # Starting fontsize for header
    y_pos = rect.y0
    for col_idx, cell_text in enumerate(header_row):
        x_pos = rect.x0 + (col_idx * col_width)
        text_rect = fitz.Rect(
            x_pos + 2, y_pos + 2, x_pos + col_width - 2, y_pos + row_height - 2
        )

        # Try with initial fontsize, reduce if needed
        for fontsize in [
            header_fontsize,
            9,
            8,
            7,
            6,
        ]:  # Try progressively smaller sizes
            try:
                # Attempt to insert the text with current fontsize
                rc = page.insert_textbox(
                    text_rect,
                    str(cell_text),
                    fontname="Helvetica-Bold",
                    fontsize=fontsize,
                    align=fitz.TEXT_ALIGN_CENTER,
                )
                # If successful (rc >= 0), break the loop
                if rc >= 0:
                    break
            except:
                # If an exception occurs, continue with smaller font size
                continue

    # Draw data cell text - with increased font size and bold
    data_fontsize = 9  # Increased from original size
    for row_idx, row in enumerate(data_rows):
        y_pos = rect.y0 + ((row_idx + 1) * row_height)
        for col_idx, cell_text in enumerate(
            row[:num_cols]
        ):  # Ensure we don't exceed columns
            x_pos = rect.x0 + (col_idx * col_width)
            text_rect = fitz.Rect(
                x_pos + 2, y_pos + 2, x_pos + col_width - 2, y_pos + row_height - 2
            )

            # Try with initial fontsize, reduce if needed
            for fontsize in [data_fontsize, 8, 7, 6]:  # Try progressively smaller sizes
                try:
                    # Use bold font for all data cells
                    rc = page.insert_textbox(
                        text_rect,
                        str(cell_text) if cell_text is not None else "",
                        fontname="Helvetica-Bold",  # Changed to bold
                        fontsize=fontsize,
                        align=fitz.TEXT_ALIGN_CENTER,
                    )
                    # If successful (rc >= 0), break the loop
                    if rc >= 0:
                        break
                except:
                    # If an exception occurs, continue with smaller font size
                    continue


if __name__ == "__main__":
    app.run(debug=True, port=5001)
