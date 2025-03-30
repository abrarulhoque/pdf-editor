import os
import base64
import io
import json
import re
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create the Flask app
app = Flask(__name__)

# OpenAI API key
OPENAI_API_KEY = ""

# For PythonAnywhere compatibility
import sys
import inspect

# First try to monkey patch the http client to avoid proxies issues
try:
    import httpx
    from openai import OpenAI
    from openai._base_client import SyncHttpxClientWrapper

    # Create our own patched version of the client wrapper
    original_init = SyncHttpxClientWrapper.__init__

    def patched_init(self, **kwargs):
        # Remove 'proxies' if it exists - this is what's causing the issue on PythonAnywhere
        if "proxies" in kwargs:
            del kwargs["proxies"]
        # Call the original init with the cleaned kwargs
        original_init(self, **kwargs)

    # Apply our monkey patch
    SyncHttpxClientWrapper.__init__ = patched_init

    # Now create the client with our patch in place
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("Using patched OpenAI client to avoid proxies issue")
except Exception as e:
    print(f"Patching failed: {str(e)}")
    try:
        # Fallback to standard import - maybe we're on a different version
        from openai import OpenAI

        # Create a basic client with explicitly specified arguments (avoiding proxies)
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1", timeout=60.0
        )
        print("Using minimal OpenAI client configuration")
    except Exception as e:
        print(f"Standard import failed: {str(e)}")
        try:
            # Last resort - legacy OpenAI
            import openai

            openai.api_key = OPENAI_API_KEY
            openai_client = openai
            print("Using legacy OpenAI client")
        except Exception as e:
            print(f"Legacy setup failed: {str(e)}")
            # We'll fail at runtime if none of these worked


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    # Check if image file was uploaded
    if "image" not in request.files and "imageData" not in request.form:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Process the image from file upload or clipboard paste
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
            # Save to BytesIO for later use
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or "PNG")
            img_byte_arr.seek(0)
            base64_image = base64.b64encode(img_byte_arr.read()).decode("utf-8")
        else:
            # Process base64 image data
            image_data = request.form["imageData"]
            # Remove data URL prefix if present
            if "base64," in image_data:
                base64_image = image_data.split("base64,")[1]
            else:
                base64_image = image_data
            # Load image for later rendering (if needed)
            image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

        # Process directly with OpenAI
        result = process_with_openai_vision(base64_image)

        # Create image with table data
        if result.get("is_valid_size_chart", False) and "converted_table" in result:
            table_image = create_table_image(result["converted_table"])

            # Convert the image to base64 for sending to frontend
            buffered = io.BytesIO()
            table_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return jsonify(
                {
                    "success": True,
                    "image": f"data:image/png;base64,{img_str}",
                    "explanation": result.get("explanation", "Conversion successful"),
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "error": result.get("explanation", "Failed to process size chart"),
                }
            )

    except Exception as e:
        import traceback

        traceback.print_exc()

        # Provide more specific error messages based on exception type
        error_message = str(e)
        if "image" in str(e).lower():
            error_message = "Error processing image. Please ensure the image is valid and try again."
        elif "openai" in str(e).lower():
            error_message = "Error communicating with OpenAI API. Please check API key and connection."

        return jsonify({"success": False, "error": error_message}), 500


def process_with_openai_vision(base64_image):
    """Process an image directly with OpenAI Vision API for size chart extraction and conversion"""
    try:
        # System prompt for OpenAI
        system_prompt = """You are a specialized assistant that extracts size charts from images and converts measurements from inches to centimeters.

Your task is to:
1. Analyze the provided image of a size chart and extract the table data accurately
2. Convert all inch measurements to centimeters (1 inch = 2.54 cm)
3. Create a structured JSON with both the original inch values and the new cm values

IMPORTANT INFORMATION:
- The table is from a sewing pattern size chart
- The measurements are in inches, even if they use comma as decimal separator (European format)
- The size indicators are often age-based (e.g., 6m, 9m, 1Y, 2Y, etc.) or size-based (S, M, L, etc.)
- For each measurement, show both the original inch value and the converted cm value
- Round all centimeter values to 1 decimal place

RESPONSE FORMAT (JSON):
{
  "is_valid_size_chart": true,
  "explanation": "Brief explanation of the conversion",
  "converted_table": {
    "header_row": ["Size", "Length (in/cm)", "Chest (in/cm)", ...],
    "data_rows": [
      ["6m", "17.0 / 43.2", "22.0 / 55.9", ...],
      ["9m", "18.2 / 46.2", "22.8 / 57.9", ...],
      ...
    ]
  }
}

For the header row, add "(in/cm)" to measurement columns.
For data cells, use the format "original_value / converted_value".
Ensure complete and accurate extraction of ALL rows and columns from the size chart.
"""

        # Fix the image URL format - must include proper MIME type
        # and this format is specifically required by OpenAI API
        image_url = {"url": f"data:image/png;base64,{base64_image}"}

        # Create the message with image content for modern API
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the size chart data from this image and convert all inch measurements to centimeters. Format the data as JSON according to the instructions.",
                    },
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ]

        print("Sending image to OpenAI for processing...")

        # Check if we're using legacy or modern client
        if hasattr(openai_client, "chat"):
            # Modern OpenAI client (v1.0+)
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000,
                )

                # Extract response text from modern client
                response_text = response.choices[0].message.content

            except Exception as e:
                # If gpt-4o-mini fails, try gpt-4-vision-preview as fallback
                print(
                    f"Error with gpt-4o-mini: {str(e)}. Trying gpt-4-vision-preview..."
                )
                response = openai_client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000,
                )
                response_text = response.choices[0].message.content

        else:
            # Legacy OpenAI client (pre-1.0)
            try:
                # Format messages for legacy client
                legacy_messages = []
                for msg in messages:
                    if isinstance(msg.get("content", ""), list):
                        # Convert modern message format to legacy
                        text_parts = []
                        image_parts = []
                        for content_item in msg["content"]:
                            if content_item.get("type") == "text":
                                text_parts.append(content_item.get("text", ""))
                            elif content_item.get("type") == "image_url":
                                image_url = content_item.get("image_url", {}).get(
                                    "url", ""
                                )
                                if image_url:
                                    image_parts.append(
                                        {"url": image_url, "detail": "auto"}
                                    )

                        # Create legacy format message
                        legacy_content = {
                            "role": msg["role"],
                            "content": text_parts[0] if text_parts else "",
                        }

                        if image_parts:
                            legacy_content["content"] = (
                                text_parts[0] if text_parts else ""
                            )
                            legacy_content["images"] = image_parts

                        legacy_messages.append(legacy_content)
                    else:
                        legacy_messages.append(msg)

                # Call legacy OpenAI API
                response = openai_client.ChatCompletion.create(
                    model="gpt-4-vision-preview",  # Legacy model that supports images
                    messages=legacy_messages,
                    temperature=0.1,
                    max_tokens=2000,
                )

                # Extract response from legacy format
                response_text = response.choices[0].message.content

            except Exception as e:
                print(f"Legacy API call failed: {str(e)}")
                # Try one more approach with the most basic format
                try:
                    response = openai_client.ChatCompletion.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": "Extract the size chart from this image and convert to cm.",
                                "image_url": f"data:image/png;base64,{base64_image}",
                            },
                        ],
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    response_text = response.choices[0].message.content
                except Exception as final_e:
                    return {
                        "is_valid_size_chart": False,
                        "explanation": f"All OpenAI API call attempts failed: {str(final_e)}",
                    }

        print("Response received from OpenAI")

        # Rest of the function for parsing response remains unchanged
        try:
            # Extract JSON from response
            json_match = None
            if "```json" in response_text:
                json_match = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_match = response_text.split("```")[1].strip()
            else:
                # Fallback to finding the first '{' and last '}'
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_match = response_text[json_start:json_end]

            if not json_match:
                return {
                    "is_valid_size_chart": False,
                    "explanation": f"Failed to extract JSON from response: {response_text[:100]}...",
                }

            # Parse JSON
            print("Parsing JSON response")
            result = json.loads(json_match)

            # Validate expected structure
            if "is_valid_size_chart" not in result:
                return {
                    "is_valid_size_chart": False,
                    "explanation": "Missing required field 'is_valid_size_chart' in response",
                }

            if (
                result.get("is_valid_size_chart", False)
                and "converted_table" not in result
            ):
                return {
                    "is_valid_size_chart": False,
                    "explanation": "Missing 'converted_table' in valid size chart response",
                }

            return result

        except Exception as json_err:
            print(f"Error parsing OpenAI response: {str(json_err)}")
            return {
                "is_valid_size_chart": False,
                "explanation": f"Error parsing OpenAI response: {str(json_err)}",
            }

    except Exception as e:
        print(f"Error processing with OpenAI: {str(e)}")
        return {
            "is_valid_size_chart": False,
            "explanation": f"Error processing with OpenAI: {str(e)}",
        }


def create_table_image(table_data):
    """Create a high-resolution image of the processed table"""
    # Extract header and data rows
    header_row = table_data.get("header_row", [])
    data_rows = table_data.get("data_rows", [])

    if not header_row or not data_rows:
        # Create a simple error image
        img = Image.new("RGB", (800, 200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        d.text(
            (50, 50), "Error: Invalid table data received", fill=(255, 0, 0), font=font
        )
        return img

    # Determine table dimensions
    num_rows = len(data_rows) + 1  # +1 for header
    num_cols = len(header_row)

    # Define styling parameters
    cell_padding = 15  # Increased padding for more space
    header_color = (250, 197, 164)  # Light peach/orange
    header_text_color = (0, 0, 0)
    data_text_color = (0, 0, 0)
    line_color = (0, 0, 0)
    highlight_color = (250, 197, 164)  # Light peach/orange for first column

    # Load fonts with increased sizes
    try:
        # Try to load bold font for headers and regular for data
        header_font = ImageFont.truetype("arialbd.ttf", 19)  # Arial Bold at larger size
    except IOError:
        try:
            # Fallback to regular Arial but larger
            header_font = ImageFont.truetype("arial.ttf", 19)
        except IOError:
            # Ultimate fallback
            header_font = ImageFont.load_default()

    try:
        # Increased size for data cells
        data_font = ImageFont.truetype("arial.ttf", 17)
    except IOError:
        # Fallback
        data_font = ImageFont.load_default()

    # Calculate cell sizes
    # Determine maximum width for each column
    col_widths = [0] * num_cols

    # Check header row
    for col, header in enumerate(header_row):
        try:
            text_width = header_font.getbbox(str(header))[2]
        except:
            try:
                # Fallback for older PIL versions
                text_width = header_font.getsize(str(header))[0]
            except:
                # Ultimate fallback
                text_width = len(str(header)) * 10  # Rough estimate
        col_widths[col] = max(col_widths[col], text_width + cell_padding * 2)

    # Check data rows
    for row in data_rows:
        for col, cell in enumerate(row[:num_cols]):  # Ensure we don't exceed columns
            if col < len(col_widths):  # Safety check
                try:
                    text_width = data_font.getbbox(str(cell))[2]
                except:
                    try:
                        # Fallback for older PIL versions
                        text_width = data_font.getsize(str(cell))[0]
                    except:
                        # Ultimate fallback
                        text_width = len(str(cell)) * 8  # Rough estimate
                col_widths[col] = max(col_widths[col], text_width + cell_padding * 2)

    # Set row height (increased for better readability)
    row_height = 45

    # Calculate total image dimensions
    table_width = sum(col_widths)
    table_height = row_height * num_rows

    # Create image with minimal margin - just enough for the border
    margin = 2  # Minimal margin
    img_width = table_width + margin * 2
    img_height = table_height + margin * 2

    # Create new RGB image with white background
    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw the table
    y_offset = margin

    # Draw header row
    x_offset = margin
    for col, header in enumerate(header_row):
        # Draw cell background
        cell_width = col_widths[col]
        draw.rectangle(
            [(x_offset, y_offset), (x_offset + cell_width, y_offset + row_height)],
            fill=header_color,
        )

        # Draw cell text
        text = str(header)
        # Center text
        try:
            text_bbox = header_font.getbbox(text)
            text_width = text_bbox[2]
            text_height = text_bbox[3]
        except:
            try:
                # Fallback for older PIL versions
                text_width, text_height = header_font.getsize(text)
            except:
                # Ultimate fallback
                text_width, text_height = len(text) * 10, 20

        text_x = x_offset + (cell_width - text_width) // 2
        text_y = y_offset + (row_height - text_height) // 2

        # Draw text in black
        draw.text((text_x, text_y), text, fill=header_text_color, font=header_font)

        # For extra boldness on some systems where bold font isn't available,
        # draw the text a second time with a slight offset (creates a faux-bold effect)
        draw.text((text_x + 1, text_y), text, fill=header_text_color, font=header_font)

        x_offset += cell_width

    # Draw data rows
    for row_idx, row in enumerate(data_rows):
        y_offset += row_height
        x_offset = margin

        for col, cell in enumerate(row[:num_cols]):  # Ensure we don't exceed columns
            if col < len(col_widths):  # Safety check
                cell_width = col_widths[col]

                # Highlight first column
                if col == 0:
                    draw.rectangle(
                        [
                            (x_offset, y_offset),
                            (x_offset + cell_width, y_offset + row_height),
                        ],
                        fill=highlight_color,
                    )

                # Draw cell text
                text = str(cell)
                # Center text
                try:
                    text_bbox = data_font.getbbox(text)
                    text_width = text_bbox[2]
                    text_height = text_bbox[3]
                except:
                    try:
                        # Fallback for older PIL versions
                        text_width, text_height = data_font.getsize(text)
                    except:
                        # Ultimate fallback
                        text_width, text_height = len(text) * 8, 18

                text_x = x_offset + (cell_width - text_width) // 2
                text_y = y_offset + (row_height - text_height) // 2
                draw.text((text_x, text_y), text, fill=data_text_color, font=data_font)

                x_offset += cell_width

    # Draw grid lines with slightly thicker lines
    y_offset = margin
    # Draw horizontal lines
    for row in range(num_rows + 1):
        draw.line(
            [(margin, y_offset), (margin + table_width, y_offset)],
            fill=line_color,
            width=2,
        )
        y_offset += row_height

    # Draw vertical lines
    x_offset = margin
    for col in range(num_cols + 1):
        draw.line(
            [(x_offset, margin), (x_offset, margin + table_height)],
            fill=line_color,
            width=2,
        )
        if col < len(col_widths):
            x_offset += col_widths[col]

    # Return the image
    return img


if __name__ == "__main__":
    # Create upload directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    # Run the app in debug mode
    app.run(debug=True)
