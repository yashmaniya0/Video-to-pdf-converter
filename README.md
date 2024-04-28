# Video to PDF Converter

## Overview

This project focuses on extracting pages from a video where a book's pages are flipped one by one. The captured frames with visible pages are then used to create a PDF containing all the book pages.

## Project Workflow

1. **Video Input**: Provide a video file (`*.mp4`) where a book's pages are being flipped. The video can have some hand motion and camera movements.

2. **Page Capture**:
   - Detect significant changes between consecutive frames to identify page flips.
   - Capture frames when a page flip is detected and when the page is stable for better visibility.

3. **Page Extraction**:
   - Use computer vision techniques to extract the page from the captured frames.
   - Remove background noise and distortion to isolate the page.

4. **PDF Generation**:
   - Save the extracted pages as images.
   - Create a PDF containing all the extracted pages.

## Usage

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/book-page-extraction.git

2. Install the required libraries:

    pip install opencv-python-headless reportlab numpy Pillow

3. Place your video file (*.mp4) in the project directory.

4. Modify the video_path variable in main.py to point to your video file.

5. Run the script:
    pyton video_to_pdf.py

6. Once the script finishes, you will find the extracted pages in the project directory as JPEG images (saved_frame_*.jpg) and a  PDF file (content_pdf_1.pdf) containing all the pages

## Dependencies

- OpenCV: For computer vision tasks such as frame extraction and page detection.
- ReportLab: For generating PDF files.
- NumPy: For numerical operations.
- Pillow: For image processing.

## Notes

- Adjust the threshold values and parameters in the code according to the specific characteristics of your video.
- Ensure proper lighting and stable camera setup for better results.
- For detailed explanations and optimizations of the code, refer to the `video_to_pdf.py` file.

