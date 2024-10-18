"""
## Summary

"""

# ------------------------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------------------------- #

# Standard library imports
import os

# Third-party imports
import torch
import cv2
from flask import Flask, request, render_template, redirect, url_for
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO

from model_utils import extract_keyframes, generate_summary

# ------------------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    det_model = YOLO('yolov8n.pt')  # Load an official Detect model

    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def upload_video():
        if request.method == 'POST':
            video = request.files['video']
            video_path = os.path.join('videos', video.filename)
            video.save(video_path)

            # Process video and generate summary
            extract_keyframes(det_model, video_path, 'frames/')
            summary = generate_summary(model, tokenizer, "A drone is capturing objects in a field.")

            return f"<h1>Video Summary:</h1><p>{summary}</p>"

        return '''
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="video">
            <button type="submit">Upload</button>
        </form>
        '''

    if __name__ == '__main__':
        app.run(debug=True)
