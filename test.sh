#!/bin/bash

# Set the Gemini API key
GEMINI_API_KEY="AIzaSyBsP3zw3aqlO85YyeDhFpP4w6DLhck1OHk"

# Define the MODEL variable
# Uncomment the model you want to use
# MODEL="gemini-1.5-flash"
# MODEL="gemini-1.5-flash-8b-exp-0827"
# MODEL="gemini-1.5-flash-exp-0827"
# MODEL="gemini-1.5-pro"
# MODEL="gemini-1.5-pro-exp-0801"
# MODEL="gemini-1.5-pro-exp-0827"
MODEL="gemini-2.0-flash"

# Set the input folder (default to current directory if not provided)
INPUT_FOLDER=${1:-.}

# Define classes with multi-word examples
CLASSES=(
    "door"
)

# Convert classes array to a space-separated string
CLASS_STRING=$(IFS=' ' ; echo "${CLASSES[*]}")

# Run the Python script
python3 gemini_bbox_detection.py \
    --google_api_key "$GEMINI_API_KEY" \
    --classes $CLASS_STRING \
    --model "$MODEL" \
    --input_folder "$INPUT_FOLDER"

echo "Detection completed using model: $MODEL"
echo "Check the 'output' directory for annotated images and JSON results."