import argparse
import json
from typing import List, Optional
import os

import google.generativeai as genai
import numpy as np
import PIL.Image
import supervision as sv

PROMPT_TEMPLATE = 'Return bounding boxes for `{}` as JSON arrays [ymin, xmin, ymax, xmax]. For example ```json\n[\n  {{\n    "person": [\n      255,\n      69,\n      735,\n      738\n    ]\n  }}\n]\n```'

def parse_response(response, size, classes: Optional[List[str]] = None):
    w, h = size
    try:
        data = json.loads(response.text.replace('json', '').replace('```', '').replace('\n', ''))
        class_name = []
        yxyx = []
        for item in data:
            for key, value in item.items():
                class_name.append(key)
                yxyx.append(value)
        yxyx = np.array(yxyx, dtype=np.float64)
        xyxy = yxyx[:, [1, 0, 3, 2]]
        xyxy /= 1000
        xyxy *= np.array([w, h, w, h])

        detections = sv.Detections(
            xyxy=xyxy,
            class_id=None if classes is None else np.array([classes.index(i) for i in class_name]),
        )
        detections.class_name = class_name
        return detections
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("Raw response:")
        print(response.text)
        return sv.Detections(xyxy=np.array([]), class_id=np.array([]))

def run_detection(model, image_path, classes):
    PROMPT = PROMPT_TEMPLATE.format(", ".join(classes))

    try:
        image = PIL.Image.open(image_path)
    except PIL.UnidentifiedImageError:
        print(f"Error: Unable to open image file {image_path}. Unsupported format.")
        return None, None

    response = model.generate_content([PROMPT, image])
    detections = parse_response(response, image.size, classes)

    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    box_annotator = sv.BoxAnnotator()

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(annotated_image, detections=detections)

    return annotated_image, detections

def save_json_result(detections, output_path):
    result = []
    for xyxy, class_name in zip(detections.xyxy, detections.class_name):
        result.append({
            "class": class_name,
            "bbox": xyxy.tolist()
        })
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

def main(args):
    genai.configure(api_key=args.google_api_key)
    model = genai.GenerativeModel(model_name=args.model)

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Get all image files in the input directory
    supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f)) and os.path.splitext(f)[1].lower() in supported_formats]

    for image_file in image_files:
        input_path = os.path.join(args.input_folder, image_file)
        print(f"Processing {input_path}...")
        annotated_image, detections = run_detection(model, input_path, args.classes)
        
        if annotated_image is None:
            continue

        # Save annotated image
        file_name, file_extension = os.path.splitext(image_file)
        output_image_path = os.path.join("output", f"{file_name}_annotated{file_extension}")
        annotated_image.save(output_image_path)
        print(f"Annotated image saved to {output_image_path}")

        # Save JSON result
        output_json_path = os.path.join("output", f"{file_name}_result.json")
        save_json_result(detections, output_json_path)
        print(f"Detection results saved to {output_json_path}")

        print("Detections:")
        if isinstance(detections, sv.Detections):
            for i, (xyxy, class_name) in enumerate(zip(detections.xyxy, detections.class_name)):
                print(f"  Detection {i+1}: Class: {class_name}, Bounding Box: {xyxy}")
        else:
            print("  Unexpected detection format. Raw detections:")
            print(detections)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Object Detection")
    parser.add_argument("--google_api_key", required=True, help="Google API Key")
    parser.add_argument("--classes", nargs="+", required=True, help="List of classes to detect")
    parser.add_argument("--model", default="gemini-1.5-flash", help="Gemini model to use (default: gemini-1.5-flash)")
    parser.add_argument("--input_folder", default=".", help="Input folder containing images (default: current directory)")

    args = parser.parse_args()
    main(args)