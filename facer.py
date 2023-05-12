import csv
import os
from PIL import Image, ImageDraw

import face_recognition


IMAGES_SOURCE_DIR = "images/yalefaces"
OUTPUT_DIR = "results/yalefaces_output"
OUTPUT_CSV = "results/yalefaces.csv"


def read_images_folder():
    images = []
    for img in os.listdir(IMAGES_SOURCE_DIR):
        images.append((img, os.path.join(IMAGES_SOURCE_DIR, img)))
    return images


def format_csv_result(img, coordinates):
    success = len(coordinates) > 0
    return [img, success, coordinates]


if __name__ == '__main__':

    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []
    for img_name in os.listdir(IMAGES_SOURCE_DIR):
        if img_name.endswith(".jpeg"):
            fr_image = face_recognition.load_image_file(os.path.join(IMAGES_SOURCE_DIR, img_name))
            face_encoding = face_recognition.face_encodings(fr_image)[0]
            known_face_encodings.append(face_encoding)
            name = img_name.split(".")[0]   # img_name eg. "subject01.jpeg"
            known_face_names.append(name)
            print("Learned encoding for " + name)

    csv_results = [
        ["Photo", "Subject", "Problem", "Successful", "Recognized", "%"]
    ]

    # Create output dir if not exist
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # Process images
    for img_name in os.listdir(IMAGES_SOURCE_DIR):
        subject_name = img_name.split('.')[0]    # img_name eg. "subject01.happy"
        test_type = img_name.split('.')[1]
        image_path = os.path.join(IMAGES_SOURCE_DIR, img_name)
        image = face_recognition.load_image_file(image_path)
        pil_image = Image.fromarray(image)

        # Get faces coordinates
        face_locations = face_recognition.face_locations(image)
        recognized_subject = "UNKNOWN"
        recognized_cert = 0
        if face_locations:

            # Match against known faces
            face_encodings = face_recognition.face_encodings(image, face_locations)
            face_encoding = face_encodings[0]   # There is always 1 face in the image
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = 0
            best_match_value = 2    # Match values are between 0 and 1
            for i in range(len(face_distances)):
                if face_distances[i] < best_match_value:
                    best_match_value = face_distances[i]
                    best_match_index = i

            if matches[best_match_index]:
                recognized_subject = subject_name
                recognized_cert = round((1 - best_match_value) * 100, 2)
            label = "{}%  {}".format(recognized_cert, recognized_subject)

            # Draw result rectangle
            draw = ImageDraw.Draw(pil_image)
            face_location = face_locations[0]
            top = face_location[0]
            right = face_location[1]
            bottom = face_location[2]
            left = face_location[3]
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=4)

            # Draw label
            text_width, text_height = draw.textsize(label)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), label, fill=(255, 255, 255, 255))
            del draw

        pil_image.save(os.path.join(OUTPUT_DIR, img_name + ".jpeg"), format="JPEG")

        csv_row = [img_name, subject_name, test_type, recognized_cert > 0, recognized_subject, recognized_cert]
        print(csv_row)
        csv_results.append(csv_row)
        pil_image.close()

    # Write csv results
    with open(OUTPUT_CSV, 'w') as csv_result_file:
        writer = csv.writer(csv_result_file)
        writer.writerows(csv_results)
