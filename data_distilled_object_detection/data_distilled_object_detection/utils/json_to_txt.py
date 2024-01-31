import json

# Specify the path to your JSON file
json_file_path = "output/annotations_filtered_80/apple_train_coco_format.json"


# Specify the path to the text file where you want to save the converted data
text_file_path = "apple_output_text_file_from_json.txt"

# Read the JSON file
with open(json_file_path, 'r') as json_file:
    # Load the JSON data
    json_data = json.load(json_file)

# Open the text file in write mode and write the JSON data as text
with open(text_file_path, 'w') as text_file:
    # Convert the JSON data to a string and write it to the text file
    text_file.write(json.dumps(json_data, indent=2))  # Use indent for pretty formatting, adjust as needed
