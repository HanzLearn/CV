import csv
import os
from itertools import groupby

def list_files_in_directory(directory):
    """List all files in the directory for debugging purposes."""
    print(f"Files in directory {directory}:")
    for filename in os.listdir(directory):
        print(filename)

def get_next_available_file(base_path, base_name, extension):
    """Find the next available file in the sequence that hasn't been processed yet."""
    counter = 1
    while True:
        input_file = os.path.join(base_path, f"{base_name}_{counter}{extension}")
        filtered_file = os.path.join(base_path, f"filtered_{base_name}_{counter}{extension}")
        
        if os.path.exists(input_file) and not os.path.exists(filtered_file):
            return input_file, filtered_file
        
        if not os.path.exists(input_file):
            return None, None
        
        counter += 1

def filter_and_group_rows(input_csv, output_csv):
    """Process the CSV file to filter and group rows."""
    try:
        with open(input_csv, mode='r') as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            if not rows:
                print(f"No data found in {input_csv}")
                return False
            
            rows.sort(key=lambda x: (x['rack_id'], x['frame_id']))
            grouped_rows = groupby(rows, key=lambda x: x['rack_id'])
            
            filtered_rows = []
            for rack_id, group in grouped_rows:
                for row in group:
                    filtered_rows.append({
                        'frame_id': row['frame_id'],
                        'rack_id': row['rack_id'],
                        'track_id': row['track_id'],
                        'class_id': row['class_id'],
                        'class_name': row['class_name'],
                        'cx': row['cx']
                    })

        print(f"Writing filtered data to: {output_csv}")
        with open(output_csv, mode='w', newline='') as outfile:
            fieldnames = ['frame_id', 'rack_id', 'track_id', 'class_id', 'class_name', 'cx']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)
        
        print(f"Successfully processed {input_csv} -> {output_csv}")
        return True

    except Exception as e:
        print(f"Error processing {input_csv}: {str(e)}")
        return False

def process_detection_files(base_dir='received_data/CSV'):
    """Process all available detection files in sequence."""
    base_name = 'detection_results'
    extension = '.csv'

    list_files_in_directory(base_dir)

    while True:
        input_file, output_file = get_next_available_file(base_dir, base_name, extension)
        if not input_file:
            print("No more files to process.")
            break

        print(f"Processing: {input_file} -> {output_file}")
        if filter_and_group_rows(input_file, output_file):
            print(f"Processed {os.path.basename(input_file)} successfully.")
        else:
            print(f"Skipping {os.path.basename(input_file)} due to errors.")

if __name__ == "__main__":
    process_detection_files()
