import os
import shutil


source_folder = r"C:\Users\Crack\2025_AI\Privet1a_012"
destination_folder = os.path.join(source_folder, "filtered")

# Create destination folder
os.makedirs(destination_folder, exist_ok=True)

# Get sorted list of all .jpg files only
jpg_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")])

# Move every 5th .jpg and its corresponding .txt
for i in range(0, len(jpg_files),5):
    image_file = jpg_files[i]
    base_name = os.path.splitext(image_file)[0]
    label_file = base_name + ".txt"

    # Full source paths
    image_src = os.path.join(source_folder, image_file)
    label_src = os.path.join(source_folder, label_file)

    # Full destination paths
    image_dst = os.path.join(destination_folder, image_file)
    label_dst = os.path.join(destination_folder, label_file)

    # Move image
    shutil.move(image_src, image_dst)
    print(f"Moved image: {image_file}")

    # Move label if it exists
    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
        print(f"Moved label: {label_file}")
    else:
        print(f"Label not found for {image_file}")

print(" Done moving every 5th image and matching label.")
