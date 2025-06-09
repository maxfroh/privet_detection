import os, shutil

OLD_DIR = r"C:\Users\Crack\2025_AI\Data"
NEW_DIR = r"C:\Users\Crack\2025_AI"
FOLDERS = ["DJI_202505291240_017_privet1a", "DJI_202505291312_018_privet1a"]

for f in FOLDERS:
    print(f"Copying {f}")
    old = os.path.join(OLD_DIR, f)
    new = os.path.join(NEW_DIR, f)
    os.mkdir(new)
    for file in os.listdir(old):
        if "_D" in file:
            shutil.copy(os.path.join(old, file), new)

    print("\tDone.")
print("Complete")