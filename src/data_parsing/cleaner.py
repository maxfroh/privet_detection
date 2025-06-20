import os, sys

DIR = "insert_label_directory_here"

files = os.listdir(DIR)

def remove_extra_classes():
    for file in files:
        if "classes" in file:
            continue
        else:
            should_fix = False
            with open(os.path.join(DIR, file), mode="r", encoding="utf-8") as f:
                lines = f.readlines()
            
            cleaned_lines = []
            for i in range(len(lines)):
                if lines[i][0] == "2":
                    should_fix = True
                elif lines[i][0] == "3":
                    cleaned_lines.append(f"0{lines[i][1:]}")
                    should_fix = True
            if should_fix:
                print(f"Cleaned {file}")
                with open(os.path.join(DIR, file), mode="w", encoding="utf-8") as f:
                    f.writelines(cleaned_lines)      
                
if __name__ == "__main__":
    remove_extra_classes()