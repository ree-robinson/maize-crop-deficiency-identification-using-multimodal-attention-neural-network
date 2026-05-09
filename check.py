import os

train_dir = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\images\test"
val_dir = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\images\valid"

train_files = set(os.listdir(train_dir))
val_files = set(os.listdir(val_dir))

common_files = train_files.intersection(val_files)

print("Total Train Images:", len(train_files))
print("Total Validation Images:", len(val_files))
print("Common Images:", len(common_files))

if len(common_files) > 0:
    print("⚠ DATA LEAKAGE DETECTED!")
    print("Some duplicate filenames:")
    print(list(common_files)[:10])
else:
    print("✅ No duplicate filenames found.")