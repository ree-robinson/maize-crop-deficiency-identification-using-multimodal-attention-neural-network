import os

train_dir = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\images\test"
val_dir = r"C:\Users\Libin Josuva\Downloads\data\clean_dataset1\images\valid"

duplicate_count = 0

for class_name in os.listdir(train_dir):

    train_class_path = os.path.join(train_dir, class_name)
    val_class_path = os.path.join(val_dir, class_name)

    # Skip if not a folder
    if not os.path.isdir(train_class_path):
        continue

    if not os.path.exists(val_class_path):
        continue

    train_images = set(os.listdir(train_class_path))
    val_images = set(os.listdir(val_class_path))

    common_images = train_images.intersection(val_images)

    for img in common_images:
        val_img_path = os.path.join(val_class_path, img)
        if os.path.isfile(val_img_path):   # Ensure it's a file
            os.remove(val_img_path)
            duplicate_count += 1
            print(f"Removed duplicate image from validation: {class_name}/{img}")

print("\nTotal duplicate images removed:", duplicate_count)
print("✅ Duplicate cleanup completed.")