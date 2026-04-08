import os
import shutil

def split_real_imagine_files():
    base_dir = "D:\inżynierka\motor imagery BCI\data\\all\\files"
    real_dir = "D:\inżynierka\motor imagery BCI\data\\real_motion"
    imagine_dir = "D:\inżynierka\motor imagery BCI\data\imagine_motion"

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(imagine_dir, exist_ok=True)

    real_tasks = ["R03", "R07", "R11"]
    imagine_tasks = ["R04", "R08", "R12"]

    for i in range(1, 110):
        folder_name = f"S{i:03d}"
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if not (file.endswith(".edf") or file.endswith(".edf.event")):
                continue

            if any(task in file for task in real_tasks):
                shutil.copy(file_path, os.path.join(real_dir, file))

            elif any(task in file for task in imagine_tasks):
                shutil.copy(file_path, os.path.join(imagine_dir, file))

