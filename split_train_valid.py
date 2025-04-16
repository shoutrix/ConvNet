import os
import random
import shutil


train_data_path = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/train"
valid_data_path = "/speech/shoutrik/Database/INaturalist/inaturalist_12K/valid"


classes = os.listdir(train_data_path)

for class_ in classes:
    if not class_.startswith("."):
        os.makedirs(os.path.join(valid_data_path, class_), exist_ok=True)
        all_files = os.listdir(os.path.join(train_data_path, class_))
        
        # for file_ in all_files:
        #     src = os.path.join(valid_data_path, class_, file_)
        #     dst = os.path.join(train_data_path, class_, file_)
            
        #     shutil.move(src, dst)
        
        random.shuffle(all_files)
        
        n_valid = int(len(all_files) * 0.2)
        
        valid_files = all_files[:n_valid]
        src_paths = [os.path.join(train_data_path, class_, f) for f in valid_files]
        dst_paths = [os.path.join(valid_data_path, class_, f) for f in valid_files]
        
        for s, d in zip(src_paths, dst_paths):
            shutil.move(s, d)
            
        