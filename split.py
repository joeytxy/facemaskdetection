import shutil, random, os
dirpath =r"dataset_trial\unmask_train"
destDirectory =r"dataset_trial\unmask_test"

filenames = random.sample(os.listdir(dirpath), 120)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)
    os.remove(srcpath)

dirpath =r"dataset_trial\mask_train"
destDirectory =r"dataset_trial\mask_test"

filenames = random.sample(os.listdir(dirpath), 32)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)
    os.remove(srcpath)
