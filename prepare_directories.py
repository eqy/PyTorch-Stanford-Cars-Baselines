import os
import shutil

import scipy.io

# preprocess the stanford cars dataset into directory-based orgranization
# for Pytorch ImageFolder style dataset
# TODO: consider cleaning this up or hardcoding the identities to remove the dependence on scipy/original MAT files
def preprocess_cars(cars_dir, meta_dir, cars_test_dir):
    meta_path = os.path.join(meta_dir, 'cars_meta.mat')
    annos_path = os.path.join(meta_dir, 'cars_train_annos.mat')
    test_annos_path = os.path.join(meta_dir, 'cars_test_annos_withlabels.mat')
    meta = scipy.io.loadmat(meta_path)
    annos = scipy.io.loadmat(annos_path)
    test_annos = scipy.io.loadmat(test_annos_path)

    classes = set()
    idx_to_class = dict()
    for row in annos['annotations'][0]:
        filename = os.path.basename(row[-1][0])
        idx = int(os.path.splitext(filename)[0])
        cur_class = int(row[-2][0][0])
        classes.add(cur_class)
        idx_to_class[idx] = cur_class

    assert len(classes) == 196

    for cur_class in classes:
        class_path = os.path.join(cars_dir, str(cur_class))
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    test_classes = set()
    test_idx_to_class = dict()

    for row in test_annos['annotations'][0]:
        filename = os.path.basename(row[-1][0])
        idx = int(os.path.splitext(filename)[0])
        cur_class = int(row[-2][0][0])
        test_classes.add(cur_class)
        test_idx_to_class[idx] = cur_class

    assert len(test_classes) == 196

    for cur_class in test_classes:
        class_path = os.path.join(cars_test_dir, str(cur_class))
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    count = 0

    for dirpath, _, filenames in os.walk(cars_dir):
        for filename in filenames:
            if os.path.basename(dirpath) != cars_dir:
                continue
            idx = int(os.path.splitext(filename)[0])
            assert idx in idx_to_class
            cur_class = idx_to_class[idx]
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(os.path.join(dirpath, str(cur_class)), filename)
            shutil.move(src_path, dst_path)
            count += 1
    print("files:", count, "directories:", len(classes))

    test_count = 0

    for dirpath, _, filenames in os.walk(cars_test_dir):
        for filename in filenames:
            if os.path.basename(dirpath) != cars_test_dir:
                continue
            idx = int(os.path.splitext(filename)[0])
            assert idx in test_idx_to_class
            cur_class = test_idx_to_class[idx]
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(os.path.join(dirpath, str(cur_class)), filename)
            shutil.move(src_path, dst_path)
            test_count += 1
    print("test files:", test_count, "directories:", len(test_classes))


def main():
    preprocess_cars('cars_train', 'devkit', 'cars_test') 

if __name__ == '__main__':
    main()
