import os
import sys
from glob import glob


def get_bin_info(file_path, info_name, width, height):
    bin_images = glob(os.path.join(file_path, r'*/*.bin'))
    print(bin_images)
    with open(info_name, 'w') as file:
        for index, img in enumerate(bin_images):
            content = ' '.join([str(index), img, width, height])
            file.write(content)
            file.write('\n')


if __name__ == '__main__':
    '''
    using like:
    python3 get_info.py prep_img_bin/ ./pre_bin.info 1344 1344
    '''
    file_path = sys.argv[1]
    info_name = sys.argv[2]
    height = sys.argv[3]
    width = sys.argv[4]
    get_bin_info(file_path, info_name, width, height)

