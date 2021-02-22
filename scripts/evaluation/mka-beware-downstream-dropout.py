"""The generated cookie downstream dataset has a class imbalance.
85% of noreps and 15% of reps. Therefore, this script generates sub-sets with
desired class ratios.


MKA beware 1.1 cookie dataset:

    train:
        norep:  # 603518 files
        rep:    # 108599 files

    val:
        norep:  # 67059 files
        rep:    # 12642 files
"""

import glob
import os
import random
import shutil

random.seed(42)


def create_dropout_set(percent: float):
    """
    Args:
        prob (float): The percentage to use from the norep distribution.
    """
    print('------------------------')
    print(f'Start creating dropout {percent} set.')
    for _type in ['motion_images', 'sequence_chunks']:
        for _set in ['train', 'val']:
            for _class in ['rep', 'norep']:
                print(f'.. creating {_type} - {_set} dataset.')
                os.makedirs('data/mka-beware-1.1/' + _type +
                            '/mka-beware-1.1_cookie_dropout-' + str(percent) +
                            '/' + _set + '/' + _class,
                            exist_ok=True)
                for idx, _file in enumerate(
                        sorted(
                            list(
                                glob.glob('data/mka-beware-1.1/' + _type +
                                          '/mka-beware-1.1_cookie/' + _set +
                                          '/' + _class + '/*')))):
                    if idx % 1000 == 0:
                        print(f'.. processed: {idx}')

                    # skip 1 - percentage frames of norep
                    if _class == 'norep':
                        if not random.random() < percent:
                            continue

                    shutil.copy(
                        _file, 'data/mka-beware-1.1/' + _type +
                        '/mka-beware-1.1_cookie_dropout-' + str(percent) + '/' +
                        _set + '/' + _class)
    print('DONE')
    print('------------------------')


if __name__ == "__main__":
    create_dropout_set(0.2)
    create_dropout_set(0.4)