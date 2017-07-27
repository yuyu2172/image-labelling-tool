import subprocess
import os
import fire


def main():
    for i in range(300):
        fn = os.path.expanduser('~/data/validation_videos/dataset34/frame{:04}.jpg'.format(i))
        if not os.path.exists(fn):
            continue
        subprocess.call(
            'python forward_one_image.py --pretrained_model model_iter_9000 '
            '{} train/at_home_nagoya_label_names.yml'.format(fn),
            shell=True
        )


if __name__ == '__main__':
    fire.Fire(main)

