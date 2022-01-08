import sys

import lib.datasets.segmentation


def main():
    task_name = sys.argv[1]

    tasks_dict = {
        'transform_segmentation_dataset': lib.datasets.segmentation.transform_dataset,
    }

    if task_name in tasks_dict:
        tasks_dict[task_name]()
    else:
        print('Task {} not defined'.format(task_name))


if __name__ == '__main__':
    main()
