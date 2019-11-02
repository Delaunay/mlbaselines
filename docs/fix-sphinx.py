import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')

    args = parser.parse_args()

    for file in os.listdir(args.input):

        full_path = file.split('.')

        # ignore olympus dir
        path = os.path.join(args.output,*full_path[1:-2])
        file_name = '.'.join(full_path[-2:])

        new_file = os.path.join(path, file_name)
        old_file = os.path.join(args.input, file)

        os.makedirs(path, exist_ok=True)
        os.rename(old_file, new_file)

        print(f'{old_file} => {new_file}')

    # find the index files and rename them
    for file in os.listdir(args.output):
        if file.endswith('.rst'):
            folder_name = file.rsplit('.', maxsplit=1)[0]
            folder_name = os.path.join(args.output, folder_name)

            if os.path.isdir(folder_name):
                old_file = os.path.join(args.output, file)
                new_file = os.path.join(folder_name, 'index.rst')

                os.rename(old_file, new_file)
                print(f'{old_file} => {new_file}')

if __name__ == '__main__':
    main()
