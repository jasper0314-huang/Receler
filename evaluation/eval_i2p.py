import os.path
import glob
import pandas as pd
import argparse


def main(file_path):
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        raise AttributeError(f'{file_path} is not a file.')

    print(len(df.index))

    categories = set(', '.join(df['categories'].unique()).split(', '))
    for c in categories:
        print('- ' * 42)
        print('categories:', c)
        df_c = df[df['categories'].str.contains(c)]
        print(f"\033[1mUnsafe Prop:\033[0m {100 * df_c['unsafe'].mean():0.4f}%")

    print('- ' * 42)
    print('categories:', 'all')
    print(f"\033[1mUnsafe Prop:\033[0m {100 * df['unsafe'].mean():0.4f}%")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--csv', type=str, required=True)

    args = parser.parse_args()
    main(file_path=args.csv)
