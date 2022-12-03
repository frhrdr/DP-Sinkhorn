import os
import argparse


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir", type=str)
  args = parser.parse_args()
  for subdir, subsubdirs, files in os.walk(args.dir):
    first_print = True
    for file in files:
      if file.startswith('fid_ep_'):
        if first_print:
          print(f'{subdir}')
          first_print = False
        with open(os.path.join(subdir, file)) as f:
          print(f'{file}: {f.readline(file)}')


if __name__ == '__main__':
  main()
