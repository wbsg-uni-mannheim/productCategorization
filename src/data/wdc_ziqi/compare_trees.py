import logging
import pickle
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parents[3]
    file_path = project_dir.joinpath('data/raw/icecat/tree/tree_icecat_without_encoding.pkl')
    with open(file_path, 'rb') as encoder_file:
        icecat_tree = pickle.load(encoder_file)

    print(icecat_tree.nodes)
    print(icecat_tree.edges)

    project_dir = Path(__file__).resolve().parents[3]
    file_path = project_dir.joinpath('data/raw/wdc_ziqi/tree/tree_wdc_ziqi.pkl')
    with open(file_path, 'rb') as encoder_file:
        wdc_ziqi_tree = pickle.load(encoder_file)

    print(wdc_ziqi_tree.nodes)
    print(wdc_ziqi_tree.edges)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()