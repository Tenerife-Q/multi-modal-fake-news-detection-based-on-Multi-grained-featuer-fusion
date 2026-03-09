import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pandas import json_normalize
from sklearn.model_selection import train_test_split


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / 'gossip'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare the GossipCop dataset splits.')
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR,
                        help='Root directory that contains the gossip dataset (default: %(default)s).')
    parser.add_argument('--json-file', type=str, default=None,
                        help='JSON file name or absolute path. Defaults to the first gossipcop*.json found.')
    parser.add_argument('--image-dir', type=Path, default=None,
                        help='Override the directory that stores *_top_img.png files.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of samples for the test split (default: %(default)s).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used by train_test_split (default: %(default)s).')
    return parser.parse_args()


def ensure_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'Directory not found: {resolved}')
    return resolved


def locate_json_file(dataset_dir: Path, explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        candidates.extend([Path('gossipcop_v3_origin.json'), Path('gossipcop_v3-1_style_based_fake.json')])
        candidates.extend(sorted(dataset_dir.glob('gossipcop*.json')))

    for candidate in candidates:
        candidate_path = candidate if candidate.is_absolute() else (dataset_dir / candidate)
        candidate_path = candidate_path.resolve()
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError('Could not find a gossipcop JSON file. Use --json-file to specify one explicitly.')


def locate_image_dir(dataset_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        image_dir = explicit if explicit.is_absolute() else (dataset_dir / explicit)
        image_dir = image_dir.resolve()
        if not image_dir.exists():
            raise FileNotFoundError(f'Image directory not found: {image_dir}')
        return image_dir

    for candidate in [dataset_dir / 'top_img', dataset_dir / 'image' / 'top_img', dataset_dir / 'images' / 'top_img']:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError('Could not auto-detect the image directory. Provide --image-dir manually.')


def to_records(raw_json: object) -> List[dict]:
    if isinstance(raw_json, list):
        return raw_json
    if isinstance(raw_json, dict):
        return list(raw_json.values())
    raise TypeError('Unsupported JSON structure. Expected list or dict of records.')


def pick_column(columns: Iterable[str], candidates: List[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f'None of the columns {candidates} were found in the dataset.')


def normalize_label(value) -> int:
    if pd.isna(value):
        raise ValueError('Encountered missing label value.')
    text = str(value).strip().lower()
    if text in {'real', 'true', '1', 'non-rumor', 'nonrumor'}:
        return 1
    if text in {'fake', 'false', '0', 'rumor'}:
        return 0
    raise ValueError(f'Unsupported label value: {value}')


def build_dataframe(json_path: Path, image_dir: Path) -> pd.DataFrame:
    with json_path.open('r', encoding='utf-8') as source:
        raw_data = json.load(source)

    df = json_normalize(to_records(raw_data))
    label_col = pick_column(df.columns, ['generated_label', 'label', 'veracity'])
    id_col = pick_column(df.columns, ['origin_id', 'id', 'news_id'])

    df = df.copy()
    df['label'] = df[label_col].apply(normalize_label)
    df['image'] = df[id_col].apply(lambda x: str((image_dir / f'{x}_top_img.png').resolve()))
    df['has_top_img'] = df['image'].apply(lambda p: Path(p).exists())
    return df


def main() -> None:
    args = parse_args()
    dataset_dir = ensure_directory(args.dataset_dir)
    json_path = locate_json_file(dataset_dir, args.json_file)
    image_dir = locate_image_dir(dataset_dir, args.image_dir)

    print(f'Using dataset directory: {dataset_dir}')
    print(f'Using JSON file: {json_path}')
    print(f'Using image directory: {image_dir}')

    full_df = build_dataframe(json_path, image_dir)
    output_csv = dataset_dir / 'gossipcop.csv'
    full_df.to_csv(output_csv, index=False)
    print(f'Saved full dataset with {len(full_df)} rows to {output_csv}')

    train_df, test_df = train_test_split(full_df, test_size=args.test_size, random_state=args.seed)
    train_csv = dataset_dir / 'train_gossipcop.csv'
    test_csv = dataset_dir / 'test_gossipcop.csv'
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f'Train split: {len(train_df)} rows -> {train_csv}')
    print(f'Test split: {len(test_df)} rows -> {test_csv}')


if __name__ == '__main__':
    main()
