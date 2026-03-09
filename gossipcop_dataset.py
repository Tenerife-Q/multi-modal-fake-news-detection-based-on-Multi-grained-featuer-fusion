import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip
import os
from entity_enricher import WikiEntityEnricher
# Determine whether to use CUDA (GPU) or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and its preprocessing function
clipmodel, preprocess = clip.load('ViT-B/32', device)

# Freeze the parameters of the CLIP model
for param in clipmodel.parameters():
    param.requires_grad = False

# Load a feature extractor from the transformers library
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224", local_files_only=True)
token = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
entity_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
entity_enricher = WikiEntityEnricher()


class gossipcop_dataset(data.Dataset):
    def __init__(self, is_train=True):
        super(gossipcop_dataset, self).__init__()
        self.label_dict = []
        self.missing_image_count = 0
        self.swin = feature_extractor
        self.preprocess = preprocess
        # Resolve dataset root dynamically with fallbacks
        preferred = []
        env_path = os.environ.get("GOSSIP_DATA_PATH")
        if env_path:
            preferred.append(env_path)
        preferred.extend([
            '/root/autodl-tmp/MMFN_yyt/gossip',
            os.path.join(os.path.dirname(__file__), 'gossip'),
            os.path.join(os.getcwd(), 'gossip'),
            '/home/yutao/MMFN/dataset/gossipcop',
        ])

        self.local_path = None
        target_csv = None
        default_csv_name = '{}_gossipcop.csv'.format('train' if is_train else 'test')
        env_csv_name = os.environ.get('GOSSIP_TRAIN_CSV' if is_train else 'GOSSIP_TEST_CSV')
        csv_candidates = [name for name in [env_csv_name, default_csv_name] if name]
        for p in preferred:
            for csv_name in csv_candidates:
                csv_path = os.path.join(p, csv_name)
                if os.path.exists(csv_path):
                    self.local_path = p
                    target_csv = csv_path
                    break
            if target_csv:
                break

        if not target_csv:
            tried = " | ".join(preferred)
            raise FileNotFoundError(f"gossipcop csv not found; tried: {tried}")

        # Read CSV file to populate label_dict
        gc = pandas.read_csv(target_csv)

        # Populate label_dict with records from the CSV file
        for i in tqdm(range(len(gc))):
            row = gc.iloc[i]

            # Text fields
            content = str(row.get('content') or row.get('text') or '')
            title = str(row.get('title') or '')
            sum_content = title if title and title != 'nan' else content

            # Label
            try:
                label = int(row.get('label'))
            except Exception:
                continue  # skip malformed rows

            # Image path is stored locally as <id>_top_img.png; skip if missing
            image_path = str(row.get('image') or '').strip()
            raw_flag = row.get('has_top_img', True)
            if isinstance(raw_flag, str):
                has_image_flag = raw_flag.strip().lower() in {'1', 'true', 'yes', 'y'}
            else:
                has_image_flag = bool(raw_flag)
            if not image_path or image_path == 'nan':
                continue
            if not os.path.exists(image_path):
                self.missing_image_count += 1
                continue
            if not has_image_flag:
                self.missing_image_count += 1
                continue

            record = {
                'image_path': image_path,
                'label': label,
                'content': content,
                'sum_content': sum_content,
            }
            self.label_dict.append(record)
        if self.missing_image_count > 0:
            print(f"[gossipcop_dataset] 跳过缺图样本: {self.missing_image_count}")

        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, item):
        record = self.label_dict[item]
        image_path, label, content, sum_content = record['image_path'], record['label'], record['content'], record['sum_content']
        sent = content
        try:
            img_rgb = Image.open(image_path).convert('RGB')
        except Exception:
            print(f"图片不存在: {image_path}")
            img_rgb = Image.new('RGB', (224, 224), (255, 255, 255))
        img_rgb = img_rgb.resize((224, 224))

        try:
            image_swin = self.swin(img_rgb, return_tensors="pt", do_normalize=True).pixel_values
        except Exception as e:
            # Fallback to blank RGB if any channel/format issue occurs
            print(f"swin preprocessing failed: {e}; mode={getattr(img_rgb, 'mode', None)}, shape={np.array(img_rgb).shape}")
            blank = Image.new('RGB', (224, 224), (255, 255, 255))
            image_swin = self.swin(blank, return_tensors="pt", do_normalize=True).pixel_values

        image_clip = self.preprocess(img_rgb)
        text_clip = sum_content
        return (sent, image_swin, image_clip, text_clip), label, item

    def __len__(self):
        return len(self.label_dict)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    pass


def collate_fn(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    imageclip = [i[0][2] for i in data]
    textclip = [i[0][3] for i in data]
    labels = [i[1] for i in data]
    sample_indices = [i[2] for i in data]

    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=300,
                                   return_tensors='pt',
                                   return_length=True)

    textclip = clip.tokenize(textclip, truncate=True)

    use_entity_enrich = os.environ.get("USE_ENTITY_ENRICH", "0") == "1"
    if use_entity_enrich:
        enrich_texts = [entity_enricher.build_background_text(sent) for sent in sents]
    else:
        enrich_texts = ["" for _ in sents]

    entity_data = entity_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=enrich_texts,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt',
        return_length=True
    )

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    entity_input_ids = entity_data['input_ids']
    entity_attention_mask = entity_data['attention_mask']
    entity_token_type_ids = entity_data['token_type_ids']
    image = torch.stack(image).squeeze(1)
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    sample_indices = torch.LongTensor(sample_indices)
    return (
        input_ids, attention_mask, token_type_ids,
        entity_input_ids, entity_attention_mask, entity_token_type_ids,
        image, imageclip, textclip, labels, sample_indices
    )
