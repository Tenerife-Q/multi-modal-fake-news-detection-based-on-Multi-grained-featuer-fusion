import torch
import torch.utils.data as data
import data.util as util
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoFeatureExtractor
import clip
import os

# Determine whether to use CUDA (GPU) or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and its preprocessing function
clipmodel, preprocess = clip.load('ViT-B/32', device)

# Freeze the parameters of the CLIP model
for param in clipmodel.parameters():
    param.requires_grad = False

# Load from local cache first to avoid network timeout during startup
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "microsoft/swin-base-patch4-window7-224", local_files_only=True
)
token = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)


def read_img(imgs, root_path, LABLEF):
    # Select a random image path from the provided list
    GT_path = imgs[np.random.randint(0, len(imgs))]
    if '/' in GT_path:
        GT_path = GT_path[GT_path.rfind('/') + 1:]
    GT_path = "{}/{}/{}".format(root_path, LABLEF, GT_path)
    # Read the Ground Truth (GT) image
    try:
        img_GT = util.read_img(GT_path)
        img_pro = Image.open(GT_path).convert('RGB')
    except Exception as e:
        img_GT = np.zeros((224, 224, 3))
        img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
    return img_GT, img_pro


class weibo_dataset(data.Dataset):
    def __init__(self, is_train=True):
        super(weibo_dataset, self).__init__()
        self.label_dict = []
        self.swin = feature_extractor
        self.preprocess = preprocess
        self.local_path = '/root/autodl-tmp/MMFN_yyt'
        self.root_path = '/root/autodl-tmp/MMFN_yyt/weibo'
        # Read CSV file to populate label_dict
        gc = pandas.read_csv(self.local_path + '/{}_weibov.csv'.format('train' if is_train else 'test'))
        # gc = gc[:100]
        # Populate label_dict with records from the CSV file
        for i in tqdm(range(len(gc))):
            images_name = str(gc.iloc[i, 1])
            label = int(gc.iloc[i, 2])
            content = str(gc.iloc[i, 3])
            sum_content = str(gc.iloc[i, 4])
            has_image = (images_name != 'nan' and len(images_name.strip()) > 0)
            if not has_image:
                continue

            if label == 0:
                label_folder = 'rumor_images'
            else:
                label_folder = 'nonrumor_images'

            valid_images = []
            for img in images_name.split('|'):
                img_name = img.strip()
                if '/' in img_name:
                    img_name = img_name[img_name.rfind('/') + 1:]
                if not img_name:
                    continue
                img_path = os.path.join(self.root_path, label_folder, img_name)
                if os.path.exists(img_path):
                    valid_images.append(img_name)

            if len(valid_images) == 0:
                continue

            record = {}
            record['images'] = '|'.join(valid_images)
            record['label'] = label
            record['content'] = content
            record['sum_content'] = sum_content
            record['has_image'] = True
            self.label_dict.append(record)
        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, item):
        record = self.label_dict[item]
        images, label, content, sum_content, has_image = record['images'], record['label'], record['content'], record[
            'sum_content'], record['has_image']
        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'
        imgs = images.split('|')
        if has_image:
            img_GT, img_pro = read_img(imgs, self.root_path, LABLEF)
        else:
            img_GT = np.zeros((224, 224, 3))
            img_pro = Image.new('RGB', (224, 224), (255, 255, 255)).convert('RGB')
            # raise IOError("Load {} Error {}".format(imgs, record['images']))
        sent = content
        image_swin = self.swin(img_GT, return_tensors="pt").pixel_values
        image_clip = self.preprocess(img_pro)
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
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(image).squeeze(1)
    imageclip = torch.stack(imageclip)
    labels = torch.LongTensor(labels)
    sample_indices = torch.LongTensor(sample_indices)
    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels, sample_indices
