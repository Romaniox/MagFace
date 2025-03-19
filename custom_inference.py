import os
import numpy as np
import cv2
import torch
from easydict import EasyDict

from inference.network_inf import builder_inf


def build_model(model_path: str, arch='iresnet50') -> torch.nn.Module:
    model_name = os.path.split(model_path)[-1]

    params = {'arch': arch,
              'inf_list': '',
              'feat_list': '',
              'workers': 4,
              'batch_size': 256,
              'embedding_size': 512,
              'resume': model_path,
              'print_freq': 100,
              'cpu_mode': False,
              'dist': 1}

    params = EasyDict(params)
    model = builder_inf(params)

    return model


def preproc(image: np.ndarray) -> torch.Tensor:
    image = cv2.resize(image, (112, 112))
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = image / 255.0
    image = torch.from_numpy(image)

    return image


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))

    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


if __name__ == '__main__':
    model_path = "/data/romanix/facedet/models/magface_epoch_00025.pth"

    model = build_model(model_path, arch='iresnet100')

    img_path = '/data/romanix/facedet/data/res/debug/(m=e-yaaGqaq)(mh=6cPiMZvXRnPAkFYe)original_743544971_face_0.png'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preproc(img)

    model.eval()
    with torch.no_grad():
        emb = model(img).detach().numpy()

        embedding = list(emb[0])

        mag = np.linalg.norm(embedding)

        print(mag)

