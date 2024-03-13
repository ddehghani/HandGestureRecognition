import torch
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset
import cv2 as cv
from torchvision.transforms import v2
import warnings

warnings.filterwarnings("ignore")
train_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize(54),              # resize shortest side
        v2.CenterCrop(96),          # crop longest side
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])
])


class VideoStreamDataset (IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        cap = cv.VideoCapture(self.file_path)
        while True:
            _, frame = cap.read()
            if frame is None:
                return None
            frame = bgremove(frame)
            frame = train_transform(frame)
            yield frame


def bgremove(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    return thresh


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def plot_accuracies(history, path):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(path)


def plot_losses(history, path):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(path)


def augmentImage(img, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (760, 400)
    fontScale = 6
    fontColor = (255, 255, 255)
    thickness = 5
    lineType = 2

    cv.putText(img, str(text), bottomLeftCornerOfText, font,
               fontScale, fontColor, thickness, lineType)
