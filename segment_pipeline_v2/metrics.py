import torch


class SegMetrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.hist = torch.zeros(num_classes, num_classes, device=device)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(
            target[keep] * self.num_classes + pred[keep],
            minlength=self.num_classes**2,
        ).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious = (ious * 100).cpu().numpy().round(2).tolist()
        miou = round(miou * 100, 2)
        return ious, miou

    def compute_dice(self):
        dice = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mdice = dice[~dice.isnan()].mean().item()
        dice = (dice * 100).cpu().numpy().round(2).tolist()
        mdice = round(mdice * 100, 2)
        return dice, mdice
