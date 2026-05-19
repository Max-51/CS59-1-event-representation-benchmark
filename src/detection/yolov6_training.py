import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
YOLO_ROOT = ROOT / "third_party" / "ergo" / "ev-YOLOv6"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from yolov6.models.losses.loss import ComputeLoss
from yolov6.models.yolo import build_model
from yolov6.utils.config import Config
from src.detection.yolov6_common import collate_yolov6_samples


class DummyArgs:
    representation = "UnifiedAdapter"


def xywh2xyxy(boxes):
    output = boxes.clone() if isinstance(boxes, torch.Tensor) else np.copy(boxes)
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def box_iou(box1, box2):
    def _area(box):
        return (box[:, 2] - box[:, 0]).clamp(0) * (box[:, 3] - box[:, 1]).clamp(0)

    area1 = _area(box1)
    area2 = _area(box2)
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter + 1e-7)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    num_classes = prediction.shape[2] - 5
    pred_candidates = torch.logical_and(
        prediction[..., 4] > conf_thres,
        torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres,
    )
    max_wh = 4096
    max_nms = 30000
    multi_label &= num_classes > 1
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for img_idx, batch_pred in enumerate(prediction):
        batch_pred = batch_pred[pred_candidates[img_idx]]
        if not batch_pred.shape[0]:
            continue

        batch_pred[:, 5:] *= batch_pred[:, 4:5]
        boxes = xywh2xyxy(batch_pred[:, :4])

        if multi_label:
            box_idx, class_idx = (batch_pred[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            batch_pred = torch.cat(
                (boxes[box_idx], batch_pred[box_idx, class_idx + 5, None], class_idx[:, None].float()),
                dim=1,
            )
        else:
            conf, class_idx = batch_pred[:, 5:].max(1, keepdim=True)
            batch_pred = torch.cat((boxes, conf, class_idx.float()), dim=1)[conf.view(-1) > conf_thres]

        if classes is not None:
            batch_pred = batch_pred[(batch_pred[:, 5:6] == torch.tensor(classes, device=batch_pred.device)).any(1)]

        num_boxes = batch_pred.shape[0]
        if not num_boxes:
            continue
        if num_boxes > max_nms:
            batch_pred = batch_pred[batch_pred[:, 4].argsort(descending=True)[:max_nms]]

        class_offset = batch_pred[:, 5:6] * (0 if agnostic else max_wh)
        keep_idx = torchvision.ops.nms(batch_pred[:, :4] + class_offset, batch_pred[:, 4], iou_thres)
        output[img_idx] = batch_pred[keep_idx[:max_det]]

    return output


def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        matches = torch.where((iou >= iouv[i]) & correct_class)
        if matches[0].shape[0]:
            matched = (
                torch.cat((torch.stack(matches, 1), iou[matches[0], matches[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if matches[0].shape[0] > 1:
                matched = matched[matched[:, 2].argsort()[::-1]]
                matched = matched[np.unique(matched[:, 1], return_index=True)[1]]
                matched = matched[np.unique(matched[:, 0], return_index=True)[1]]
            correct[matched[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls):
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes = np.unique(target_cls)
    num_classes = unique_classes.shape[0]
    ap = np.zeros((num_classes, tp.shape[1]))
    precision = np.zeros((num_classes, 1000))
    recall = np.zeros((num_classes, 1000))
    px = np.linspace(0, 1, 1000)

    for class_idx, cls_id in enumerate(unique_classes):
        cls_mask = pred_cls == cls_id
        num_labels = (target_cls == cls_id).sum()
        num_preds = cls_mask.sum()
        if num_preds == 0 or num_labels == 0:
            continue

        false_positive = (1 - tp[cls_mask]).cumsum(0)
        true_positive = tp[cls_mask].cumsum(0)
        recall_curve = true_positive / (num_labels + 1e-16)
        precision_curve = true_positive / (true_positive + false_positive)
        recall[class_idx] = np.interp(-px, -conf[cls_mask], recall_curve[:, 0], left=0)
        precision[class_idx] = np.interp(-px, -conf[cls_mask], precision_curve[:, 0], left=1)

        for ap_idx in range(tp.shape[1]):
            ap[class_idx, ap_idx], _, _ = compute_ap(recall_curve[:, ap_idx], precision_curve[:, ap_idx])

    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    return precision, recall, ap, f1, unique_classes.astype("int32")


def build_yolov6_model(config_path, device, img_size, num_classes=2, number_of_channels=12):
    cfg = Config.fromfile(str(config_path))
    model = build_model(
        cfg,
        num_classes=num_classes,
        device=device,
        number_of_channels=number_of_channels,
        args=DummyArgs(),
    )
    criterion = ComputeLoss(
        num_classes=num_classes,
        ori_img_size=img_size,
        warmup_epoch=0,
        use_dfl=cfg.model.head.use_dfl,
        reg_max=cfg.model.head.reg_max,
        iou_type=cfg.model.head.iou_type,
        fpn_strides=cfg.model.head.strides,
    )
    return cfg, model, criterion


def create_optimizer(model, lr, momentum, weight_decay):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def create_scheduler(optimizer, epochs, min_lr_ratio=0.01):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio)


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_yolov6_samples,
        pin_memory=torch.cuda.is_available(),
    )


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def checkpoint_payload(epoch, model, optimizer, scheduler, best_metric, args_dict):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "args": args_dict,
    }


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def _loss_items_to_dict(loss, loss_items):
    return {
        "total": float(loss.detach().item()),
        "iou": float(loss_items[0].detach().item()),
        "dfl": float(loss_items[1].detach().item()),
        "cls": float(loss_items[2].detach().item()),
    }


def train_one_epoch(model, criterion, optimizer, dataloader, device, epoch_index, img_size, log_every=50):
    model.train()
    totals = []
    start = time.perf_counter()
    for step_idx, (images, labels, _, _) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)[0]
        loss, loss_items = criterion(
            predictions,
            labels,
            epoch_num=epoch_index,
            step_num=step_idx - 1,
            batch_height=img_size,
            batch_width=img_size,
        )
        loss.backward()
        optimizer.step()
        totals.append(_loss_items_to_dict(loss, loss_items))
        if log_every and (step_idx % log_every == 0 or step_idx == len(dataloader)):
            print(
                f"[train] epoch {epoch_index + 1} step {step_idx}/{len(dataloader)} "
                f"loss={totals[-1]['total']:.4f}"
            )

    mean_total = sum(item["total"] for item in totals) / max(len(totals), 1)
    return {
        "steps": len(totals),
        "seconds": round(time.perf_counter() - start, 2),
        "mean_total_loss": round(mean_total, 6),
        "last_loss": totals[-1] if totals else None,
    }


def evaluate_detection(
    model,
    criterion,
    dataloader,
    device,
    img_size,
    conf_thres=0.03,
    iou_thres=0.65,
    class_names=None,
):
    model.eval()
    class_names = tuple(class_names or ())
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels, _, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images)[0]
            detections = non_max_suppression(predictions, conf_thres, iou_thres, multi_label=True)

            for sample_idx, det in enumerate(detections):
                sample_labels = labels[labels[:, 0] == sample_idx, 1:]
                target_classes = sample_labels[:, 0].tolist() if len(sample_labels) else []
                if len(det) == 0:
                    if len(sample_labels):
                        stats.append(
                            (
                                torch.zeros((0, iouv.numel()), dtype=torch.bool),
                                torch.tensor([]),
                                torch.tensor([]),
                                target_classes,
                            )
                        )
                    continue

                correct = torch.zeros((det.shape[0], iouv.numel()), dtype=torch.bool, device=device)
                if len(sample_labels):
                    target_boxes = xywh2xyxy(sample_labels[:, 1:5])
                    target_boxes[:, [0, 2]] *= images[sample_idx].shape[2]
                    target_boxes[:, [1, 3]] *= images[sample_idx].shape[1]
                    label_boxes = torch.cat((sample_labels[:, 0:1], target_boxes), dim=1)
                    correct = process_batch(det, label_boxes, iouv)
                stats.append((correct.cpu(), det[:, 4].detach().cpu(), det[:, 5].detach().cpu(), target_classes))

    metric_payload = {
        "batches": len(dataloader),
        "seconds": round(time.perf_counter() - start, 2),
        "mean_total_loss": None,
        "last_loss": None,
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "per_class": {},
    }

    if not stats:
        return metric_payload

    stats = list(zip(*stats))
    if not stats:
        return metric_payload
    true_positives = np.concatenate([item.numpy() if torch.is_tensor(item) else item for item in stats[0]], axis=0)
    confidences = np.concatenate([item.numpy() if torch.is_tensor(item) else item for item in stats[1]], axis=0)
    predicted_classes = np.concatenate([item.numpy() if torch.is_tensor(item) else item for item in stats[2]], axis=0)
    target_classes = np.array([cls for group in stats[3] for cls in group], dtype=np.float32)
    if true_positives.size == 0 or target_classes.size == 0:
        return metric_payload

    precision, recall, ap, _, ap_class = ap_per_class(true_positives, confidences, predicted_classes, target_classes)
    ap50 = ap[:, 0]
    ap5095 = ap.mean(1)
    metric_payload["precision"] = round(float(precision[:, 0].mean()), 6)
    metric_payload["recall"] = round(float(recall[:, 0].mean()), 6)
    metric_payload["map50"] = round(float(ap50.mean()), 6)
    metric_payload["map50_95"] = round(float(ap5095.mean()), 6)
    metric_payload["per_class"] = {
        class_names[int(class_idx)] if int(class_idx) < len(class_names) else str(int(class_idx)): {
            "ap50": round(float(ap50[row]), 6),
            "ap50_95": round(float(ap5095[row]), 6),
        }
        for row, class_idx in enumerate(ap_class.tolist())
        if int(class_idx) >= 0
    }
    return metric_payload


def progress_payload(method, epoch, epochs, split_metrics=None, train_metrics=None, started_at=None, stage=None):
    elapsed = None if started_at is None else round(time.time() - started_at, 2)
    payload = {
        "method": method,
        "epoch": epoch,
        "epochs": epochs,
        "percent": round(100.0 * epoch / max(epochs, 1), 2),
        "stage": stage,
        "elapsed_seconds": elapsed,
    }
    if train_metrics is not None:
        payload["train"] = train_metrics
    if split_metrics is not None:
        payload.update(split_metrics)
    return payload
