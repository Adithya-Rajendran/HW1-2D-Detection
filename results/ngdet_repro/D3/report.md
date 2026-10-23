# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.249 |
| hf_detr:PekingU/rtdetr_r50vd | 0.250 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.245 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.459 |
| hf_detr:PekingU/rtdetr_r50vd | 0.459 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.459 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 102.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.249 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.459 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.240 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.170 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.240 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.321 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.108 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.309 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.279 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.339 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.391 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.499 |
| person | 0.247 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (1000 imgs, 216.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.250 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.459 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.239 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.172 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.241 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.107 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.318 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.360 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.324 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.355 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.395 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.512 |
| person | 0.237 |
| cyclist | 0.001 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (1000 imgs, 290.3s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.245 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.459 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.162 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.107 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.310 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.332 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.370 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.484 |
| person | 0.250 |
| cyclist | 0.000 |
