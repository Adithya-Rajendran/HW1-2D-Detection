# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.242 |
| hf_detr:PekingU/rtdetr_r50vd | 0.244 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.234 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.444 |
| hf_detr:PekingU/rtdetr_r50vd | 0.444 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.434 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 99.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.242 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.444 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.236 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.320 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.108 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.309 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.279 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.339 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.391 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.486 |
| person | 0.240 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (1000 imgs, 209.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.244 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.444 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.136 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.107 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.318 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.360 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.324 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.355 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.395 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.499 |
| person | 0.231 |
| cyclist | 0.001 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (1000 imgs, 289.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.234 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.434 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.117 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.107 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.310 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.332 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.370 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.460 |
| person | 0.241 |
| cyclist | 0.000 |
