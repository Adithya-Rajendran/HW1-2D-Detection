# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.248 |
| hf_detr:PekingU/rtdetr_r50vd | 0.248 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.238 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.466 |
| hf_detr:PekingU/rtdetr_r50vd | 0.461 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.464 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (120 imgs, 23.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.248 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.466 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.145 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.243 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.362 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.092 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.297 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.335 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.284 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.453 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.488 |
| person | 0.255 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (120 imgs, 38.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.248 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.461 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.226 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.139 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.366 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.088 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.303 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.359 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.347 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.465 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.498 |
| person | 0.245 |
| cyclist | 0.000 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (120 imgs, 35.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.464 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.205 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.139 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.335 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.085 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.295 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.324 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.300 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.399 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.466 |
| person | 0.247 |
| cyclist | 0.000 |
