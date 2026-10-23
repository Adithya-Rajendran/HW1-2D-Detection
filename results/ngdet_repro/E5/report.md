# ngdet evaluation report

Unified taxonomy: **driving5** = `['car', 'truck', 'bus', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.238 |
| hf_detr:PekingU/rtdetr_r50vd | 0.230 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.228 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.408 |
| hf_detr:PekingU/rtdetr_r50vd | 0.397 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.402 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 98.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.408 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.246 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.171 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.219 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.335 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.190 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.356 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.376 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.323 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.364 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.438 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.466 |
| truck | 0.245 |
| bus | 0.000 |
| person | 0.240 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (1000 imgs, 211.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.230 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.397 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.172 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.212 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.324 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.198 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.398 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.449 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.447 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.426 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.492 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.486 |
| truck | 0.200 |
| bus | 0.000 |
| person | 0.231 |
| cyclist | 0.001 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (1000 imgs, 289.0s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.228 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.402 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.107 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.212 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.198 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.384 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.402 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.433 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.378 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.430 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.452 |
| truck | 0.220 |
| bus | 0.000 |
| person | 0.241 |
| cyclist | 0.001 |
