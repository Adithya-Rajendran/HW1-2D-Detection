# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.236 |
| hf_detr:PekingU/rtdetr_r50vd | 0.240 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.236 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.434 |
| hf_detr:PekingU/rtdetr_r50vd | 0.441 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.460 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (120 imgs, 428.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.434 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.136 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.230 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.344 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.091 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.284 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.291 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.204 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.291 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.382 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.472 |
| person | 0.236 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (120 imgs, 43.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.240 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.441 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.222 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.134 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.344 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.087 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.285 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.298 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.239 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.291 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.380 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.489 |
| person | 0.232 |
| cyclist | 0.000 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (120 imgs, 41.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.460 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.204 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.231 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.332 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.083 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.292 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.313 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.290 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.303 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.395 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.462 |
| person | 0.246 |
| cyclist | 0.000 |
