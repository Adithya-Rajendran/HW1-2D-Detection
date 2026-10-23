# ngdet evaluation report

Unified taxonomy: **driving5** = `['car', 'truck', 'bus', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.243 |
| hf_detr:PekingU/rtdetr_r50vd | 0.235 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.237 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.420 |
| hf_detr:PekingU/rtdetr_r50vd | 0.411 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.422 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 102.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.243 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.420 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.249 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.195 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.222 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.336 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.190 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.356 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.376 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.323 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.364 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.438 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.478 |
| truck | 0.247 |
| bus | 0.000 |
| person | 0.247 |
| cyclist | 0.000 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (1000 imgs, 224.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.411 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.205 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.215 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.198 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.398 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.449 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.447 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.426 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.492 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.501 |
| truck | 0.203 |
| bus | 0.000 |
| person | 0.237 |
| cyclist | 0.001 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (1000 imgs, 290.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.237 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.422 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.245 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.147 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.215 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.326 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.198 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.384 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.402 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.433 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.378 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.430 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.474 |
| truck | 0.224 |
| bus | 0.000 |
| person | 0.250 |
| cyclist | 0.001 |
