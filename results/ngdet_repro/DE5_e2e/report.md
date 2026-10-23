# ngdet evaluation report

Unified taxonomy: **driving5** = `['car', 'truck', 'bus', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.244 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.423 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 107.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.244 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.423 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.251 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.189 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.230 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.213 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.402 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.452 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.469 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.439 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.459 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.488 |
| truck | 0.244 |
| bus | 0.000 |
| person | 0.244 |
| cyclist | 0.001 |
