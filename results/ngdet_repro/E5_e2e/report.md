# ngdet evaluation report

Unified taxonomy: **driving5** = `['car', 'truck', 'bus', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.238 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.409 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 100.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.409 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.246 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.159 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.227 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.213 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.402 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.452 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.469 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.439 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.459 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.471 |
| truck | 0.242 |
| bus | 0.000 |
| person | 0.237 |
| cyclist | 0.001 |
