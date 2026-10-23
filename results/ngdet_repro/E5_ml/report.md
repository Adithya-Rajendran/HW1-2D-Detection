# ngdet evaluation report

Unified taxonomy: **driving5** = `['car', 'truck', 'bus', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.244 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.417 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 204.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.244 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.417 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.253 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.181 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.225 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.345 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.219 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.421 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.453 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.456 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.441 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.468 |

| Class | AP@[.5:.95] |
|---|---|
| car | 0.475 |
| truck | 0.259 |
| bus | 0.000 |
| person | 0.240 |
| cyclist | 0.001 |
