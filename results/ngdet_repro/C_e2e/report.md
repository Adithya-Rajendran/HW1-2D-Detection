# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.239 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.441 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 100.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.239 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.441 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.132 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.236 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.316 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.110 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.363 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.322 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.367 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.386 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.481 |
| person | 0.237 |
| cyclist | 0.000 |
