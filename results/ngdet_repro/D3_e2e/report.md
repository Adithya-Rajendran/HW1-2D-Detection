# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.247 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.457 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 106.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.247 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.457 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.167 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.240 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.317 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.110 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.363 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.322 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.367 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.386 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.495 |
| person | 0.244 |
| cyclist | 0.000 |
