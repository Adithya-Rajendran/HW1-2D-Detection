# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.243 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| yolo:yolo26x.pt | 0.446 |

## Full COCO tables (per run)

### yolo:yolo26x.pt  ×  kitti  (1000 imgs, 207.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.243 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.446 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.237 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.321 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.115 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.343 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.378 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.331 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.382 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.398 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.487 |
| person | 0.240 |
| cyclist | 0.001 |
