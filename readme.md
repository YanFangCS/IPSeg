# IPSeg: Image Posterior Mitigates Semantic Drift in Class-Incremental Segmentation

### Datasets
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
        --- saliency_map/
        
    --- ADEChallengeData2016/
        --- annotations/
        --- images/
        --- sal/
```

### Requirements
```
pip install -r requirements.txt
```
### Training
```
python run.py
```