# READ ME

_Steps to run Reid person enabler on x86 device._

## 1. Download Weight File
Download `checkpoint.rar` from one of the two address below: 

[Baidu Netdisk] https://pan.baidu.com/s/1ZI_UVUsPC9NKPFmjyF4MiQ  
Password: 1234  

[Google Drive] https://drive.google.com/file/d/1QYQ6I3j8iwXWwnV6uKbKosRkzCpUpBdv/view?usp=sharing

## 2. Unzip
unzip it under `deep_sort/deep/`. 

## 3. Create Conda Environment
> Python version is 3.7 in our tests. Creating a new environment is recommended. 
```sh
conda create -n [NEW_ENVIRONMENT] python=3.7
# Enter y to confirm
conda activate [NEW_ENVIRONMENT]
```

## 4. Install Requirements
```sh
pip install -r requirements.txt
```
If you encounter errors similar to this:
```sh
ERROR: Could not find a version that satisfies the requirement **** (from versions: none)
ERROR: No matching distribution found for ****
```
This means pip cannot find the package on the mirror site in your environment settings. Please check the installation source settings or install the mentioned package seperately: 
```sh
pip install **** -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If you encounter `socket.timeout`, please retry later with better internet connections.  

## 5. Corrections For Torch
This model is based on yolov5, you may encouter the following error: 
```sh
AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
```
This may be caused by imcompability between different torch versions. We can simply solve this problem by modifying the file `upsampling.py`: 
```sh
vim [YOUR_ENVIRONMENT_PATH]/lib/python3.7/site-packages/torch/nn/modules/upsampling.py
```
In line 156 and 157, remove the unwanted parameter:
```python
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
                             # recompute_scale_factor=self.recompute_scale_factor)
```
Save and quit after modification. 

## 6. Prepare Test Image / Video / Webcam 
We did not provide a test video in this package.

## 7. Run File `person_count.py`

### Run with video input
Run directly by: 
```sh
python3 person_count.py --input_path '[YOUR_VIDEO_PATH]'
```
The accuracy is improved in `reid_improved.py`, it can be tested by: 
```sh
python3 reid_improved.py --input_path '[YOUR_VIDEO_PATH]'
```

### Run with image input
Argument `--img` needs to appear in order to run test on a image input:
```sh
python3 reid_improved.py --input_path '[YOUR_IMAGE_PATH]' --img
```
Or with the improved method: 
```sh
python3 reid_improved.py --input_path '[YOUR_IMAGE_PATH]' --img
```

### Improvements of the result
With any of the above files, use `--help` to see supported arguments.  
Specially, if different persons are not being well distinguished, you may run with the argument `--id_thres`. Larger threshold is recommended for crowded scenes.   
If the count is much larger than the number of persons, you may run with a higher confidence threshold by specifying the argument `--conf-thres`. 