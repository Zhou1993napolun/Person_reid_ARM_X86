# Person_reid_ARM_X86

## This project is suitable for both ARM and x86 environment.



### Some of the files were not uploaded because they were too large:

```
./deep_sort/deep/checkpoint
```

For the `./deep_sort/deep/checkpoint` the link is : `https://pan.baidu.com/s/1ZI_UVUsPC9NKPFmjyF4MiQ` And the password is : `1234`

or you can try this link to download the file : `https://drive.google.com/file/d/1QYQ6I3j8iwXWwnV6uKbKosRkzCpUpBdv/view?usp=sharing`



**The python version of these project is 3.7**

After download these weights file and put it to the correct place, we should run this command to setup our environment.

```
pip install -r requirement.txt
```

use the command :

```
python3 person_count.py --video_path 'your video path'
```

if you have this problem ：

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

Then use this command to install opencv-python-headless :

```
pip install opencv-python-headless
```

After install all the dependences of the project. We should go to our virtual environment path:

```
cd your_virtual_environment_path/lib/lib/python3.7/site-packages/torch/nn/modules/upsampling.py
```

And change the `upsampling.py` file. In line 152:

```
def forward(self, input: Tensor) -> Tensor:
# 	return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
	return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)
```

And if your environment is x86 environment, please open the `reid_clothes.py` file to the file 265:

```
ln = self.clo_net1.getLayerNames()
# ln = [ln[i - 1] for i in self.clo_net1.getUnconnectedOutLayers()]
ln = [ln[i[0] - 1] for i in self.clo_net1.getUnconnectedOutLayers()]
```

Finally, we only should use these command to run the project.

```
python3 person_count.py --video_path 'your video path'
```

And we improved the accuracy of the reid. And the new file is `reid_improved.py` you can use this command to use this file:

```
python3 reid_improved.py --video_path 'your video path'
```

And to accommodate reid detection in a wider range of situations. We add `self.new_ID_Threshold` in line 105. You can adjust this parameter, which ranges from 0 to 1, depending on how densely populated the scene is, and turn it down when the scene is densely populated. When the scene is sparse with people, you can turn the parameter up.


For the camera as input option, we will add it in a later optimization.
