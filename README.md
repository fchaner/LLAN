# LLAN
Our project is based on MMEditing. 
However, MMEditing and MMGeneration were replaced by [MMagic](https://github.com/open-mmlab/mmagic) nowadays.
Therefore, Running this project requires downloading [MMagic](https://github.com/open-mmlab/mmagic)
and placing the model and configuration files in the corresponding locations. 
Among them, the class in <font color=#00FFFF>sr_backbone</font> 
and <font color=#00FFFF>llan</font> need to be registered in the initialization file, 
and then the model needs to be registered in the <font color=#00FFFF>editors</font> package.
The training and testing steps are consistent with other models in [MMagic](https://github.com/open-mmlab/mmagic).