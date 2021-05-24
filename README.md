# 基于CRNN模型的特定图片上的字符识别
- 步骤
```python
1:自制数据集工具TextGenerator
2:特定图片字符识别训练与测试
```

## 一、自制数据集工具TextGenerator
```python
# 实现的功能：
- 生成基于不同语料的，不同字体、字号、颜色、旋转角度的文字贴图
- 支持多进程快速生成
- 文字贴图按照指定的布局模式填充到布局块中
- 在图像中寻找平滑区域当作布局块
- 支持文字区域的图块抠取导出（导出json文件，txt文件和图片文件，可生成voc数据，coco格式coming soon!）
- 支持用户自己配置各项生成配(图像读取，生成路径，各种概率)
# 使用方式
- 环境安装(Python3.6+，建议使用conda环境)   
  pip install requirements.txt
  sh make.sh
- 编辑配置文件`config.yml`（可选） 
- 执行生成脚本
  python3 run.py
- 生成的数据
  生成的数据存放在`config.yml`中的`provider> layout> out_put_dir`指定的目录下。
```

## 二、特定图片字符识别训练与测试
```python
# 容器中创建python 虚拟环境
/usr/local/bin/python3.6 -m venv venv
# 激活虚拟环境
source venv/bin/activate
# 切换到项目主目录执行
pip install -r requirements.txt
# 设置训练与验证数据路径
vim lib/config/OWN_config.yaml
修改ROOT, JSON_FIL
# 训练模型
python train.py
# 测试模型
python demo.py
```
