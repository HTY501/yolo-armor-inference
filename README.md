# YOLO Armor Inference

YOLO armor detection &amp; inference assignment
# YOLO 装甲板视频推理与关键点检测

## 1. 项目功能
本项目使用统一提供的 YOLO26n 装甲板模型 `best.pt`，对自行拍摄的装甲板视频进行推理。
能够识别 B1、B2、B3、R1、R2、R3 六类装甲板，并预测每个目标的 4 个二维关键点。
推理结果能够显示检测框、类别、置信度，并将带有预测关键点的结果视频保存到本地。

## 2. 运行环境
- **Python 版本**：3.14
- **主要依赖**：ultralytics, opencv-python, torch (CPU版本)
- **运行设备**：CPU（未配置显卡直通，推理速度约 25ms/帧，满足非实时推理要求）

## 3. 模型权重准备
从课程作业参考材料中下载 `best.pt` 文件，并将其放置于项目根目录（与 `infer.py` 同级目录）下。

## 4. 安装与运行
依赖安装
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
运行推理

python infer.py --model best.pt --source armor.mp4 --output output_result.mp4
--model: 模型权重路径（默认 best.pt）

--source: 输入视频路径（默认 armor.mp4）

--output: 输出结果保存路径（默认 output_result.mp4）
运行结束后，结果视频将保存在项目根目录下的 output_result.mp4。
5. 视频拍摄说明
拍摄地点：水建学院 D402

视频时长：约 19 秒

拍摄场景：横屏拍摄，包含清晰的 B3 类型装甲板，拍摄时适当调整了距离和角度以测试模型鲁棒性。

7. 识别表现与问题记录
识别表现：模型对 B3 类型的装甲板识别效果良好，置信度较高，关键点位置基本准确。

遇到的问题及处理：

拍摄角度过大时，偶有漏检。后续通过尽量正对装甲板拍摄解决。

运行环境中缺少 libGL.so.1，通过执行 sudo apt install libgl1-mesa-glx 解决。
