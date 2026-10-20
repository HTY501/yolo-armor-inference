## 第二次作业：模型导出与性能对比

### 1. 模型信息
- 原始模型：YOLOv8 (或你的模型名) `.pt` 格式
- 导出模型：ONNX `.onnx` 格式

### 2. 导出方法
使用 Ultralytics 提供的 `export` 接口，运行 `python export.py`，将 `.pt` 权重转换为通用交换格式 `.onnx`。

### 3. 运行环境
- Python 3.x
- 核心依赖：ultralytics, onnx, onnxruntime, opencv-python
- 运行设备：CPU (虚拟机环境，未进行特定硬件推理加速)

### 4. 运行命令
```bash
# 运行对比推理测试
python compare.py
```
5. 结果对比截图
原始 .pt 模型推理结果：
https://result_pt.png/

导出 .onnx 模型推理结果：
https://result_onnx.png/

6. 性能对比数据
对比项目	原始 .pt 模型	导出 .onnx 模型
模型格式	.pt	.onnx
模型文件大小	5.62 MB	9.72 MB
平均推理耗时 (CPU)	46.79 ms	71.98 ms
检测效果	正常检测出 B3 目标	正常检测出 B3 目标，框和关键点与 pt 基本一致
7. 实验结论与思考
体积差异：导出的 ONNX 模型（9.72MB）反而比原始 PyTorch 模型（5.62MB）更大。这是因为 .pt 是高度压缩的权重文件，而 .onnx 包含完整的计算图结构和未压缩的权重数据。

速度差异：在当前的纯 CPU 虚拟机环境下，ONNX 的推理耗时（71.98ms）慢于 .pt（46.79ms）。这主要是因为 PyTorch 在 CPU 上的底层计算库非常成熟且高效。

格式理解：虽然在此次实验中 ONNX 没有展现出优势，但它的核心价值在于跨平台通用性。如果要真正发挥部署格式的加速效果，通常需要结合特定硬件的推理引擎（如 TensorRT）。
