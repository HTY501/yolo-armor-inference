import time
import os
import cv2
from ultralytics import YOLO

def measure_time(model, source, is_video=False, warmup=5, runs=20):
    """测量模型推理耗时的函数"""
    if is_video:
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()
        if not ret: return 0
    else:
        frame = cv2.imread(source)

    # 1. 预热 (Warmup) - 避免第一次运行时的冷启动延迟
    for _ in range(warmup):
        model(frame)

    # 2. 开始测量
    start_time = time.time()
    results = None
    for _ in range(runs):
        results = model(frame)
    end_time = time.time()

    avg_time = (end_time - start_time) / runs * 1000  # 转换为毫秒
    return avg_time, results

def main():
    # 测试数据（用你之前作业的视频或抽一帧图片）
    source = "armor.mp4" 
    
    # 1. 加载两个模型
    pt_model = YOLO("best.pt")
    onnx_model = YOLO("best.onnx")

    # 2. 获取文件大小
    pt_size = os.path.getsize("best.pt") / (1024 * 1024) # MB
    onnx_size = os.path.getsize("best.onnx") / (1024 * 1024)
    
    # 3. 测速
    print("正在测试 .pt 模型...")
    pt_time, pt_results = measure_time(pt_model, source, is_video=True)
    
    print("正在测试 .onnx 模型...")
    onnx_time, onnx_results = measure_time(onnx_model, source, is_video=True)

    # 4. 打印对比结果
    print("\n" + "="*30)
    print("模型性能对比结果：")
    print(f"原始 .pt  文件大小: {pt_size:.2f} MB | 平均推理耗时: {pt_time:.2f} ms")
    print(f"导出 .onnx文件大小: {onnx_size:.2f} MB | 平均推理耗时: {onnx_time:.2f} ms")
    print("="*30)

    # 5. 保存推理结果截图用于 README
    # 读取一帧视频进行推理并保存图片
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    if ret:
        pt_res_img = pt_results[0].plot() if pt_results else pt_model(frame)[0].plot()
        onnx_res_img = onnx_results[0].plot() if onnx_results else onnx_model(frame)[0].plot()
        cv2.imwrite("result_pt.png", pt_res_img)
        cv2.imwrite("result_onnx.png", onnx_res_img)
        print("对比结果截图已保存为 result_pt.png 和 result_onnx.png")

if __name__ == '__main__':
    main()
