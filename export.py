from ultralytics import YOLO

def main():
    # 加载原始 .pt 模型
    model = YOLO("best.pt")
    
    # 导出为 ONNX 格式，设置输入尺寸为 640 (保持一致)
    print("开始导出 ONNX 模型...")
    model.export(format="onnx", imgsz=640, simplify=True)
    print("导出完成！")

if __name__ == '__main__':
    main()
