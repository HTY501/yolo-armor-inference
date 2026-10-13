import cv2
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Armor Inference")
    parser.add_argument('--model', type=str, default='best.pt', help='模型权重路径 (best.pt)')
    parser.add_argument('--source', type=str, default='armor.mp4', help='输入视频路径')
    parser.add_argument('--output', type=str, default='output_result.mp4', help='输出结果视频路径')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载模型
    model = YOLO(args.model)
    
    # 2. 读取视频
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {args.source}")
        return

    # 3. 初始化视频写入器
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 编码
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"开始处理视频: {args.source}，输出至: {args.output}...")

    # 4. 逐帧推理并写入
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results[0].plot()  # 绘制框、类别、置信度、关键点
        
        out.write(annotated_frame)  # 写入输出视频
        cv2.imshow("YOLO Armor Inference", annotated_frame)  # 显示画面
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ 处理完成！结果视频已保存至: {args.output}")

if __name__ == '__main__':
    main()
