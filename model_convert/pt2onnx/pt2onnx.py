from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
success = model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=13
)

print("\n" + "=" * 60)
print(f"模型导出成功! 保存路径: {success}")
print("=" * 60)