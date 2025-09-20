#!/bin/bash
echo "💜 Starting mingming LoRA Training"
echo
cd "E:\comfy-image\ComfyUI\custom_nodes\ComfyUI-mingming-loramaker\data\lora_outputs\mingming"
python3 "E:\comfy-image\ComfyUI\custom_nodes\ComfyUI-mingming-loramaker\data\lora_outputs\mingming\train_mingming.py"
if [ $? -eq 0 ]; then
  echo
  echo "🎉 Training completed successfully!"
else
  echo
  echo "❌ Training failed with error code $?"
fi
echo
read -p "Press Enter to exit..."
