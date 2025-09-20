@echo off
echo 💜 Starting mingming LoRA Training
echo.
cd /d "E:\comfy-image\ComfyUI\custom_nodes\ComfyUI-mingming-loramaker\data\lora_outputs\mingming"
python "E:\comfy-image\ComfyUI\custom_nodes\ComfyUI-mingming-loramaker\data\lora_outputs\mingming\train_mingming.py"
if %ERRORLEVEL% EQU 0 (
  echo.
  echo 🎉 Training completed successfully!
) else (
  echo.
  echo ❌ Training failed with error code %ERRORLEVEL%
)
echo.
echo Press any key to exit...
pause >nul
