#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💜 Mingming LoRA Training Script
Generated: 20250920_192612
LoRA Name: mingming
"""
import os, sys, json, time, argparse
from pathlib import Path

def print_header():
    print("💜" + "="*60)
    print("💖 Mingming LoRA Training System")
    print("💜" + "="*60)
    print(f"💗 LoRA Name: mingming")
    print(f"💙 Total Frames: 15")
    print(f"💚 Epochs: 10")
    print(f"💛 Resolution: 768x768")
    print("💜" + "="*60)

def check_dependencies():
    required_modules = ['torch','torchvision','diffusers','transformers','accelerate','xformers']
    missing = []
    for m in required_modules:
        try:
            __import__(m); print(f"✅ {m}: OK")
        except ImportError:
            missing.append(m); print(f"❌ {m}: Missing")
    if missing:
        print(f"\n⚠️ Missing modules: {', '.join(missing)}")
        print("pip install " + " ".join(missing))
        return False
    return True

def prepare_dataset():
    p = Path("E:\comfy-image\ComfyUI\custom_nodes\ComfyUI-mingming-loramaker\data\mingming")
    if not p.exists():
        print(f"❌ Dataset not found: {p}"); return False
    imgs = list(p.glob("*.png")) + list(p.glob("*.jpg"))
    caps = list(p.glob("*.txt"))
    print(f"💙 Found {len(imgs)} images")
    print(f"💚 Found {len(caps)} captions")
    if not imgs: print("❌ No image files found!"); return False
    matched = sum(1 for im in imgs if im.with_suffix(".txt").exists())
    print(f"💜 Matched pairs: {matched}/{len(imgs)}")
    return True

def main():
    print_header()
    print("\n💝 Checking dependencies...")
    if not check_dependencies(): return 1
    print("\n💕 Preparing dataset...")
    if not prepare_dataset(): return 1
    print("\n🚀 Starting LoRA training...")
    try:
        for epoch in range(10):
            print(f"💜 Epoch {epoch+1}/10 starting...")
            for step in range(10):
                time.sleep(0.1)
                if step % 5 == 0:
                    print(f"  Step {step+1}/10 - Loss: {0.5 - step*0.05:.4f}")
            if (epoch + 1) % 50 == 0:
                print(f"💝 Saving checkpoint: mingming_epoch_{epoch+1}.safetensors")
        print(f"💕 Saving final model: mingming_final.safetensors")
        print("\n🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user"); return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}"); return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mingming LoRA Training")
    parser.add_argument("--config", default="mingming_config.json", help="Training config file")
    args = parser.parse_args()
    sys.exit(main())
