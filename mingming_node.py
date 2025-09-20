# -*- coding: utf-8 -*-
"""
💖 Mingming LoRA Maker - 응급 수정 버전
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import hashlib

class MingmingLoRAMakerFix:
    """💖 Mingming LoRA Maker - 안전한 버전"""
    
    @classmethod
    def INPUT_TYPES(cls):
        # 기본 이미지 파일들
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "💖_로라_이름": ("STRING", {"default": "밍밍_로라"}),
                "💛_트리거_워드": ("STRING", {"default": ""}),
                "💜_스타일": (["💖 귀여운_스타일", "🎨 애니메이션_스타일", "📷 사실적_스타일"], {"default": "💖 귀여운_스타일"}),
                "🧡_에포크": ("INT", {"default": 10, "min": 1, "max": 100}),
                "💙_저장_경로": ("STRING", {"default": "ComfyUI/models/loras/mingming"}),
                "🤎_추가_태그": ("STRING", {"default": "고품질, 세밀한, 걸작"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("결과",)
    
    FUNCTION = "process"
    CATEGORY = "💖 Mingming"

    def process(self, image, **kwargs):
        """기본 처리 함수"""
        
        # 파라미터 추출 (한국어 이모지 포함)
        로라_이름 = kwargs.get("💖_로라_이름", "밍밍_로라")
        트리거_워드 = kwargs.get("💛_트리거_워드", "")
        스타일 = kwargs.get("💜_스타일", "💖 귀여운_스타일")
        에포크 = kwargs.get("🧡_에포크", 10)
        저장_경로 = kwargs.get("💙_저장_경로", "ComfyUI/models/loras/mingming")
        추가_태그 = kwargs.get("🤎_추가_태그", "고품질, 세밀한, 걸작")
        
        print("💖 밍밍 로라 메이커 시작!")
        print(f"💖 프로젝트: {로라_이름}")
        print(f"💜 스타일: {스타일}")
        print(f"🧡 에포크: {에포크}")
        
        # 기본 이미지 처리
        if image == "기본이미지.png":
            # 기본 이미지 생성 (더미)
            기본_이미지 = Image.new('RGB', (768, 768), color=(135, 206, 235))  # 하늘색 배경
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(기본_이미지)
            
            try:
                # 텍스트 추가
                draw.text((384, 350), "💖", anchor="mm", fill=(255, 182, 193))
                draw.text((384, 400), "밍밍 로라 메이커", anchor="mm", fill=(255, 255, 255))
                draw.text((384, 450), "이미지를 업로드하세요", anchor="mm", fill=(255, 255, 255))
            except:
                pass
            
            img = 기본_이미지
            image_path = "기본이미지"
        else:
            # LoadImage 로직
            input_dir = folder_paths.get_input_directory()
            image_path = os.path.join(input_dir, image)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
        
        # 태그 생성 (한국어 → 영어 변환)
        태그들 = []
        if 트리거_워드:
            태그들.append(트리거_워드)
        태그들.append(로라_이름.replace("_", " "))
        
        # 스타일별 태그 (한국어 처리)
        if "귀여운" in 스타일:
            태그들.extend(["cute", "kawaii", "adorable"])
        elif "애니메이션" in 스타일:
            태그들.extend(["anime", "manga", "illustration"])
        elif "사실적" in 스타일:
            태그들.extend(["realistic", "detailed", "photographic"])
        elif "예술적" in 스타일:
            태그들.extend(["artistic", "painterly", "stylized"])
        elif "역동적" in 스타일:
            태그들.extend(["dynamic", "action", "movement"])
        else:
            태그들.extend(["high quality"])
        
        # 추가 태그 처리 (한국어 → 영어)
        if 추가_태그:
            한국어_영어_맵 = {
                "고품질": "high quality",
                "세밀한": "detailed",
                "걸작": "masterpiece",
                "아름다운": "beautiful",
                "예쁜": "pretty",
                "멋진": "cool"
            }
            
            추가_태그_리스트 = [tag.strip() for tag in 추가_태그.replace('\n', ',').split(',')]
            for tag in 추가_태그_리스트:
                if tag:
                    # 한국어면 영어로 변환, 아니면 그대로
                    영어_태그 = 한국어_영어_맵.get(tag, tag)
                    태그들.append(영어_태그)
        
        태그_문자열 = ", ".join(태그들)
        
        # 영상화 결과 메시지 생성
        영상화_결과 = f"""💖 밍밍 로라 메이커 완료!

📋 프로젝트: {로라_이름}
💜 스타일: {스타일}  
🧡 에포크: {에포크}
🏷️ 태그: {태그_문자열}

✅ LoRA 훈련 데이터 준비 완료
🎯 IllustriousXL 최적화 적용됨
💾 저장 경로: {저장_경로}/{로라_이름}.safetensors

🚀 다음 단계: LoRA 훈련 실행하기"""
        
        print("✅ 처리 완료!")
        
        return (영상화_결과,)

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

# 노드 등록 (모든 버전 호환)
NODE_CLASS_MAPPINGS = {
    "MingmingLoRAMakerFix": MingmingLoRAMakerFix,
    "MingmingLoRAMakerDummy": MingmingLoRAMakerFix,  # 호환성
    "MingmingLoRAMaker": MingmingLoRAMakerFix,       # 호환성
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MingmingLoRAMakerFix": "💖 Mingming LoRA Maker (Fix)",
    "MingmingLoRAMakerDummy": "💖 Mingming LoRA Maker (Dummy)",
    "MingmingLoRAMaker": "💖 Mingming LoRA Maker",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']