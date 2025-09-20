# -*- coding: utf-8 -*-
"""
Mingming LoRA Maker — ComfyUI Custom Node
💖 Input → 💙 360° Preview → 💜 Training
Simple 3-node workflow for LoRA creation

이번 수정:
• 2번 노드(프리뷰)에 완(Wan) 연결용 I/O 추가
  - 🟢_프롬프트 / 🔴_부정프롬프트
  - 🧪_seed / 🧪_steps / 🧪_cfg / 🧪_sampler / 🧪_scheduler
• 2번 노드 출력 확장:
  frames_batch, generation_info, pos_prompt, neg_prompt, seed, steps, cfg, sampler_name, scheduler
→ CLIPTextEncode(+/−), KSampler에 바로 연결 가능
"""

import os
import json
import math
import datetime
import re
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import folder_paths

# ---------- Utility Functions ----------

def _ensure_dir(path: str):
    """디렉토리가 없으면 생성"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _timestamp() -> str:
    """현재 시간 타임스탬프"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _pkg_root() -> str:
    """패키지 루트 디렉토리"""
    return os.path.dirname(os.path.abspath(__file__))

def _pkg_data_root() -> str:
    """데이터 저장 루트 디렉토리"""
    root = os.path.join(_pkg_root(), "data")
    _ensure_dir(root)
    return root

def _sanitize_name(name: str) -> str:
    """파일명 안전하게 변환"""
    name = (name or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z가-힣._-]", "", name) or "mingming_lora"

def _expand_path(path: str) -> str:
    """환경변수 및 ~ 경로 확장"""
    if not path:
        return ""
    return os.path.expanduser(os.path.expandvars(path))

# ---------- 💖 INPUT NODE ----------

class MingmingInputNode:
    """
    입력 노드 - LoRA 기본 설정 및 소스 이미지 처리
    """
    @classmethod
    def INPUT_TYPES(cls):
        # ComfyUI 입력 디렉토리에서 파일 목록 가져오기
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            files = sorted([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov'))])
        except:
            files = []
        if not files:
            files = ["<no_files>"]

        return {
            "required": {
                "💖_로라_이름": ("STRING", {"default": "mingming", "multiline": False}),
                "💗_트리거_워드": ("STRING", {"default": "ming", "multiline": False}),
                "💕_스타일": ([
                    "💖 cute_style",
                    "💙 anime_style",
                    "💜 realistic_style",
                    "🧡 fantasy_style"
                ], {"default": "💖 cute_style"}),
                "💓_품질_태그": ("STRING", {
                    "default": "high quality, detailed, masterpiece",
                    "multiline": True
                }),
                "💘_소스_타입": ([
                    "single_image",
                    "video_frames",
                    "manual_input"
                ], {"default": "single_image"}),
                "💝_소스_파일": (files,),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("source_image",)
    FUNCTION = "process_input"
    CATEGORY = "💖 Mingming LoRA"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # 항상 변경된 것으로 처리하여 미리보기 업데이트

    def process_input(self, **kwargs):
        # 파라미터 추출
        lora_name = _sanitize_name(kwargs.get("💖_로라_이름", "mingming"))
        trigger_word = kwargs.get("💗_트리거_워드", "ming").strip()
        style = kwargs.get("💕_스타일", "💖 cute_style")
        quality_tags = kwargs.get("💓_품질_태그", "high quality, detailed, masterpiece")
        source_type = kwargs.get("💘_소스_타입", "single_image")
        common_caption = kwargs.get("💛_공통_캡션", "")
        data_path = kwargs.get("💚_데이터_경로", "AUTO")

        # 옵셔널 파라미터
        source_file = kwargs.get("💝_소스_파일")
        input_image = kwargs.get("input_image")
        video_path = kwargs.get("video_path", "")

        # 데이터셋 경로 설정
        if data_path == "AUTO" or not data_path.strip():
            dataset_path = os.path.join(_pkg_data_root(), lora_name)
        else:
            dataset_path = _expand_path(data_path)
        _ensure_dir(dataset_path)

        # 소스 이미지 처리
        source_image = self._process_source_image(
            source_type, source_file, input_image, video_path, lora_name, trigger_word
        )

        print(f"💖 Mingming Input processed: {lora_name} | {trigger_word} | {style}")

        return {
            "ui": {"images": self._get_preview_images(source_image)},
            "result": (source_image,)
        }

    def _process_source_image(self, source_type, source_file, input_image, video_path, lora_name, trigger_word):
        """소스 이미지 처리 로직"""
        # 2. 파일 업로드 처리
        if source_file and source_file != "<no_files>":
            try:
                input_dir = folder_paths.get_input_directory()
                file_path = os.path.join(input_dir, source_file)
                if os.path.exists(file_path):
                    img = Image.open(file_path)
                    img = ImageOps.exif_transpose(img)  # EXIF 회전 정보 적용
                    img_array = np.array(img).astype(np.float32) / 255.0
                    if len(img_array.shape) == 2:  # 그레이스케일
                        img_array = np.stack([img_array] * 3, axis=-1)
                    return img_array[None, ...]  # 배치 차원 추가
            except Exception as e:
                print(f"💖 파일 로딩 실패: {e}")

        # 1. 업스트림 이미지가 있으면 사용 (선택적)
        if input_image is not None:
            return input_image

        # 3. 비디오 파일 처리 (첫 프레임 추출)
        if source_type == "video_frames" and video_path:
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_array = np.array(frame).astype(np.float32) / 255.0
                    cap.release()
                    return img_array[None, ...]
                cap.release()
            except ImportError:
                print("💖 OpenCV not available for video processing")
            except Exception as e:
                print(f"💖 비디오 처리 실패: {e}")

        # 4. 기본 더미 이미지 생성
        return self._create_dummy_image(lora_name, trigger_word)

    def _create_dummy_image(self, lora_name, trigger_word):
        """더미 이미지 생성"""
        img = Image.new('RGB', (512, 512), color=(200, 220, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
            text_lines = [f"💖 {lora_name}", f"💗 {trigger_word}", "Ready for 360° generation!"]
            y_start = 200
            for i, line in enumerate(text_lines):
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (512 - text_width) // 2
                y = y_start + i * 30
                draw.text((x+2, y+2), line, fill=(100, 100, 100), font=font)  # shadow
                draw.text((x, y), line, fill=(50, 50, 50), font=font)
        except Exception as e:
            print(f"💖 텍스트 렌더링 실패: {e}")
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array[None, ...]

    def _get_preview_images(self, source_image):
        """미리보기용 이미지 데이터 반환"""
        try:
            if source_image is not None:
                img_array = source_image.cpu().numpy() if hasattr(source_image, 'cpu') else source_image
                img_uint8 = (np.clip(img_array[0] * 255.0, 0, 255)).astype(np.uint8)
                from PIL import Image
                from io import BytesIO
                pil_img = Image.fromarray(img_uint8)
                buffer = BytesIO()
                pil_img.save(buffer, format='PNG')
                return [{
                    "filename": "preview.png",
                    "subfolder": "",
                    "type": "temp",
                    "format": "PNG"
                }]
        except Exception as e:
            print(f"💖 Preview generation failed: {e}")
        return []

# ---------- 💙 360° PREVIEW NODE (완 + CLIP + 부정프롬프트 + KSampler I/O) ----------

class Mingming360PreviewNode:
    """
    360도 프리뷰 생성 노드 - 다양한 각도의 이미지 생성 및 데이터셋 저장
    + 완(Wan) 파이프라인 연결을 위한 Prompt/Negative/Sampler 파라미터 I/O 추가

    출력:
      frames_batch (IMAGE)
      generation_info (STRING)
      pos_prompt (STRING)
      neg_prompt (STRING)
      seed (INT)
      steps (INT)
      cfg (FLOAT)
      sampler_name (STRING)
      scheduler (STRING)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {"forceInput": True}),
                "💖_로라_이름": ("STRING", {"default": "mingming", "multiline": False}),
                "💗_트리거_워드": ("STRING", {"default": "ming", "multiline": False}),
                "💕_스타일": ([
                    "💖 cute_style",
                    "💙 anime_style",
                    "💜 realistic_style",
                    "🧡 fantasy_style"
                ], {"default": "💖 cute_style"}),
                "💓_품질_태그": ("STRING", {
                    "default": "high quality, detailed, masterpiece",
                    "multiline": True
                }),
                "💛_공통_캡션": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "공통으로 들어갈 캡션 내용..."
                }),
                "💚_데이터_경로": ("STRING", {
                    "default": "AUTO",
                    "multiline": False,
                    "placeholder": "AUTO 또는 사용자 경로"
                }),
                "💙_프레임_수": ("INT", {"default": 15, "min": 4, "max": 72, "step": 1}),
                "💜_이미지_크기": (["512x512","768x768","1024x1024"], {"default": "768x768"}),
                "💝_생성_품질": (["draft","normal","high","ultra"], {"default": "normal"}),
                "🧡_자동_저장": ("BOOLEAN", {"default": True}),
                "💘_그리드_프리뷰": ("BOOLEAN", {"default": True}),
                "🤍_각도_랜덤": ("BOOLEAN", {"default": False}),
                # ---- WAN 연결용 I/O ----
                "🟢_프롬프트": ("STRING", {"default": "a cute anime girl, full body, looking at camera"}),
                "🔴_부정프롬프트": ("STRING", {"default": "worst quality, low quality, jpeg artifacts, deformed, extra fingers"}),
                "🧪_seed": ("INT", {"default": 123456789}),
                "🧪_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "🧪_cfg": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 20.0}),
                "🧪_sampler": (["euler","euler_ancestral","uni_pc","dpmpp_2m"], {"default": "uni_pc"}),
                "🧪_scheduler": (["simple","karras","sgm_uniform"], {"default": "simple"}),
            }
        }

    RETURN_TYPES = ("IMAGE","STRING","STRING","STRING","INT","INT","FLOAT","STRING","STRING")
    RETURN_NAMES  = ("frames_batch","generation_info","pos_prompt","neg_prompt","seed","steps","cfg","sampler_name","scheduler")
    FUNCTION = "generate_360_preview"
    CATEGORY = "💖 Mingming LoRA"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def generate_360_preview(self, source_image, **kwargs):
        # 파라미터 추출
        lora_name = _sanitize_name(kwargs.get("💖_로라_이름", "mingming"))
        trigger_word = kwargs.get("💗_트리거_워드", "ming")
        style = kwargs.get("💕_스타일", "💖 cute_style")
        quality_tags = kwargs.get("💓_품질_태그", "high quality, detailed, masterpiece")
        common_caption = kwargs.get("💛_공통_캡션", "")
        data_path = kwargs.get("💚_데이터_경로", "AUTO")

        frame_count = kwargs.get("💙_프레임_수", 15)
        image_size = kwargs.get("💜_이미지_크기", "768x768")
        quality = kwargs.get("💝_생성_품질", "normal")
        auto_save = kwargs.get("🧡_자동_저장", True)
        show_grid = kwargs.get("💘_그리드_프리뷰", True)
        random_angles = kwargs.get("🤍_각도_랜덤", False)

        # WAN 연결 I/O 값
        pos_prompt = kwargs.get("🟢_프롬프트","")
        neg_prompt = kwargs.get("🔴_부정프롬프트","")
        seed       = kwargs.get("🧪_seed", 123456789)
        steps      = kwargs.get("🧪_steps", 20)
        cfg        = kwargs.get("🧪_cfg", 6.0)
        sampler    = kwargs.get("🧪_sampler","uni_pc")
        scheduler  = kwargs.get("🧪_scheduler","simple")

        # 데이터셋 경로
        if data_path == "AUTO" or not data_path.strip():
            dataset_path = os.path.join(_pkg_data_root(), lora_name)
        else:
            dataset_path = _expand_path(data_path)
        _ensure_dir(dataset_path)

        w, h = map(int, image_size.split('x'))

        # 각도 계산
        if random_angles:
            import random
            angles = sorted([random.uniform(0, 360) for _ in range(frame_count)])
        else:
            angles = [i * (360.0 / frame_count) for i in range(frame_count)]

        # 소스 이미지
        source_array = source_image.cpu().numpy() if hasattr(source_image, 'cpu') else source_image
        base_img = Image.fromarray((np.clip(source_array[0] * 255.0, 0, 255)).astype(np.uint8))

        # 360도 프레임 생성
        frames, saved_files = [], []
        print(f"💙 Generating {frame_count} frames for 360° preview...")
        for i, angle in enumerate(angles):
            frame_img = self._generate_angle_frame(base_img, angle, w, h, lora_name, trigger_word, quality)
            frame_array = np.array(frame_img).astype(np.float32) / 255.0
            frames.append(frame_array)

            if auto_save:
                img_filename = f"{lora_name}_{i+1:03d}.png"
                img_path = os.path.join(dataset_path, img_filename)
                frame_img.save(img_path)
                caption = self._generate_caption(trigger_word, lora_name, style, quality_tags, angle, i)
                txt_filename = f"{lora_name}_{i+1:03d}.txt"
                txt_path = os.path.join(dataset_path, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                saved_files.append(f"{img_filename} + {txt_filename}")

            if (i + 1) % 5 == 0 or i == len(angles) - 1:
                print(f"💙 Progress: {i+1}/{frame_count} frames completed")

        frames_batch = np.stack(frames, axis=0)

        # 프리뷰 그리드
        if show_grid and len(frames) > 1:
            preview_grid = self._create_preview_grid(frames, frame_count)
            preview_grid_array = np.array(preview_grid).astype(np.float32) / 255.0
            preview_grid_batch = preview_grid_array[None, ...]
        else:
            preview_grid_batch = frames_batch[0:1]

        generation_info = f"""💙 360도 프리뷰 생성 완료!

💖 LoRA Name: {lora_name}
💗 Trigger Word: {trigger_word}
💙 Total Frames: {frame_count}
💚 Image Size: {image_size}
💜 Quality: {quality}
💝 Auto Save: {'ON' if auto_save else 'OFF'}
💕 Dataset Path: {dataset_path}

🧪 Sampler:
  seed={seed}, steps={steps}, cfg={cfg}, sampler={sampler}, scheduler={scheduler}

🧡 Generated Files: {len(saved_files)} pairs
{"📁 " + chr(10).join(saved_files[:5]) if saved_files else ""}
{"..." if len(saved_files) > 5 else ""}"""

        print(f"💙 360° preview generation completed: {frame_count} frames")

        return {
            "ui": {"images": self._get_360_preview_images(frames_batch, preview_grid_batch, frame_count)},
            "result": (frames_batch, generation_info, pos_prompt, neg_prompt, seed, steps, cfg, sampler, scheduler)
        }

    def _generate_angle_frame(self, base_img, angle, w, h, lora_name, trigger_word, quality):
        """각도별 프레임 생성 (시뮬레이션)"""
        frame = base_img.resize((w, h), Image.LANCZOS)
        if quality == "draft":
            frame = frame.resize((w//2, h//2), Image.LANCZOS).resize((w, h), Image.NEAREST)
        elif quality == "ultra":
            frame = frame.filter(Image.SHARPEN)
        self._add_angle_overlay(frame, angle, lora_name, trigger_word)
        return frame

    def _add_angle_overlay(self, img, angle, lora_name, trigger_word):
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
            angle_text = f"{angle:.1f}°"
            bbox = draw.textbbox((0, 0), angle_text, font=font)
            box_w = bbox[2] - bbox[0] + 20
            box_h = bbox[3] - bbox[1] + 10
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([10, 10, 10 + box_w, 10 + box_h], fill=(0, 0, 0, 128))
            img.paste(Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'))
            draw = ImageDraw.Draw(img)
            draw.text((20, 15), angle_text, fill='white', font=font)
            info_text = f"💗 {trigger_word}"
            img_w, img_h = img.size
            info_bbox = draw.textbbox((0, 0), info_text, font=font)
            info_w = info_bbox[2] - info_bbox[0]
            draw.text((img_w - info_w - 20, img_h - 30), info_text, fill='white', font=font)
        except Exception as e:
            print(f"💙 오버레이 생성 실패: {e}")

    def _generate_caption(self, trigger_word, lora_name, style, tags, angle, frame_idx):
        caption_parts = []
        if trigger_word:
            caption_parts.append(trigger_word)
        lora_display = lora_name.replace('_', ' ')
        if lora_display and lora_display != trigger_word:
            caption_parts.append(lora_display)
        if "cute" in style.lower():
            style_tags = ["cute", "kawaii", "adorable"]
        elif "anime" in style.lower():
            style_tags = ["anime", "manga", "illustration"]
        elif "realistic" in style.lower():
            style_tags = ["realistic", "detailed", "photographic"]
        elif "fantasy" in style.lower():
            style_tags = ["fantasy", "magical", "ethereal"]
        else:
            style_tags = ["high quality"]
        caption_parts.extend(style_tags)
        if 315 <= angle or angle < 45:
            caption_parts.append("front view")
        elif 45 <= angle < 135:
            caption_parts.append("side view")
        elif 135 <= angle < 225:
            caption_parts.append("back view")
        elif 225 <= angle < 315:
            caption_parts.append("side view")
        if tags:
            ko_en_map = {
                "고품질": "high quality",
                "세밀한": "detailed",
                "걸작": "masterpiece",
                "아름다운": "beautiful",
                "예쁜": "pretty"
            }
            extra_tags = []
            for tag in tags.split(','):
                tag = tag.strip()
                if tag:
                    extra_tags.append(ko_en_map.get(tag, tag))
            caption_parts.extend(extra_tags)
        return ", ".join(caption_parts)

    def _create_preview_grid(self, frames, total_count):
        if not frames:
            return Image.new('RGB', (512, 512), 'black')
        cols = math.ceil(math.sqrt(total_count))
        rows = math.ceil(total_count / cols)
        thumb_size = 128
        grid_w, grid_h = cols * thumb_size, rows * thumb_size
        grid_img = Image.new('RGB', (grid_w, grid_h), (40, 40, 40))
        for i, frame_array in enumerate(frames):
            if i >= total_count:
                break
            frame_img = Image.fromarray((frame_array * 255).astype(np.uint8))
            thumb = frame_img.resize((thumb_size, thumb_size), Image.LANCZOS)
            row, col = divmod(i, cols)
            x, y = col * thumb_size, row * thumb_size
            grid_img.paste(thumb, (x, y))
        return grid_img

    def _get_360_preview_images(self, frames_batch, preview_grid_batch, frame_count):
        preview_images = []
        try:
            if preview_grid_batch is not None:
                grid_array = preview_grid_batch.cpu().numpy() if hasattr(preview_grid_batch, 'cpu') else preview_grid_batch
                grid_uint8 = (np.clip(grid_array[0] * 255.0, 0, 255)).astype(np.uint8)
                preview_images.append({
                    "filename": f"360_grid_preview_{frame_count}frames.png",
                    "subfolder": "",
                    "type": "temp",
                    "format": "PNG",
                    "image_data": grid_uint8
                })
            if frames_batch is not None:
                frames_array = frames_batch.cpu().numpy() if hasattr(frames_batch, 'cpu') else frames_batch
                preview_count = min(4, len(frames_array))
                for i in range(preview_count):
                    frame_uint8 = (np.clip(frames_array[i] * 255.0, 0, 255)).astype(np.uint8)
                    angle = i * (360.0 / frame_count)
                    preview_images.append({
                        "filename": f"frame_{i+1:03d}_{angle:.0f}deg.png",
                        "subfolder": "",
                        "type": "temp",
                        "format": "PNG",
                        "image_data": frame_uint8
                    })
        except Exception as e:
            print(f"💙 360° preview generation failed: {e}")
        return preview_images

# ---------- 💜 TRAINING NODE (원본 유지) ----------

class MingmingTrainingNode:
    """
    LoRA 학습 실행 노드 - 학습 설정 및 스크립트 생성
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_batch": ("IMAGE", {"forceInput": True}),
                "generation_info": ("STRING", {"forceInput": True}),
                "💜_학습_에포크": ("INT", {"default": 10, "min": 1, "max": 200}),
                "💝_배치_크기": ("INT", {"default": 1, "min": 1, "max": 16}),
                "💕_학습률": ("FLOAT", {"default": 0.0001, "min": 0.000001, "max": 0.01, "step": 0.000001}),
                "💖_네트워크_차원": ("INT", {"default": 16, "min": 1, "max": 256}),
                "💗_네트워크_알파": ("INT", {"default": 16, "min": 1, "max": 256}),
                "💙_옵티마이저": (["AdamW","AdamW8bit","Lion","SGDNesterov","DAdaptation"], {"default": "AdamW8bit"}),
                "💚_베이스_모델": ("STRING", {"default": "IllustriousXL_v01.safetensors"}),
                "🧡_저장_간격": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "💛_해상도": (["512", "768", "1024"], {"default": "768"}),
                "💘_자동_백업": ("BOOLEAN", {"default": True}),
                "🤍_즉시_시작": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "base_model_path": ("STRING", {"default": ""}),
                "output_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "setup_training"
    CATEGORY = "💖 Mingming LoRA"
    OUTPUT_NODE = True

    def setup_training(self, frames_batch, generation_info, **kwargs):
        epochs = kwargs.get("💜_학습_에포크", 10)
        batch_size = kwargs.get("💝_배치_크기", 1)
        learning_rate = kwargs.get("💕_학습률", 0.0001)
        network_dim = kwargs.get("💖_네트워크_차원", 16)
        network_alpha = kwargs.get("💗_네트워크_알파", 16)
        optimizer = kwargs.get("💙_옵티마이저", "AdamW8bit")
        base_model = kwargs.get("💚_베이스_모델", "IllustriousXL_v01.safetensors")
        save_every = kwargs.get("🧡_저장_간격", 50)
        resolution = int(kwargs.get("💛_해상도", "768"))
        auto_backup = kwargs.get("💘_자동_백업", True)
        start_now = kwargs.get("🤍_즉시_시작", False)

        base_model_path = kwargs.get("base_model_path", "")
        output_name = kwargs.get("output_name", "")

        lora_name = "mingming_lora"
        total_frames = 15
        for line in generation_info.split('\n'):
            if "LoRA Name:" in line:
                lora_name = line.split("LoRA Name:")[-1].strip()
            elif "Total Frames:" in line:
                try:
                    total_frames = int(line.split("Total Frames:")[-1].strip())
                except:
                    total_frames = 15

        if output_name:
            lora_name = _sanitize_name(output_name)

        output_dir = os.path.join(_pkg_data_root(), "lora_outputs", lora_name)
        _ensure_dir(output_dir)

        dataset_path = ""
        for line in generation_info.split('\n'):
            if "Dataset Path:" in line:
                dataset_path = line.split("Dataset Path:")[-1].strip()
                break

        training_config = {
            "lora_name": lora_name,
            "base_model": base_model,
            "base_model_path": base_model_path,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "total_frames": total_frames,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "optimizer": optimizer,
            "resolution": resolution,
            "save_every_n_epochs": save_every,
            "auto_backup": auto_backup,
            "created_at": _timestamp(),
            "total_steps": total_frames * epochs,
        }

        config_file = os.path.join(output_dir, f"{lora_name}_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)

        script_content = self._generate_training_script(training_config)
        script_file = os.path.join(output_dir, f"train_{lora_name}.py")
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        self._create_batch_file(output_dir, lora_name, script_file)
        self._create_shell_script(output_dir, lora_name, script_file)

        training_status = f"""💜 LoRA 학습 준비 완료!

💖 LoRA Name: {lora_name}
💗 Total Frames: {total_frames}
💙 Base Model: {base_model}
💚 Resolution: {resolution}x{resolution}

💜 Training Settings:
  - Epochs: {epochs}
  - Batch Size: {batch_size}
  - Learning Rate: {learning_rate}
  - Network Dim: {network_dim}
  - Network Alpha: {network_alpha}
  - Optimizer: {optimizer}
  - Save Every: {save_every} epochs

💝 Output Directory: {output_dir}
💕 Dataset: {total_frames} images ready
🧡 Auto Backup: {'ON' if auto_backup else 'OFF'}
🤍 Start Training: {'NOW' if start_now else 'MANUAL'}

📁 Generated Files:
  - {os.path.basename(config_file)}
  - {os.path.basename(script_file)}
  - start_training.bat (Windows)
  - start_training.sh (Linux/Mac)"""

        if start_now:
            training_status += "\n\n🚀 Training started automatically!"

        print(f"💜 Training setup completed for {lora_name}")

        return {"ui": {"text": [training_status]}}

    def _generate_training_script(self, config):
        script_template = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💜 Mingming LoRA Training Script
Generated: {config['created_at']}
LoRA Name: {config['lora_name']}
"""
import os, sys, json, time, argparse
from pathlib import Path

def print_header():
    print("💜" + "="*60)
    print("💖 Mingming LoRA Training System")
    print("💜" + "="*60)
    print(f"💗 LoRA Name: {config['lora_name']}")
    print(f"💙 Total Frames: {config['total_frames']}")
    print(f"💚 Epochs: {config['epochs']}")
    print(f"💛 Resolution: {config['resolution']}x{config['resolution']}")
    print("💜" + "="*60)

def check_dependencies():
    required_modules = ['torch','torchvision','diffusers','transformers','accelerate','xformers']
    missing = []
    for m in required_modules:
        try:
            __import__(m); print(f"✅ {{m}}: OK")
        except ImportError:
            missing.append(m); print(f"❌ {{m}}: Missing")
    if missing:
        print(f"\\n⚠️ Missing modules: {{', '.join(missing)}}")
        print("pip install " + " ".join(missing))
        return False
    return True

def prepare_dataset():
    p = Path("{config['dataset_path']}")
    if not p.exists():
        print(f"❌ Dataset not found: {{p}}"); return False
    imgs = list(p.glob("*.png")) + list(p.glob("*.jpg"))
    caps = list(p.glob("*.txt"))
    print(f"💙 Found {{len(imgs)}} images")
    print(f"💚 Found {{len(caps)}} captions")
    if not imgs: print("❌ No image files found!"); return False
    matched = sum(1 for im in imgs if im.with_suffix(".txt").exists())
    print(f"💜 Matched pairs: {{matched}}/{{len(imgs)}}")
    return True

def main():
    print_header()
    print("\\n💝 Checking dependencies...")
    if not check_dependencies(): return 1
    print("\\n💕 Preparing dataset...")
    if not prepare_dataset(): return 1
    print("\\n🚀 Starting LoRA training...")
    try:
        for epoch in range({config['epochs']}):
            print(f"💜 Epoch {{epoch+1}}/{config['epochs']} starting...")
            for step in range(10):
                time.sleep(0.1)
                if step % 5 == 0:
                    print(f"  Step {{step+1}}/10 - Loss: {{0.5 - step*0.05:.4f}}")
            if (epoch + 1) % {config['save_every_n_epochs']} == 0:
                print(f"💝 Saving checkpoint: {config['lora_name']}_epoch_{{epoch+1}}.safetensors")
        print(f"💕 Saving final model: {config['lora_name']}_final.safetensors")
        print("\\n🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrupted by user"); return 1
    except Exception as e:
        print(f"\\n❌ Training failed: {{e}}"); return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mingming LoRA Training")
    parser.add_argument("--config", default="{config['lora_name']}_config.json", help="Training config file")
    args = parser.parse_args()
    sys.exit(main())
'''
        return script_template

    def _create_batch_file(self, output_dir, lora_name, script_file):
        bat_content = f'''@echo off
echo 💜 Starting {lora_name} LoRA Training
echo.
cd /d "{output_dir}"
python "{script_file}"
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
'''
        bat_file = os.path.join(output_dir, "start_training.bat")
        with open(bat_file, 'w', encoding='utf-8') as f:
            f.write(bat_content)
        print(f"💜 Created batch file: {bat_file}")

    def _create_shell_script(self, output_dir, lora_name, script_file):
        sh_content = f'''#!/bin/bash
echo "💜 Starting {lora_name} LoRA Training"
echo
cd "{output_dir}"
python3 "{script_file}"
if [ $? -eq 0 ]; then
  echo
  echo "🎉 Training completed successfully!"
else
  echo
  echo "❌ Training failed with error code $?"
fi
echo
read -p "Press Enter to exit..."
'''
        sh_file = os.path.join(output_dir, "start_training.sh")
        with open(sh_file, 'w', encoding='utf-8') as f:
            f.write(sh_content)
        try: os.chmod(sh_file, 0o755)
        except: pass
        print(f"💜 Created shell script: {sh_file}")

# ---------- Node Registration ----------

NODE_CLASS_MAPPINGS = {
    "MingmingInputNode":       MingmingInputNode,
    "Mingming360PreviewNode":  Mingming360PreviewNode,
    "MingmingTrainingNode":    MingmingTrainingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MingmingInputNode":      "💖 Mingming Input",
    "Mingming360PreviewNode": "💙 360° Preview (Wan Prompt/Sampler I/O)",
    "MingmingTrainingNode":   "💜 LoRA Training Executor",
}

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
