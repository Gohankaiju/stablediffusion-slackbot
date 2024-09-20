import os
import sys
import random
import glob
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from safetensors import safe_open

# 環境変数の読み込み
import env

# カスタムモジュールのインポート
from utils import make_pipe

# グローバル変数の設定
device = "cuda"
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality"

# Slackアプリの初期化
app = App(token=os.environ["SLACK_BOT_TOKEN"])

# モデルの初期化
print("モデルを初期化中...")
pipe = make_pipe()
print("モデルの初期化完了")

# ヘルパー関数

class QueueObject:
    def __init__(self, height, width, scale, steps, prompt, n_prompt="low quality"):
        self.height = height
        self.width = width
        self.scale = scale
        self.steps = steps
        self.p_prompt = prompt
        self.n_prompt = n_prompt

def t2i(pipe, generator, queue_obj):
    image = pipe(
        prompt=queue_obj.p_prompt,
        negative_prompt=queue_obj.n_prompt,
        height=queue_obj.height,
        width=queue_obj.width,
        generator=generator,
        guidance_scale=queue_obj.scale,
        num_inference_steps=queue_obj.steps,
    ).images[0]
    return image

def create_info_file(seed, prompt, n_prompt, steps, scale, filename, height, width):
    info_content = f"""
seed: {seed}
prompt: {prompt}
negative_prompt: {n_prompt}
steps: {steps}
scale: {scale}
height: {height}
width: {width}
"""
    info_filename = f"{filename}_info.txt"
    with open(info_filename, 'w') as f:
        f.write(info_content.strip())
    return info_filename

def generate(p_prompt, save_dir, create_info=True, n_prompt=None, height=1024, width=1024, scale=7, steps=25, num_images=1):
    default_n_prompt = "low quality, worst quality, missing limb, bad hands, missing fingers, extra digit, fewer digits, deformed, realism"
    queue_obj = QueueObject(
        height=height,
        width=width,
        scale=scale,
        steps=steps,
        prompt=p_prompt,
        n_prompt=n_prompt if n_prompt is not None else default_n_prompt,
    )
    seeds = []
    image_paths = [] 
    info_paths = []

    for _ in range(num_images):
        seed = random.randint(0, 99999999999999)
        seeds.append(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"画像生成を開始します (seed: {seed})")
        image = t2i(pipe, generator, queue_obj)
        print("画像を保存します")
        filename = f"{seed}"
        image_path = os.path.join(save_dir, f"{filename}.jpg")
        image.save(image_path)
        print("画像の保存が完了しました")
        image_paths.append(image_path)
        
        if create_info:
            info_path = create_info_file(seed, p_prompt, queue_obj.n_prompt, queue_obj.steps, queue_obj.scale, os.path.join(save_dir, filename), height, width)
            info_paths.append(info_path)

    return image_paths, info_paths

def cleanup_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"ファイルを削除しました: {file_path}")
        except Exception as e:
            print(f"ファイルの削除中にエラーが発生しました: {file_path}, エラー: {str(e)}")

def parse_message(message):
    parts = message.split('|')
    main_prompt = parts[0].strip()
    n_prompt = parts[1].strip() if len(parts) > 1 and parts[1].strip() else DEFAULT_NEGATIVE_PROMPT
    height = int(parts[2].strip()) if len(parts) > 2 else 1024
    width = int(parts[3].strip()) if len(parts) > 3 else 1024
    cfg_scale = float(parts[4].strip()) if len(parts) > 4 else 7.0
    steps = int(parts[5].strip()) if len(parts) > 5 else 25
    num_images = int(parts[6].strip()) if len(parts) > 6 else 1

    return main_prompt, n_prompt, height, width, cfg_scale, steps, num_images

def extract_parameters(event_text):
    # メンションを除去
    user_input = ' '.join(word for word in event_text.split() if not word.startswith('<@'))
    # @botの後ろの部分を取得
    user_input = user_input.split('@bot ')[-1].strip()
    return parse_message(user_input)

# Slackイベントハンドラ
@app.event("app_mention")
def handle_app_mention(event, say):
    text = event['text']
    prompt, n_prompt, height, width, cfg_scale, steps, num_images = extract_parameters(text)
    create_info = "create_info" in text
    
    n_prompt_message = f"ネガティブプロンプト: {n_prompt}" if n_prompt != DEFAULT_NEGATIVE_PROMPT else "ネガティブプロンプト: デフォルト"
    
    say(f"画像生成を開始します。{num_images}枚の画像を生成します。\n"
        f"プロンプト: {prompt.strip()}\n"
        f"{n_prompt_message}\n"
        f"サイズ: {width}x{height}\n"
        f"CFG Scale: {cfg_scale}\n"
        f"Steps: {steps}\n"
    )

    try:
        save_dir = "./output"

        # 画像生成
        image_paths, info_paths = generate(prompt, save_dir, create_info=create_info, n_prompt=n_prompt, 
                                           height=height, width=width, scale=cfg_scale, steps=steps, num_images=num_images)
        
        # Slackに画像をアップロード
        for image_path in image_paths:
            upload_response = app.client.files_upload_v2(
                channel=event['channel'],
                file=image_path,
                filename=os.path.basename(image_path)
            )

        # Slackに情報ファイルをアップロード（作成された場合）
        if create_info:
            for info_path in info_paths:
                info_upload = app.client.files_upload_v2(
                    channel=event['channel'],
                    file=info_path,
                    filename=os.path.basename(info_path),
                    initial_comment="画像生成情報"
                )

        say("画像生成が完了しました。")

        # ファイルのクリーンアップ / 画像を保存したい場合はこの行をコメントアウト
        cleanup_files(image_paths + info_paths)

    except Exception as e:
        say(f"エラーが発生しました: {str(e)}")

# アプリの起動
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()