from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler


device = "cuda"
MODEL_PATH = "./models/YOUR_MODEL_PATH"

# Stable Diffusion v1.5 モデルを使用する場合、enable_SDXL=Falseに設定
def make_pipe(model_path=MODEL_PATH, enable_SDXL=True):
    try:
        if enable_SDXL:
            pipe = StableDiffusionXLPipeline.from_single_file(model_path).to(device)
        else:
            pipe = StableDiffusionPipeline.from_single_file(model_path).to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()
        return pipe
    except Exception as e:
        error_message = f"モデルファイルのロードに失敗しました: {str(e)}"
        print(error_message)
        raise RuntimeError(error_message)
