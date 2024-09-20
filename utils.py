from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler


device = "cuda"
MODEL_PATH = "./models/YOUR_MODEL_PATH"

def make_pipe(model_path = MODEL_PATH):
    
    pipe = StableDiffusionXLPipeline.from_single_file(model_path).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    
    return pipe
