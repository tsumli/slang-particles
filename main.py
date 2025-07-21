from slang_particles import App, RenderType
from omegaconf import OmegaConf


def convert_to_render_type(type: str) -> RenderType:
    if type == "greedy":
        return RenderType.GREEDY
    elif type == "tiled":
        return RenderType.TILED
    else:
        print(f"fallback to greedy")
        return RenderType.GREEDY


if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    render_type = convert_to_render_type(cfg_cli.get("type", None))
    num_particles = cfg_cli.get("num_particles", 100)
    app = App(render_type, num_particles)
    app.main_loop()
