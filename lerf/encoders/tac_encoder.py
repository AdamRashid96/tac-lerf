from lerf.encoders.image_encoder import BaseImageEncoder, BaseImageEncoderConfig

@dataclass
class TacNetworkConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: TacNetwork)
    tac_dim: int = 128
    model_dir: str = "/home/ravenhuang/tac_vis/tac_vision/output/contrastive238607/models/epoch=449-step=2250.ckpt"

class TacNetwork(BaseImageEncoder):
    def __init__(self, config: TacNetworkConfig):
        self.config = config
        self.model = ContrastiveModule.load_from_checkpoint(self.config.model_dir).eval().cuda()
