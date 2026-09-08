import torch
from app.unet import Unet
from app.data import preprocess_image


class Model:
    def __init__(self, weights_path: str):
        ck = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net = Unet(base=ck.get("base", 16))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.val_dice = ck.get("val_dice")
        # Warm-up. The first forward pass is measurably slower than the rest;
        # pay it here so the first real user does not.
        with torch.no_grad():
            self.net(torch.zeros(1, ck["in_size"], ck["in_size"]))

    @torch.no_grad()
    def predict(self, image_path: str, thr: float = 0.5) -> dict:
        x = preprocess_image(image_path)                 # (1, 684, 684)
        prob = torch.sigmoid(self.net(x)[:1].float())[0] # (500, 500)
        return {"lung_fraction": float((prob > thr).float().mean().item())}
