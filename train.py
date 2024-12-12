import torch
import torch.nn.functional as F
import lightning as L
from dataset import get_dataloader
from model import FlowMatchingModel
import argparse
from probability_paths import LinearPath

class FlowMatchingModule(L.LightningModule):
    def __init__(self, model, prob_path, lr=1e-3):
        super().__init__()
        self.model = model
        self.prob_path = prob_path
        self.learning_rate = lr
        
    def forward(self, x, t):
        return self.model(x, t)
    

    def training_step(self, batch, batch_idx):
        x1 = batch
        x0 = torch.randn_like(x1, device=self.device)
        x0.requires_grad_(True)
        t = torch.rand(size=(x1.size(0), 1), device=self.device)
        t.requires_grad_(True)
        
        sample = self.prob_path.sample(x0, x1, t)
        dxt = sample.dxt
        xt = sample.xt
                
        loss = F.mse_loss(self.model(xt, t), dxt)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return {
            "optimizer": optimizer,
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_folder", type=str, default="animals10/")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--prob_path", type=str, options=["linear"], default="linear")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 
    torch.set_float32_matmul_precision("medium")
    train_loader, _, _1 = get_dataloader(args.batch_size, image_size=args.image_size)
    net = FlowMatchingModel()
    trainer = L.Trainer(max_epochs=args.n_epochs)
    if args.prob_path == "linear":
        prob_path = LinearPath()
    model = FlowMatchingModule(model = net, prob_path = prob_path)
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.ckpt)