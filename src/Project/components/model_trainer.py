import torch
from src.Project.logger import logging

class ModelTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            front = batch["front"].to(self.device)
            left  = batch["left"].to(self.device)
            right = batch["right"].to(self.device)
            seg   = batch["seg"].to(self.device)
            state = batch["state"].to(self.device)
            target = batch["target"].to(self.device)

            preds = self.model(front, left, right, seg, state)
            loss_dict = self.loss_fn(preds, target)
            loss = loss_dict["total"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                front = batch["front"].to(self.device)
                left  = batch["left"].to(self.device)
                right = batch["right"].to(self.device)
                seg   = batch["seg"].to(self.device)
                state = batch["state"].to(self.device)
                target = batch["target"].to(self.device)

                preds = self.model(front, left, right, seg, state)
                loss_dict = self.loss_fn(preds, target)
                loss = loss_dict["total"]

                total_loss += loss.item()

        return total_loss / len(dataloader)
