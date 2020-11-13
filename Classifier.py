import numpy as np
import torch


class Classifier:
    def __init__(self, model, n_epochs, train_loader, valid_loader, optimizer, criterion, save_path):
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path

    def fit(self):
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf

        for epoch in range(1, self.n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() / len(self.train_loader)

            if self.valid_loader:
                self.model.eval()
                for batch_idx, (data, target) in enumerate(self.valid_loader):
                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    # update the average validation loss
                    with torch.no_grad():
                        output = self.model(data)
                        loss = self.criterion(output.squeeze(), target.float())
                        valid_loss += loss.item() / len(self.valid_loader)

                if valid_loss < valid_loss_min:
                    torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')
                    valid_loss_min = valid_loss

            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        return self.model

    def evaluate(self):
        state_dict = torch.load(f'{self.save_path}/model.pt')
        self.model.load_state_dict(state_dict)

        TP = 0
        FP = 0
        FN = 0

        self.model.eval()
        for data, target in self.valid_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(data)
                output = output.squeeze()
                y_pred = output > .5
                TP += ((target == 1) & (y_pred == 1)).sum().item()
                FN += ((target == 1) & (y_pred == 0)).sum().item()
                FP += ((target == 0) & (y_pred == 1)).sum().item()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')

