import argparse
import pyro
import sys
import torch.nn.functional
import torchmetrics

from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.nn import PyroModule, PyroParam

from src.utils import mnist
from src.models.model import BNN

def train(train_loader, model, svi, n_epochs, kl_factor, test_loader):
    running_loss = 0.0
    eval_every = 100
    steps = 0
    counter = 0

    for epoch in range(n_epochs):
        for images, labels in train_loader:
            steps += 1

            labels = torch.nn.functional.one_hot(labels, num_classes=10)
            loss = svi.step(images, labels)
            running_loss += loss

            if steps % eval_every == 0:
                model.eval()
                test_acc = test(test_loader, model, guide)
                print(f"[Epoch {epoch}] Loss: {running_loss/eval_every:.5} Test accuracy: {test_acc:.4} ")
                checkpoint = {
                    'hidden_size' : model.hidden_size,
                    'n_classes' : 10,
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, 'models/checkpoint'+str(counter)+'.pth')
                counter += 1
                running_loss = 0
                model.train()




def test(test_loader, model, guide):
    metric = torchmetrics.Accuracy()
    predictive_dist = Predictive(model, guide=guide, num_samples=100, return_sites=['_RETURN'])
    counter = 0
    for images, labels in test_loader:
        test_samples = predictive_dist(images)
        acc = metric(torch.argmax((test_samples['_RETURN']).mean(dim=0), dim=-1), labels)
        counter += 1
        if counter == 10:
            break
    acc = metric.compute()
    return acc



if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    args = parser.parse_args(sys.argv[2:])

    # Load data
    train_loader, test_loader = mnist()

    # Prepare inference
    model = BNN(hidden_size=args.hidden_size, n_classes=10)
    guide = model.guide
    optimizer = pyro.optim.Adam({"lr": args.lr})
    elbo = TraceMeanField_ELBO()
    kl_factor = train_loader.batch_size / len(train_loader)
    svi = SVI(model.forward, guide, optimizer, elbo)

    train(train_loader, model, svi, args.n_epochs, kl_factor, test_loader)
