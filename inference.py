import torch

def inference(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
    return output

if __name__ == '__main__':
    model = torch.load("ddcolor.pt")
    image = torch.randn(1, 3, 256, 256)
    output = inference(model, image)
    print(output.shape)