import torch


def main():
    ckpt = torch.load(input("Enter checkpoint path: "))
    print(ckpt["hyper_parameters"])


if __name__ == '__main__':
    main()
