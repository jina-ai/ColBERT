import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--colbert_checkpoint', type=str, required=True, help='Path to the ColBERT checkpoint')
    parser.add_argument('--dnn_checkpoint', type=str, required=True, help='Path to the DNN checkpoint')
    args = parser.parse_args()

    dnn_checkpoint = torch.load(args.dnn_checkpoint, map_location='cpu')
    state_dict = dnn_checkpoint['model_state_dict']
    torch.save(state_dict, args.colbert_checkpoint)
