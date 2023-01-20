import torch
import torchvision
import numpy as np

def predict_custom_images(model: torch.nn.Module,
                          data: torchvision.datasets,
                          classes: list,
                          images: list,
                          device: torch.device) -> None:
    data_list = []

    for sample in list(data):
        if len(data_list) < 10:
            data_list.append(sample)

    pred_probs = make_predictions(
        model=model,
        data=data_list,
        device=device
    )

    nrows = len(data_list)
    ncols = 1

    pred_classes = pred_probs.argmax(dim=1)


    for i, sample in enumerate(data_list):
        image = sample.squeeze().permute(1, 2, 0)

        list_best_preds = []
        list_best_args = []
        pred_list = pred_probs.tolist()[i]

        if i <= len(data_list):
            for j in range(3):
                list_best_preds.append(max(pred_list))
                list_best_args.append(np.argmax(pred_list))
                pred_list[np.argmax(pred_list)] = 0

        pred_label = classes[pred_classes[i]]

        title_text = f"Pred: {pred_label}"
        score_1 = f"{classes[list_best_args[0]]} = {list_best_preds[0]*100:.2f}%, "
        score_2 = f"{classes[list_best_args[1]]} = {list_best_preds[1]*100:.2f}%, "

        if len(classes) > 2:
            score_3 = f"{classes[list_best_args[2]]} = {list_best_preds[2]*100:.2f}%"
        else:
            score_3 = ''

        text = images[i] + '  ==>  ' + score_1 + score_2 + score_3

        print(text)
    print()

def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device) -> torch.Tensor:

    pred_probs = []
    model.eval()

    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)