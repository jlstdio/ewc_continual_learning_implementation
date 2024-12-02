import os
import numpy as np
import torch


def test(cnn, device, test_loader, criterion):
    cnn.eval()
    test_loss = 0.0
    correct = 0

    all_preds = []
    all_targets = []
    all_confidences = []  # 신뢰도 저장

    # 먼저 전체 평가를 수행 (torch.no_grad())
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = cnn(data)
            loss = criterion(output, target)
            # 출력 로그잇을 확률로 변환
            probabilities = torch.softmax(output, dim=1)
            test_loss += loss.item()  # 배치 손실 합산
            pred = output.argmax(dim=1, keepdim=True)  # 최대 확률의 인덱스 추출
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(probabilities.cpu().numpy())

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / len(test_loader), correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_targets, all_preds, all_confidences
