import torch


def train(cnn, num_epochs, device, train_loader, val_loader, optimizer, criterion, patience=10, save_path='best_model_11101024.pth'):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    lossHistory = []

    cnn.to(device)
    cnn.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for index, (data, target) in enumerate(train_loader):
            # 데이터의 형태: (batch_size, 4, max_length)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()

            # 그라디언트 클리핑 (필요 시 활성화)
            # torch.nn.utils.clip_grad_norm_(position_classifier.parameters(), max_norm=0.05)

            optimizer.step()

            lossHistory.append(loss.item())
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')

        # 검증 단계
        cnn.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = cnn(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / len(val_loader.dataset)
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        cnn.train()

        # Early Stopping 및 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(cnn.state_dict(), save_path)
            print(f'-- 새로운 최적 모델 저장: Epoch {epoch + 1}')
        else:
            epochs_no_improve += 1
            print(f'-- 성능 향상 없음 ({epochs_no_improve}/{patience})')

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered. 학습을 중단합니다. 최고 성능 Epoch: {best_epoch}')
            break

    print(f'최종 최고 검증 손실: {best_val_loss:.4f} (Epoch {best_epoch})')
    return lossHistory
