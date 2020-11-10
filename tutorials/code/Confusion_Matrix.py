"""## Confusion Matrix"""

model_name = args.model_name
PATH = '/content/drive/My Drive/resnet101_2_75.pth'
checkpoint = torch.load(PATH)
# classes = ('0','1') 
model = make_model(
        model_name,
        pretrained=True,
        num_classes=2,
        input_size= None,
    )

model.load_state_dict(checkpoint['state_dict'])

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in test_loader:
        images = data['image']
        labels = data['label']
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))


for i in range(2):
    print('Correct of %5s : %2d, Incorrect of %5s : %2d' % (
        i, class_correct[i], i, 200-class_correct[i]))