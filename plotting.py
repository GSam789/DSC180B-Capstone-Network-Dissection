import matplotlib.pyplot as plt

def plot(train_loss, train_acc, val_loss, val_acc, model_name):
    x = range(1, 1+len(train_loss))
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle(f'Model: {model_name}')
    ax1.plot(x, train_loss, label = "Training loss")
    ax1.plot(x, val_loss, label = "Test loss")

    ax1.set_title("Training & Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(x, train_acc, label = "Training acc")
    ax2.plot(x, val_acc, label = "Test acc")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    
    ax1.legend()
    ax2.legend()

    fig.tight_layout(pad = 0.5)
    plt.savefig(f"plots/{model_name}.png")
    plt.show()
    
    print(f"Plot saved on plots/{model_name}.png")