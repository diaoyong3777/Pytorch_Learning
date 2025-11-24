from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("test_logs")

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()