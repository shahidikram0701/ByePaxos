import numpy as np
import matplotlib.pyplot as plt
f = open("data20_01.txt")
epochs = np.array(list(range(20)))

mselosses = [np.zeros(20) for i in range(4)]
totallosses = [np.zeros(20) for i in range(4)]
accuracyes = [np.zeros(20) for i in range(4)]
total_accuracy = np.zeros(20)

for i in range(20):
    f.readline()
    for j in range(4):
        f.readline()
        losses = f.readline().split(" ")
        mselosses[j][i] = float(losses[0].strip())
        total_loss = f.readline().split(": ")[1]
        totallosses[j][i] = float(total_loss.strip())
        acc = f.readline().split(": ")[1]
        accuracyes[j][i] = float(acc.strip())
    all_acc = f.readline().split(": ")[1]
    total_accuracy[i] = float(all_acc.strip())

plt.figure()
# plt.plot(epochs, accuracyes[0], color = "blue")
# plt.plot(epochs, accuracyes[1], color = "red")
# plt.plot(epochs, accuracyes[2], color = "green")
# plt.plot(epochs, accuracyes[3], color = "orange")
plt.plot(epochs, total_accuracy)
plt.savefig("temp.png")

f.close()