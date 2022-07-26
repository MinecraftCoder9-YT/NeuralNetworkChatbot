"""
NeuralNetworkChatbot.
This makes a chatbot using NumPy.

"""

import tkinter
from tkinter.ttk import Label, Progressbar
import numpy as np
import json
import threading


root = tkinter.Tk()
root.geometry("500x500")

TRAINING_EPOCHS = 1_000_000 # Change this to adjust the neural network training loops

def sigmoid(x, deriv=False): # Sigmoid Function
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))
x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]) # Data X
y = np.array([[0, 0, 1, 1]]).T # Data Y. Rounded output should equal y


np.random.seed(1) # Set Seed (helps with debugging)



print("Please wait...")
pleasewaitlabel = Label(root, text="Please Wait...")
pleasewaitlabel.pack()
progress = Progressbar(root, orient = tkinter.HORIZONTAL,
              length = 100, mode = 'determinate')
progress.pack()
def train_thread_func():
    val = 0 
    syn0 = 2*np.random.random((3, 1)) - 1
    for i in range(TRAINING_EPOCHS):
        #    print("Epoch", i, ":")
        l0 = x
        l1 = sigmoid(np.dot(l0, syn0))

        l1_error = y - l1
        l1_delta = l1_error * sigmoid(l1, True)

        syn0 += np.dot(l0.T, l1_delta)
        if i % 10000 == 0:
            val += 1
            progress["value"] = val
            pleasewaitlabel.config(text = "Please wait... " + str(val) + "%")

        if i == TRAINING_EPOCHS - 1:
            print("Final Test Output: ", l1)
            print()
    print("Rounding Output...")
    l1_str = str(l1)
    l1_list = l1_str.split("\n")
    for i, line in enumerate(l1_list):
        if line.endswith("]]"):
            break
        line += "," # Add commas because str(l1) does not have commas
        l1_list[i] = line # Change the line to the modified line
    l1_list = "".join(l1_list)
    l1_list = json.loads(l1_list) # Change the string to a list
    print(l1_list)
    for i, num in enumerate(l1_list):
        if num[0] >= 0.99:
            num = 1 # Round the number to 1 if the network predicts the number is close to 1
        else:
            num = 0 # Round the number to 0 if the network predicts the number is close to 0
        l1_list[i] = num # Change the numbers to the modified
    print("Rounded output. Output =", l1_list)

train_thread = threading.Thread(target=train_thread_func)
  




train_thread.start()
root.mainloop()
