import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
import TESTfunctions as test
from PIL import Image, ImageTk
def read_signals(file_path):
    indices = []
    values = []
    # Open the specified file and read its content
    with open(file_path, 'r') as file:
        file.readline()  # Skip first line
        file.readline()  # Skip second line
        N = int(file.readline())
        # Read the sample indices and values
        for _ in range(N):
            line = file.readline().strip()
            parts = line.split()
            index = int(parts[0])
            value = int(parts[1])
            indices.append(index)
            values.append(value)
    return indices, values



def plot_signal(og_indices, og_values, res_indices, res_values, label):
    # Plot the signal
    if (og_values and og_indices):
        plt.plot(og_indices, og_values, marker='o', linestyle='-', label = "Original Signal")
    if (res_values and res_indices):
        plt.plot(res_indices, res_values, marker='x', linestyle='--',label=label)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.title("Original and " + label)
    plt.show()
def add_signals(signal1, signal2):
    # Extract indices and values from the signal lists
    signal1_indices, signal1_values = signal1
    signal2_indices, signal2_values = signal2

    # Create a set of all unique indices from both signals
    all_indices = sorted(set(signal1_indices) | set(signal2_indices))

    addition_result = [0] * len(all_indices)

    for i, index in enumerate(all_indices):
        value1 = signal1_values[signal1_indices.index(index)] if index in signal1_indices else 0
        value2 = signal2_values[signal2_indices.index(index)] if index in signal2_indices else 0
        addition_result[i] = value1 + value2
    return all_indices, addition_result





def sub_signals(signal1, signal2):

    values2 = signal2[1]
    values2 = [-value for value in values2]
    #signal2 = (signal2[0], values2)

    indices, values = add_signals(signal1, (signal2[0], values2))
    return indices , values



def delay_advancing_signals(signal1, constant):
    indices = signal1[0]
    indices = [index - constant for index in indices]
    #signal1 = (indices, signal1[1])
    return indices, signal1[1]


def folding_signals(signal1):
    indices, values = signal1

    folded_indices = [-i for i in indices]
    folded_values = values[:]  # Creates a copy of the values list

    folded_indices.sort()
    folded_values.reverse()

    return folded_indices, folded_values



def multiply_signal(signal1):
    indices, values = signal1
    multiplication_result = []
    constant = simpledialog.askfloat("Input", "Enter the constant value to multiply the signal:")
    for value in values:
        multiplication_result.append(value * constant)
    test.MultiplySignalByConst(5,indices,multiplication_result)
    # plt.plot(indices, values, marker='o', linestyle='-', label="Original Signal")
    # plt.plot(indices, multiplication_result, marker='x', linestyle='--',label="Multiplied Signal")
    # plt.xlabel("Sample Index")
    # plt.ylabel("Amplitude")
    # plt.grid(True)
    # plt.legend()
    # plt.title(f"Original and Multiplied Signal")
    # plt.show()
    plot_signal(signal1[0], signal1[1], indices, multiplication_result, "Multiplied Signal")
    return indices,multiplication_result

def on_signal1_button_click():
    global signal1
    signal1 = read_signals('Signal1.txt')
def on_displaysignal1_button_click():
    plot_signal(signal1[0], signal1[1], 0, 0, 'Signal 1')

def on_signal2_button_click():
    global signal2
    signal2 = read_signals('Signal2.txt')

def on_displaysignal2_button_click():
    plot_signal(signal2[0], signal2[1], 0, 0, 'Signal 2')

def on_add_signals_button_click():
    # Add the two signals
    indices, added_values = add_signals(signal1, signal2)
    test.AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt",indices, added_values)

    # Plot the result
    plot_signal(0, 0, indices, added_values, 'Signal 1 + Signal 2')

def on_sub_signals_button_click():
    indeces, values = sub_signals(signal1, signal2)

    test.SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indeces, values)
    plot_signal(0, 0, indeces, values, "Subtraction Result")

def on_multiply_signal1_button_click():
    multiply_signal(signal1)
def on_fold_signal1_button_click():
    indices, values = folding_signals(signal1)
    test.Folding(indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Folded Signal")
def on_delay_advancing_signal1_button_click():
    constant = simpledialog.askfloat("Input", "Enter the constant value to shift the signal:")
    indices, values = delay_advancing_signals(signal1, constant)
    test.ShiftSignalByConst(constant, indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Shifted Signal")


# GUI
root = tk.Tk()
root.title("DSP")
root.geometry("1550x1550")

# Load and set the background image
background_image = Image.open("bg.jpeg")
bg_image = ImageTk.PhotoImage(background_image)

# Create a label for the background image
background_label = tk.Label(root, image=bg_image)
background_label.place(relwidth=1, relheight=1)

# Create a title label
title_label = tk.Label(root, text="DSP-Task 1", font=("Helvetica", 32), bg='lightgrey')
title_label.place(relx=0.5, rely=0.1, anchor='center')  # Center the title label

# Create buttons directly in the root window with a transparent effect
signal1_button = tk.Button(root, text="Read Signal 1", command=on_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
signal1_button.place(relx=0.5, rely=0.2, anchor='center')

displaysignal1_button = tk.Button(root, text="Display Signal 1", command=on_displaysignal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
displaysignal1_button.place(relx=0.5, rely=0.27, anchor='center')

signal2_button = tk.Button(root, text="Read Signal 2", command=on_signal2_button_click, width=20, height=2, bg='lightgrey', relief='flat')
signal2_button.place(relx=0.5, rely=0.34, anchor='center')

displaysignal2_button = tk.Button(root, text="Display Signal 2", command=on_displaysignal2_button_click, width=20, height=2, bg='lightgrey', relief='flat')
displaysignal2_button.place(relx=0.5, rely=0.41, anchor='center')

add_signals_button = tk.Button(root, text="Add Signals", command=on_add_signals_button_click, width=20, height=2, bg='lightgrey', relief='flat')
add_signals_button.place(relx=0.5, rely=0.48, anchor='center')

sub_signals_button = tk.Button(root, text="Subtract Signals", command=on_sub_signals_button_click, width=20, height=2, bg='lightgrey', relief='flat')
sub_signals_button.place(relx=0.5, rely=0.55, anchor='center')

multiply_signal1_button = tk.Button(root, text="Multiply Signal", command=on_multiply_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
multiply_signal1_button.place(relx=0.5, rely=0.62, anchor='center')

fold_signal1_button = tk.Button(root, text="Fold Signal", command=on_fold_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
fold_signal1_button.place(relx=0.5, rely=0.69, anchor='center')

delay_advancing_signal1_button = tk.Button(root, text="Delay/Advance Signal", command=on_delay_advancing_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
delay_advancing_signal1_button.place(relx=0.5, rely=0.76, anchor='center')

root.mainloop()