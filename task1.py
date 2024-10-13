import tkinter as tk
import matplotlib.pyplot as plt
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

def plot_signal(indices, values, label):
    # Plot the signal
    plt.plot(indices, values, marker='o', linestyle='-', label=label)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.title(label)
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
def on_signal1_button_click():
    global signal1
    signal1 = read_signals('Signal1.txt')
def on_displaysignal1_button_click():
    plot_signal(signal1[0], signal1[1], 'Signal 1')

def on_signal2_button_click():
    global signal2
    signal2 = read_signals('Signal2.txt')

def on_displaysignal2_button_click():
    plot_signal(signal2[0], signal2[1], 'Signal 2')

def on_add_signals_button_click():
    # Add the two signals
    indices, added_values = add_signals(signal1, signal2)
    # Plot the result
    plot_signal(indices, added_values, 'Signal 1 + Signal 2')

root = tk.Tk()
root.title("DSP")
root.geometry("1550x1550")  # Set default size for the window

# Create a label for the title
title_label = tk.Label(root, text="DSP-Task 1", font=("Helvetica", 16))
title_label.pack(pady=10)  # Add padding around the label

frame = tk.Frame(root)
frame.pack(pady=20)

signal1_button = tk.Button(frame, text="Read Signal 1", command=on_signal1_button_click, width=20, height=2)
signal1_button.pack(pady=5)

displaysignal1_button = tk.Button(frame, text="Display Signal 1", command=on_displaysignal1_button_click, width=20, height=2)
displaysignal1_button.pack(pady=5)

signal2_button = tk.Button(frame, text="Read Signal 2", command=on_signal2_button_click, width=20, height=2)
signal2_button.pack(pady=5)

displaysignal2_button = tk.Button(frame, text="Display Signal 2", command=on_displaysignal2_button_click, width=20, height=2)
displaysignal2_button.pack(pady=5)

add_signals_button = tk.Button(frame, text="Add Signals", command=on_add_signals_button_click, width=20, height=2)
add_signals_button.pack(pady=5)

root.mainloop()


