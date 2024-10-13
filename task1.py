import tkinter as tk
import matplotlib.pyplot as plt


def read_signals(file_path):
    indices = []
    values = []
    # Open the specified file and read its content
    with open(file_path, 'r') as file:
        file.readline()  # Skip first line
        file.readline()  # Skip second line
        N = int(file.readline().strip())  # Read the number of samples
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
    # Extract values from the signal lists
    signal1_values = signal1[1]
    signal2_values = signal2[1]
    # Determine the lengths of both signals
    len1 = len(signal1_values)
    len2 = len(signal2_values)
    maxlen = max(len1, len2)

    # Pad the shorter signal with zeros
    if len1 < len2:
        signal1_values += [0] * (len2 - len1)
    elif len2 < len1:
        signal2_values += [0] * (len1 - len2)
    addition_result = []
    # Add the values of both signals
    for i in range(maxlen):
        addition_result.append(signal1_values[i] + signal2_values[i])

    # Return a combined list of indices and the summed values
    if len1>len2:
        return signal1[0], addition_result  # Return indices of signal1 and the summed values
    else:
        return signal2[0], addition_result  # Return indices of signal1 and the summed values


def on_signal1_button_click():
    global signal1
    signal1 = read_signals('Signal1.txt')
    plot_signal(signal1[0], signal1[1], 'Signal 1')

def on_signal2_button_click():
    global signal2
    signal2 = read_signals('Signal2.txt')
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

# Create a frame to hold the buttons
frame = tk.Frame(root)
frame.pack(pady=20)  # Add padding around the frame

# Create buttons for each signal with larger size
signal1_button = tk.Button(frame, text="Display Signal 1", command=on_signal1_button_click, width=20, height=2)
signal1_button.pack(pady=5)

signal2_button = tk.Button(frame, text="Display Signal 2", command=on_signal2_button_click, width=20, height=2)
signal2_button.pack(pady=5)

# Button to add signals
add_signals_button = tk.Button(frame, text="Add Signals", command=on_add_signals_button_click, width=20, height=2)
add_signals_button.pack(pady=5)

# Start the GUI event loop
root.mainloop()