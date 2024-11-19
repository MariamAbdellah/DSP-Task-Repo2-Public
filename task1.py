import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import CompareSignal
import QuanTest1
import math
import QuanTest1 as qt1
import QuanTest2 as qt2
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
            value = float(parts[1])   #was int & don't know if it will cause an error in other functions
            indices.append(index)
            values.append(value)
    return indices, values


def plot_signal(x1, y1, x2, y2,label, label1,label2):
    window = tk.Toplevel()
    window.geometry("350x250")

    # Add the representation combobox
    tk.Label(window, text="Representation Type:").grid(row=0, column=0, padx=10, pady=5)
    representation = ttk.Combobox(window, values=["Continuous", "Discrete"])
    representation.grid(row=0, column=1, padx=10, pady=5)
    def plot():
        representation_value = representation.get()  # Get the selected value when plotting
        if representation_value == 'Continuous':
            if (isinstance(x1, (list, np.ndarray)) and len(x1) > 0) and (isinstance(y1, (list, np.ndarray)) and len(y1) > 0):
                plt.plot(x1, y1, marker='o', linestyle='-', label=label1, color='blue',markerfacecolor='orange', markeredgecolor='orange')
            if (isinstance(x2, (list, np.ndarray)) and len(x2) > 0) and (isinstance(y2, (list, np.ndarray)) and len(y2) > 0):
                plt.plot(x2, y2, marker='o', linestyle='-', label=label2, color='red',markerfacecolor='red')
        else:
            if (isinstance(x1, (list, np.ndarray)) and len(x1) > 0) and (isinstance(y1, (list, np.ndarray)) and len(y1) > 0):
                plt.stem(x1, y1, linefmt='b-', markerfmt='bo', basefmt=' ', label=label1)
            if (isinstance(x2, (list, np.ndarray)) and len(x2) > 0) and (isinstance(y2, (list, np.ndarray)) and len(y2) > 0):
                plt.stem(x2, y2, linefmt='r-', markerfmt='ro', basefmt=' ', label=label2)

        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.title(label)
        plt.show()
    # Add a button to plot the signal
    plot_button = tk.Button(window, text="Plot", command=plot)
    plot_button.grid(row=1, column=1, padx=10, pady=10)

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
    plot_signal(signal1[0], signal1[1], indices, multiplication_result, "Multiplied Signal","Original Signal","Multiplied Signal")
    return indices,multiplication_result


def on_signal1_button_click():
    global signal1
    signal1 = read_signals('Signal1.txt')
def on_displaysignal1_button_click():
    plot_signal(signal1[0], signal1[1], 0, 0, 'Signal 1',"Signal 1","")

def on_signal2_button_click():
    global signal2
    signal2 = read_signals('Signal2.txt')

def on_displaysignal2_button_click():
    plot_signal(signal2[0], signal2[1], 0, 0, 'Signal 2'," Signal 2","")

def on_add_signals_button_click():
    # Add the two signals
    indices, added_values = add_signals(signal1, signal2)
    test.AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt",indices, added_values)

    # Plot the result
    plot_signal(0, 0, indices, added_values, 'Addition Result',"","Addition Result")

def on_sub_signals_button_click():
    indeces, values = sub_signals(signal1, signal2)

    test.SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indeces, values)
    plot_signal(0, 0, indeces, values, "Subtraction Result"," ",'Subtraction Result')

def on_multiply_signal1_button_click():
    multiply_signal(signal1)
def on_fold_signal1_button_click():
    indices, values = folding_signals(signal1)
    test.Folding(indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Folded Signal","Original Signal","Folded Signal")
def on_delay_advancing_signal1_button_click():
    constant = simpledialog.askfloat("Input", "Enter the constant value to shift the signal:")
    indices, values = delay_advancing_signals(signal1, constant)
    test.ShiftSignalByConst(constant, indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Shifted Signal","Original Signal","Shifted Signal")

def on_displaybothsignals_button_click():
    plot_signal(signal1[0], signal1[1],signal2[0], signal2[1],'Both Signals',"Signal 1",'Signal 2')

def on_generate_signal_button_click():
    window = tk.Toplevel()
    window.title("Signal Generation")
    window.geometry("450x300")

    tk.Label(window, text = "Signal Type:").grid(row = 0, column = 0, padx=10, pady=5)
    signal_type = ttk.Combobox(window, values = ["Sine", "Cosine"])
    signal_type.grid(row = 0, column = 1, padx=10, pady=5)

    tk.Label(window, text = "Amplitude (A):").grid(row = 1, column = 0, padx=10, pady=5)
    amplitude_entry = tk.Entry(window)
    amplitude_entry.grid(row = 1, column = 1, padx=10, pady=5)

    tk.Label(window, text="Phase Shift (θ):").grid(row=2, column=0, padx=10, pady=5)
    phase_shift_entry = tk.Entry(window)
    phase_shift_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Label(window, text="Analog Frequency:").grid(row=3, column=0, padx=10, pady=5)
    analog_freq_entry = tk.Entry(window)
    analog_freq_entry.grid(row=3, column=1, padx=10, pady=5)

    tk.Label(window, text="Sampling Frequency:").grid(row=4, column=0, padx=10, pady=5)
    sampling_freq_entry = tk.Entry(window)
    sampling_freq_entry.grid(row=4, column=1, padx=10, pady=5)

    def on_generate():
        signal_type_value = signal_type.get()
        amplitude_value = float(amplitude_entry.get())
        phase_shift_value = float(phase_shift_entry.get())
        analog_freq_value = float(analog_freq_entry.get())
        sampling_freq_value = float(sampling_freq_entry.get())
        signal, time = generate_signal(signal_type_value, amplitude_value, phase_shift_value, analog_freq_value, sampling_freq_value)
        plot_signal(time, signal, [], [], 'Generate Signal', "Generated Signal", "")

    tk.Button(window, text = "Generate", command = on_generate).grid(row = 6, column = 1, padx=10, pady=5)


def generate_signal(signal_type, amp, phase_shift, analog_freq, sample_freq):

#    try:

    if sample_freq < 2 * analog_freq:
        messagebox.showerror("Error!!", f"Sampling frequency should be at least {2 * analog_freq} Hz.")
        return None
        #raise ValueError("Invalid sampling frequency value! Sampling frequency should be greater than or equal 2 * analog frequency.")

    else :
        t = np.arange(0, 1, 1 / sample_freq)
        if signal_type == "Sine":
            signal = amp * np.sin(2 * np.pi * analog_freq * t + phase_shift)
        else:
            signal = amp * np.cos(2 * np.pi * analog_freq * t + phase_shift)

    return signal,t
        #
        # plt.figure()
        #
        # if representation == "Continuous":
        #     plt.plot(t, signal, label = f'{signal_type} (Continuous)')
        # else:
        #     plt.stem(t, signal, label = f'{signal_type} (Discrete)')  # , linefmt='b-', markerfmt='bo', basefmt="r-")
        #
        # plt.title(f'{signal_type} Signal')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.show()

  #  except ValueError as e:
  #      messagebox.showerror("Invalid Input", str(e))


def on_quantize_signal_button_click():
    window = tk.Toplevel()
    window.title("Signal Quantization")
    window.geometry("450x300")

    tk.Label(window, text = "Quantize Using:").grid(row = 0, column = 0, padx = 10, pady = 5)
    quant_type = ttk.Combobox(window, values = ["Levels", "Bits"])
    quant_type.grid(row = 0, column = 1, padx = 10, pady = 5)

    tk.Label(window, text = "Value:").grid(row = 1, column = 0, padx = 10, pady = 5)
    type_entry = tk.Entry(window)
    type_entry.grid(row = 1, column = 1, padx = 10, pady = 5)

    def on_quantize():
        levels = int(type_entry.get())
        s=quant_type.get()
        if quant_type.get() == "Bits":
            levels = 2 ** int(type_entry.get())
        quantize_signal(levels,s)
    tk.Button(window, text = "Generate", command = on_quantize).grid(row = 2, column = 1, padx = 10, pady = 5)
    #QuanTest1.QuantizationTest1("Quan1_input.txt", quantized_values, )

def quantize_signal(levels,s):
    if s=='Bits':
        signal=read_signals("Quan1_input.txt")
        index=signal[0]
        values =signal[1]
    else:
        signal=read_signals("Quan2_input.txt")
        index=signal[0]
        values =signal[1]

    encoded_signal=[]
    minx = min(values)
    maxx = max(values)
    #levels = simpledialog.askinteger("Input", "Enter number of levels:")
    delta = (maxx - minx) / levels

    ranges = []
    interval_indices=[]

    #if levels is not None:
    for i in range(levels):
        low = minx + i * delta
        high = minx + (i + 1) * delta
        ranges.append((low, high))

    points = []
    for sub in ranges:
        mid=(sub[0] + sub[1]) / 2
        points.append(mid)

    quantized = []

    for signal_value in values:
        closest_midpoint = None
        minimum_difference = float('inf')
        for midpoint in points:
            difference = abs(midpoint - signal_value)
            if difference < minimum_difference:
                minimum_difference = difference
                closest_midpoint = midpoint
        quantized.append(closest_midpoint)


    quantization_error = np.array(quantized) - np.array(values)
    avg_power_error = np.mean(quantization_error ** 2)
    bits = math.ceil(math.log2(levels))

    for closest_midpoint in quantized:
        index = points.index(closest_midpoint)
        binary_encoded = format(index, f'0{bits}b')
        encoded_signal.append(binary_encoded)

    for encoded in encoded_signal:
        interval_index = int(encoded, 2) + 1  # Convert binary to integer and add 1
        interval_indices.append(interval_index)
        # Display
    plt.figure(figsize=(12, 8))

    # Original and Quantized Signal
    plt.subplot(3, 1, 1)
    plt.plot(signal[0], signal[1], 'o-', label="Original Signal", color='blue')
    plt.step(signal[0], quantized, label="Quantized Signal", color='orange', where='mid', marker='o')
    plt.title("Original and Quantized Signal")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.xticks(signal[0])

    # Quantization Error
    plt.subplot(3, 1, 2)
    plt.plot(signal[0], quantization_error, 'ro-', label="Quantization Error")
    plt.title("Quantization Error")
    plt.xlabel("Index")
    plt.ylabel("Error")
    plt.grid(True)
    plt.xticks(signal[0])

    # Binary Encoding text
    binary_text = "Index   Binary Encoding\n" + "\n".join(
        [f"{i:<5}: {encoded_signal[i]}" for i in range(levels)])
    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.text(0.5, 0.5, binary_text, ha='center', va='center', fontsize=10, color="black", family="monospace", wrap=True)
    plt.title("Encoding of Quantized Signal")

    # Add average power error to the plot
    plt.figtext(0.05, 0.2, f"Average Power Error: {avg_power_error:.4f}", ha='left', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()
    if s=='Bits':
        qt1.QuantizationTest1('Quan1_Out.txt',encoded_signal,quantized)
    else:
        qt2.QuantizationTest2('Quan2_Out.txt', interval_indices, encoded_signal, quantized, quantization_error)



def on_compute_average_button_click():
    window = tk.Toplevel()
    window.title("Compute Signal Average")
    window.geometry("450x300")
    tk.Label(window, text="Window Size", fg="#003366", font = ("Helvetica", 10)).grid(row=0, column=0, padx=10, pady=5)
    window_size = tk.Entry(window)
    window_size.grid(row=0, column=1, padx=10, pady=5)
    def on_average():
        windows = int(window_size.get())
        average_signal(windows)
    tk.Button(window, text='Average', fg="#003366", font = ("Helvetica", 10), command=on_average).grid(row=1, column=1, padx=15, pady=5)


def average_signal(window_size):
    indices, values = read_signals("Moving Average testcases/MovingAvg_input.txt")
    new_indicis = []
    new_values = []
    new_index = 0
    for i in range(len(indices) - window_size + 1):
        sum = 0
        for j in range(i, i+window_size):
            if j < len(values):
                sum += values[j]
                #print(j, "\t", values[j])
        avg = round(sum/window_size, 2)
        new_indicis.append(new_index)
        new_values.append(avg)
        new_index += 1

    print(new_indicis)
    print(new_values)

    if window_size == 3:
        CompareSignal.CompareSignal('Moving Average testcases/MovingAvg_out1.txt',new_indicis ,new_values)

    else:
        CompareSignal.CompareSignal('Moving Average testcases/MovingAvg_out2.txt',new_indicis ,new_values)

    plot_signal(new_indicis, new_values, 0, 0, 'Average Signal', '', '')

def Conv_Signals():
    # y(n)=x(n)*h(x)
    # y(n)=sum [x(k)*h(n-k)]
    indices1, values1=read_signals('Convolution testcases/Signal 1.txt')
    indices2, values2=read_signals('Convolution testcases/Signal 2.txt')

    result_indices = []
    result_values = []

    min_index = indices1[0] + indices2[0]
    max_index = indices1[-1] + indices2[-1]

    for n in range(min_index, max_index + 1):
        conv_sum = 0
        for k in range(len(values1)):
            index_k = indices1[k]
            index_h = n - index_k
            if index_h in indices2:
                h_index = indices2.index(index_h)
                conv_sum += values1[k] * values2[h_index]
        result_indices.append(n)
        result_values.append(conv_sum)
    CompareSignal.CompareSignal('Convolution testcases/Conv_output.txt',result_indices,result_values)
    plt.figure(figsize=(8, 6))
    plt.stem(result_indices, result_values, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.title("Convolved Signal")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def on_sharpening_button_click():
    indices, values = read_signals("Derivative testcases/Derivative_input.txt")
    first_derivative = []
    sec_derivative = []
    for i in range(1, len(indices)):

        first_derivative.append(values[i] - (0 if i == 0 else values[i - 1]))
        sec_derivative.append(0 if i == len(values) - 1 else (values[i+1] - (2 * values[i]) + (0 if i == 0 else values[i - 1])))

    sec_derivative.pop()

    f_indices = indices.copy()
    f_indices.pop()
    s_indices = f_indices.copy()
    s_indices.pop()

    CompareSignal.CompareSignal('Derivative testcases/1st_derivative_out.txt',f_indices ,first_derivative)
    CompareSignal.CompareSignal('Derivative testcases/2nd_derivative_out.txt',s_indices ,sec_derivative)


    plot_signal(f_indices, first_derivative, s_indices, sec_derivative, 'Sharpening Signal', 'First Derivative Signal', 'Second Derivative Signal')

    print(indices)
    print(first_derivative)
    print(sec_derivative)

def on_convolution_button_click():
    Conv_Signals()

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


title_label = tk.Label(root, text="DSP-Task", font=("Helvetica", 32), bg='lightgrey')
title_label.place(relx=0.49, rely=0.04, anchor='center')  # Center the title label

# Buttons
signal1_button = tk.Button(root, text="Read Signal 1", command=on_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
signal1_button.place(relx=0.48, rely=0.12, anchor='e')

displaysignal1_button = tk.Button(root, text="Display Signal 1", command=on_displaysignal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
displaysignal1_button.place(relx=0.48, rely=0.19, anchor='e')

signal2_button = tk.Button(root, text="Read Signal 2", command=on_signal2_button_click, width=20, height=2, bg='lightgrey', relief='flat')
signal2_button.place(relx=0.48, rely=0.26, anchor='e')

displaysignal2_button = tk.Button(root, text="Display Signal 2", command=on_displaysignal2_button_click, width=20, height=2, bg='lightgrey', relief='flat')
displaysignal2_button.place(relx=0.48, rely=0.33, anchor='e')

displaybothsignal_button = tk.Button(root, text="Display Both Signals", command=on_displaybothsignals_button_click, width=20, height=2, bg='lightgrey', relief='flat')
displaybothsignal_button.place(relx=0.48, rely=0.4, anchor='e')


add_signals_button = tk.Button(root, text="Add Signals", command=on_add_signals_button_click, width=20, height=2, bg='lightgrey', relief='flat')
add_signals_button.place(relx=0.5, rely=0.12, anchor='w')

sub_signals_button = tk.Button(root, text="Subtract Signals", command=on_sub_signals_button_click, width=20, height=2, bg='lightgrey', relief='flat')
sub_signals_button.place(relx=0.5, rely=0.19, anchor='w')

multiply_signal1_button = tk.Button(root, text="Multiply Signal", command=on_multiply_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
multiply_signal1_button.place(relx=0.5, rely=0.26, anchor='w')

fold_signal1_button = tk.Button(root, text="Fold Signal", command=on_fold_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
fold_signal1_button.place(relx=0.5, rely=0.33, anchor='w')

delay_advancing_signal1_button = tk.Button(root, text="Delay/Advance Signal", command=on_delay_advancing_signal1_button_click, width=20, height=2, bg='lightgrey', relief='flat')
delay_advancing_signal1_button.place(relx=0.5, rely=0.4, anchor='w')

generate_signal_button = tk.Button(root, text="Generate Signal", command=on_generate_signal_button_click, width=20, height=2, bg='lightgrey', relief='flat')
generate_signal_button.place(relx=0.5, rely=0.47, anchor='w')

quantize_button = tk.Button(root, text="Quantize Signal", command=on_quantize_signal_button_click, width=20, height=2, bg='lightgrey', relief='flat')
quantize_button.place(relx=0.48, rely=0.47, anchor='e')

compute_average_button = tk.Button(root, text="Compute Signal Average", command=on_compute_average_button_click, width=20, height=2, bg='lightgrey', relief='flat')
compute_average_button.place(relx=0.48, rely=0.54, anchor='e')

sharpening_button = tk.Button(root, text="Sharpen Signal", command=on_sharpening_button_click, width=20, height=2, bg='lightgrey', relief='flat')
sharpening_button.place(relx=0.5, rely=0.54, anchor='w')

conv_button = tk.Button(root, text="Convolve Signals", command=on_convolution_button_click, width=20, height=2, bg='lightgrey', relief='flat')
conv_button.place(relx=0.48, rely=0.61, anchor='e')

root.mainloop()