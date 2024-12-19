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
import signalcompare as sc
import math
import CorrCompareSignal
import numpy as np
from scipy.fft import fft, ifft


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
            value = float(parts[1])  # was int & don't know if it will cause an error in other functions
            indices.append(index)
            values.append(value)
    return indices, values


def plot_signal(x1, y1, x2, y2, label, label1, label2):
    window = tk.Toplevel()
    window.geometry("350x250")

    # Add the representation combobox
    tk.Label(window, text = "Representation Type:").grid(row = 0, column = 0, padx = 10, pady = 5)
    representation = ttk.Combobox(window, values = ["Continuous", "Discrete"])
    representation.grid(row = 0, column = 1, padx = 10, pady = 5)

    def plot():
        representation_value = representation.get()  # Get the selected value when plotting
        if representation_value == 'Continuous':
            if (isinstance(x1, (list, np.ndarray)) and len(x1) > 0) and (
                    isinstance(y1, (list, np.ndarray)) and len(y1) > 0):
                plt.plot(x1, y1, marker = 'o', linestyle = '-', label = label1, color = 'blue',
                         markerfacecolor = 'orange', markeredgecolor = 'orange')
            if (isinstance(x2, (list, np.ndarray)) and len(x2) > 0) and (
                    isinstance(y2, (list, np.ndarray)) and len(y2) > 0):
                plt.plot(x2, y2, marker = 'o', linestyle = '-', label = label2, color = 'red', markerfacecolor = 'red')
        else:
            if (isinstance(x1, (list, np.ndarray)) and len(x1) > 0) and (
                    isinstance(y1, (list, np.ndarray)) and len(y1) > 0):
                plt.stem(x1, y1, linefmt = 'b-', markerfmt = 'bo', basefmt = ' ', label = label1)
            if (isinstance(x2, (list, np.ndarray)) and len(x2) > 0) and (
                    isinstance(y2, (list, np.ndarray)) and len(y2) > 0):
                plt.stem(x2, y2, linefmt = 'r-', markerfmt = 'ro', basefmt = ' ', label = label2)

        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.title(label)
        plt.show()

    # Add a button to plot the signal
    plot_button = tk.Button(window, text = "Plot", command = plot)
    plot_button.grid(row = 1, column = 1, padx = 10, pady = 10)


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
    # signal2 = (signal2[0], values2)

    indices, values = add_signals(signal1, (signal2[0], values2))
    return indices, values


def delay_advancing_signals(signal1, constant):
    indices = signal1[0]
    indices = [index - constant for index in indices]
    # signal1 = (indices, signal1[1])
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
    test.MultiplySignalByConst(5, indices, multiplication_result)
    plot_signal(signal1[0], signal1[1], indices, multiplication_result, "Multiplied Signal", "Original Signal",
                "Multiplied Signal")
    return indices, multiplication_result


def on_signal1_button_click():
    global signal1
    signal1 = read_signals('Signal1.txt')


def on_displaysignal1_button_click():
    plot_signal(signal1[0], signal1[1], 0, 0, 'Signal 1', "Signal 1", "")


def on_signal2_button_click():
    global signal2
    signal2 = read_signals('Signal2.txt')


def on_displaysignal2_button_click():
    plot_signal(signal2[0], signal2[1], 0, 0, 'Signal 2', " Signal 2", "")


def on_add_signals_button_click():
    # Add the two signals
    indices, added_values = add_signals(signal1, signal2)
    test.AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indices, added_values)

    # Plot the result
    plot_signal(0, 0, indices, added_values, 'Addition Result', "", "Addition Result")


def on_sub_signals_button_click():
    indeces, values = sub_signals(signal1, signal2)

    test.SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt", indeces, values)
    plot_signal(0, 0, indeces, values, "Subtraction Result", " ", 'Subtraction Result')


def on_multiply_signal1_button_click():
    multiply_signal(signal1)


def on_fold_signal1_button_click():
    indices, values = folding_signals(signal1)
    test.Folding(indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Folded Signal", "Original Signal", "Folded Signal")


def on_delay_advancing_signal1_button_click():
    constant = simpledialog.askfloat("Input", "Enter the constant value to shift the signal:")
    indices, values = delay_advancing_signals(signal1, constant)
    test.ShiftSignalByConst(constant, indices, values)
    plot_signal(signal1[0], signal1[1], indices, values, "Shifted Signal", "Original Signal", "Shifted Signal")


def on_displaybothsignals_button_click():
    plot_signal(signal1[0], signal1[1], signal2[0], signal2[1], 'Both Signals', "Signal 1", 'Signal 2')


def on_generate_signal_button_click():
    window = tk.Toplevel()
    window.title("Signal Generation")
    window.geometry("450x300")

    tk.Label(window, text = "Signal Type:").grid(row = 0, column = 0, padx = 10, pady = 5)
    signal_type = ttk.Combobox(window, values = ["Sine", "Cosine"])
    signal_type.grid(row = 0, column = 1, padx = 10, pady = 5)

    tk.Label(window, text = "Amplitude (A):").grid(row = 1, column = 0, padx = 10, pady = 5)
    amplitude_entry = tk.Entry(window)
    amplitude_entry.grid(row = 1, column = 1, padx = 10, pady = 5)

    tk.Label(window, text = "Phase Shift (θ):").grid(row = 2, column = 0, padx = 10, pady = 5)
    phase_shift_entry = tk.Entry(window)
    phase_shift_entry.grid(row = 2, column = 1, padx = 10, pady = 5)

    tk.Label(window, text = "Analog Frequency:").grid(row = 3, column = 0, padx = 10, pady = 5)
    analog_freq_entry = tk.Entry(window)
    analog_freq_entry.grid(row = 3, column = 1, padx = 10, pady = 5)

    tk.Label(window, text = "Sampling Frequency:").grid(row = 4, column = 0, padx = 10, pady = 5)
    sampling_freq_entry = tk.Entry(window)
    sampling_freq_entry.grid(row = 4, column = 1, padx = 10, pady = 5)

    def on_generate():
        signal_type_value = signal_type.get()
        amplitude_value = float(amplitude_entry.get())
        phase_shift_value = float(phase_shift_entry.get())
        analog_freq_value = float(analog_freq_entry.get())
        sampling_freq_value = float(sampling_freq_entry.get())
        signal, time = generate_signal(signal_type_value, amplitude_value, phase_shift_value, analog_freq_value,
                                       sampling_freq_value)
        plot_signal(time, signal, [], [], 'Generate Signal', "Generated Signal", "")

    tk.Button(window, text = "Generate", command = on_generate).grid(row = 6, column = 1, padx = 10, pady = 5)


def generate_signal(signal_type, amp, phase_shift, analog_freq, sample_freq):
    #    try:

    if sample_freq < 2 * analog_freq:
        messagebox.showerror("Error!!", f"Sampling frequency should be at least {2 * analog_freq} Hz.")
        return None
        # raise ValueError("Invalid sampling frequency value! Sampling frequency should be greater than or equal 2 * analog frequency.")

    else:
        t = np.arange(0, 1, 1 / sample_freq)
        if signal_type == "Sine":
            signal = amp * np.sin(2 * np.pi * analog_freq * t + phase_shift)
        else:
            signal = amp * np.cos(2 * np.pi * analog_freq * t + phase_shift)

    return signal, t
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
        s = quant_type.get()
        if quant_type.get() == "Bits":
            levels = 2 ** int(type_entry.get())
        quantize_signal(levels, s)

    tk.Button(window, text = "Generate", command = on_quantize).grid(row = 2, column = 1, padx = 10, pady = 5)
    # QuanTest1.QuantizationTest1("Quan1_input.txt", quantized_values, )


def quantize_signal(levels, s):
    if s == 'Bits':
        signal = read_signals("Quan1_input.txt")
        index = signal[0]
        values = signal[1]
    else:
        signal = read_signals("Quan2_input.txt")
        index = signal[0]
        values = signal[1]

    encoded_signal = []
    minx = min(values)
    maxx = max(values)
    # levels = simpledialog.askinteger("Input", "Enter number of levels:")
    delta = (maxx - minx) / levels

    ranges = []
    interval_indices = []

    # if levels is not None:
    for i in range(levels):
        low = minx + i * delta
        high = minx + (i + 1) * delta
        ranges.append((low, high))

    points = []
    for sub in ranges:
        mid = (sub[0] + sub[1]) / 2
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
    plt.figure(figsize = (12, 8))

    # Original and Quantized Signal
    plt.subplot(3, 1, 1)
    plt.plot(signal[0], signal[1], 'o-', label = "Original Signal", color = 'blue')
    plt.step(signal[0], quantized, label = "Quantized Signal", color = 'orange', where = 'mid', marker = 'o')
    plt.title("Original and Quantized Signal")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.xticks(signal[0])

    # Quantization Error
    plt.subplot(3, 1, 2)
    plt.plot(signal[0], quantization_error, 'ro-', label = "Quantization Error")
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
    plt.text(0.5, 0.5, binary_text, ha = 'center', va = 'center', fontsize = 10, color = "black", family = "monospace",
             wrap = True)
    plt.title("Encoding of Quantized Signal")

    # Add average power error to the plot
    plt.figtext(0.05, 0.2, f"Average Power Error: {avg_power_error:.4f}", ha = 'left', fontsize = 12, color = 'black')

    plt.tight_layout()
    plt.show()
    if s == 'Bits':
        qt1.QuantizationTest1('Quan1_Out.txt', encoded_signal, quantized)
    else:
        qt2.QuantizationTest2('Quan2_Out.txt', interval_indices, encoded_signal, quantized, quantization_error)


def on_compute_average_button_click():
    window = tk.Toplevel()
    window.title("Compute Signal Average")
    window.geometry("450x300")
    tk.Label(window, text = "Window Size", fg = "#003366", font = ("Helvetica", 10)).grid(row = 0, column = 0,
                                                                                          padx = 10, pady = 5)
    window_size = tk.Entry(window)
    window_size.grid(row = 0, column = 1, padx = 10, pady = 5)

    def on_average():
        windows = int(window_size.get())
        average_signal(windows)

    tk.Button(window, text = 'Average', fg = "#003366", font = ("Helvetica", 10), command = on_average).grid(row = 1,
                                                                                                             column = 1,
                                                                                                             padx = 15,
                                                                                                             pady = 5)


def average_signal(window_size):
    indices, values = read_signals("Moving Average testcases/MovingAvg_input.txt")
    new_indicis = []
    new_values = []
    new_index = 0
    for i in range(len(indices) - window_size + 1):
        sum = 0
        for j in range(i, i + window_size):
            if j < len(values):
                sum += values[j]
                # print(j, "\t", values[j])
        avg = round(sum / window_size, 2)
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

def Conv_Signals(signal1, signal2):
    # y(n)=x(n)*h(x)
    # y(n)=sum [x(k)*h(n-k)]
    indices1, values1 = signal1
    indices2, values2 = signal2

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

    return result_indices, result_values


def on_sharpening_button_click():
    indices, values = read_signals("Derivative testcases/Derivative_input.txt")
    first_derivative = []
    sec_derivative = []
    for i in range(1, len(indices)):
        first_derivative.append(values[i] - (0 if i == 0 else values[i - 1]))
        sec_derivative.append(
            0 if i == len(values) - 1 else (values[i + 1] - (2 * values[i]) + (0 if i == 0 else values[i - 1])))

    sec_derivative.pop()

    f_indices = indices.copy()
    f_indices.pop()
    s_indices = f_indices.copy()
    s_indices.pop()

    CompareSignal.CompareSignal('Derivative testcases/1st_derivative_out.txt',f_indices ,first_derivative)
    CompareSignal.CompareSignal('Derivative testcases/2nd_derivative_out.txt',s_indices ,sec_derivative)


    plot_signal(f_indices, first_derivative, s_indices, sec_derivative, 'Sharpening Signal', 'First Derivative Signal',
                'Second Derivative Signal')

    print(indices)
    print(first_derivative)
    print(sec_derivative)


def read_output(file_path):
    amplitudeout = []
    phaseout = []
    with open(file_path, 'r') as file:
        lines = file.readlines()[3:]  # Skip the first 3 lines
        for line in lines:
            data = line.strip().split()
            if len(data) == 2:
                # Extract amplitude and phase values from the signal file
                value, phase_val = data
                value = value[:-1] if value.endswith('f') else value  # Remove the trailing 'f'
                value = float(value)
                phase_val = phase_val[:-1] if phase_val.endswith('f') else phase_val  # Remove the trailing 'f'
                phase_val = float(phase_val)
                # Append the values to the lists
                amplitudeout.append(value)
                phaseout.append(phase_val)
    return amplitudeout,phaseout


def DFT(dft_in):

    samplingFreq = simpledialog.askfloat("Input", "Enter the sampling frequency in HZ:")
    indices = dft_in[0]
    values = dft_in[1]


    # Formula: X[k] = sum(n=0 to N-1) x[n] * exp(-j * 2 * pi * k * n / N)
    N = len(values)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k]+=values[n]*np.exp(-1j * 2 * np.pi * k * n / N)

    frequencies = np.arange(N) * samplingFreq / N

    return frequencies, X




def on_convolution_button_click():
    signal1 = read_signals('Convolution testcases/Signal 1.txt')
    signal2 = read_signals('Convolution testcases/Signal 2.txt')
    result_indices, result_values = Conv_Signals(signal1, signal2)

    CompareSignal.CompareSignal('Convolution testcases/Conv_output.txt', result_indices, result_values)
    plt.figure(figsize = (8, 6))
    plt.stem(result_indices, result_values, linefmt = 'b-', markerfmt = 'bo', basefmt = 'r-')
    plt.title("Convolved Signal")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()




def on_dft_button_click():
    dft_in = read_signals('Test Cases/DFT/input_Signal_DFT.txt')
    amplitudeout, phaseout = read_output('Test Cases/DFT/Output_Signal_DFT_A,Phase.txt')
    frequencies, X = DFT(dft_in)

    amplitude = np.abs(X)
    phase = np.angle(X)

    Amp = sc.SignalComapreAmplitude(amplitude, amplitudeout)
    Phase = sc.SignalComaprePhaseShift(phase, phaseout)

    if Amp and Phase:
        messagebox.showinfo("Passed", "Amplitude and Phase values match in the two files.")
    else:
        messagebox.showerror("Failed", "Amplitude and/or Phase values don't match in the two files.")

    # Plot Frequency vs Amplitude
    plt.figure(figsize = (12, 6))

    # Amplitude plot
    plt.subplot(1, 2, 1)
    plt.stem(frequencies, amplitude)
    plt.title("Frequency vs Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()

    # Phase plot
    plt.subplot(1, 2, 2)
    plt.stem(frequencies, phase)
    plt.title("Frequency vs Phase")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid()

    plt.tight_layout()
    plt.show()
    print(X)

def on_idft_button_click():
    amp, phase = read_output('Test Cases/IDFT/Input_Signal_IDFT_A,Phase.txt')
    indices, inv_X, real_inv_X = idft_signal([amp, phase])

    indices_valid, inv_valid = read_signals('Test Cases/IDFT/Output_Signal_IDFT.txt')
    indx = sc.SignalComapreAmplitude(indices, indices_valid)
    inv = sc.SignalComaprePhaseShift(inv_X, inv_valid)

    if indx and inv:
        messagebox.showinfo("Passed", "Indices and Inverse values match in the two files.")
    else:
        messagebox.showerror("Failed", "Indices and/or Inverse values don't match in the two files.")

    plot_signal(indices, real_inv_X, 0, 0, 'Fourier Transformation', 'IDFT', '')


def idft_signal(signal):
    amp = signal[0]
    phase = signal[1]
    # X = []
    # for a, p in zip(amp, phase):
    #     temp = a * (np.cos(p) + 1j * np.sin(p)) if p >= 0 else a * (np.cos(p) - 1j * np.sin(p))
    #     X.append(temp)

    X = amp * (np.cos(phase) + 1j * np.sin(phase))

    N = len(X)
    inv_X = [0 + 0j] * N  # Initialize the output sequence as complex numbers
    # X(n) = inv(X(k)) = 1/N sum(X(k) * (e ** j*k*2*pi*n/N))   / e**jtheta = cos(theta) + jsin(theta)
    indices = []
    for n in range(N):
        indices.append(n)
        for k in range(N):
            inv_X[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
        inv_X[n] /= N

    real_inv_X = [round(value.real) for value in inv_X]  # Keep real parts and round

    X2 = amp * (np.cos(phase) + 1j * np.sin(phase))
    inv_X_valid = np.fft.ifft(X2).real

    return indices, inv_X, real_inv_X



def on_corr_button_click():
    window = tk.Toplevel()
    window.title("Signals Correlation")
    window.geometry("450x300")
    # Add the representation combobox
    tk.Label(window, text = "Option:").grid(row = 0, column = 0, padx = 10, pady = 5)
    corr_opt = ttk.Combobox(window, values = ["Correlation", "Time Delay", "Classify Signals"])
    corr_opt.grid(row = 0, column = 1, padx = 10, pady = 5)



    def on_continue():
        opt = corr_opt.get()
        if opt == 'Correlation':
            signal1 = read_signals("Correlation Task Files/Point1 Correlation/Corr_input signal1.txt")
            signal2 = read_signals("Correlation Task Files/Point1 Correlation/Corr_input signal2.txt")
            Kindx, K = corr_signal(signal1, signal2)

            CorrCompareSignal.Compare_Signals("Correlation Task Files/Point1 Correlation/CorrOutput.txt", Kindx, K)
            plt.plot(Kindx, K, marker = 'o', linestyle = '-', color = 'blue',
                     markerfacecolor = 'orange', markeredgecolor = 'orange')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.title("Signals Correlation")
            plt.show()

        elif opt == 'Time Delay':
            time_delay()

        else:
            for i in range(1,3):
                verdict = classify_signal(f"Correlation Task Files/point3 Files/Test Signals/Test{i}.txt")
                messagebox.showinfo("Test Classification", f"Test{i} classified as {verdict}.")


    tk.Button(window, text = 'Continue', fg = "#003366", font = ("Helvetica", 10), command = on_continue).grid(row = 1,
                                                                                                             column = 1,
                                                                                                             padx = 15,
                                                                                                             pady = 30)



def corr_signal(signal1, signal2):
    # k[i] = 1/N * sum( x1[n] * x2[n+i])

    indices1 = signal1[0]
    values1 = signal1[1]
    indices2 = signal2[0]
    values2 = signal2[1]

    N1 = len(indices1)
    N2 = len(indices2)

    # If the signals have different lengths, extend the shorter signal with zeros
    if N1 != N2:
        N = N1 + N2 - 1
        values1 = np.append(values1, [0] * (N - N1))
        values2 = np.append(values2, [0] * (N - N2))
    else:
        N = N1


    #K = np.zeros(indices1)
    #N = len(indices1)
    #lags = np.arange(-len(indices1) + 1, len(indices2))  # Lag indices
    # K = [0] * len(lags)
    # K = [0] * len(indices1)
    #
    # sum_signal1 = 0
    # sum_signal2 = 0
    #
    # for i in range(N):
    #     sum_signal1 += values1[i] ** 2
    #     sum_signal2 += values2[i] ** 2
    #
    #
    # norm = 1/N * ((sum_signal1 * sum_signal2)**0.5)
    # lags = []
    # n = len(indices1)
    # m = len(indices2)
    #
    # for i in range(-n + 1, m):
    #     lags.append(i)
    #     corr = 0
    #     for n in range(0, N):
    #         if n + i >= N: n -= N
    #         K[i] += values1[n] * values2[n + i]
    #     K[i] /= N
    #     K[i] /= norm
    #
    #
    # return indices1, K

    # Initialize correlation and lags
    correlation = []
    lags = []

    # Compute the correlation for each lag
    for k in range(N):
        sum_corr = 0
        for n in range(N):
            valid_index = (n + k) % N
            sum_corr += values1[n] * values2[valid_index]
        correlation.append((1 / N) * sum_corr)
        lags.append(k)

    # Normalization
    norm = (1 / N) * (np.sqrt(np.dot(values1, values1) * np.dot(values2, values2)))
    correlation = np.array(correlation) / norm

    return lags, correlation


def time_delay():
    fs = simpledialog.askinteger("Input", "Enter the sampling frequency:")

    signal1_idx, signal1_val = read_signals("Correlation Task Files/Point2 Time analysis/TD_input signal1.txt")
    signal2_idx, signal2_val = read_signals("Correlation Task Files/Point2 Time analysis/TD_input signal2.txt")

    corr_lags, corr_val = corr_signal([signal1_idx, signal1_val], [signal2_idx, signal2_val])
    print(corr_lags)
    print(corr_val)

    # Find the lag with the maximum correlation
    max_lag = np.argmax(corr_val)#corr_lags[np.argmax(corr_val)]
    print(max_lag)

    td = max_lag / fs  # Convert lag to time
    messagebox.showinfo("Time Delay Result", f"Time Delay is {td} seconds")

def read_corr_signals(file_path):
    indices = []
    values = []
    # Open the specified file and read its content
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Convert each line to an integer and strip any whitespace
    values = [int(line.strip()) for line in data if line.strip()]
    indices = list(range(len(values)))
    print(indices)
    print(values)
    return indices, values

def classify_signal(path):
    signal1 = read_corr_signals(path)

    down_max_sum = 0
    up_max_sum = 0

    for i in range(1, 6):
        down_signal_i = read_corr_signals(f"Correlation Task Files/point3 Files/Class 1/down{i}.txt")
        corr = corr_signal(signal1, down_signal_i)
        #down_Corr.append(corr)
        down_max_sum += max(corr[1])

    avg_down = down_max_sum / 5

    for i in range(1, 6):
        up_signal_i = read_corr_signals(f"Correlation Task Files/point3 Files/Class 2/up{i}.txt")
        corr = corr_signal(signal1, up_signal_i)
        up_max_sum += max(corr[1])

    avg_up = up_max_sum / 5

    verdict = "Class 1" if avg_down >= avg_up else "Class 2"
    return verdict



def convolution(signal, h):
    """
    Perform convolution using numpy's built-in convolve function.

    Args:
        signal: List of tuples (index, value) representing the input signal
        h: List of tuples (index, value) representing the filter coefficients

    Returns:
        List of tuples (index, value) representing the convolved signal
    """
    # Extract values from tuples
    x = [val for _, val in signal]
    h_vals = [val for _, val in h]

    # Perform convolution using numpy
    y = np.convolve(x, h_vals, mode='full')

    # Generate indices for output
    # Start index will be the sum of the minimum indices of signal and h
    start_index = signal[0][0] + h[0][0]

    # Create list of tuples (index, value) for output
    result = [(start_index + i, float(val)) for i, val in enumerate(y)]

    return result


#######################################################################
def fast_convolution(signal, h):

    # Extract values from tuples

    signal_values = np.array([x[1] for x in signal])
    h_values = np.array([x[1] for x in h])
    print(h[0])
    # Get lengths
    N = len(signal_values)
    M = len(h_values)

    # Pad signals to prevent circular convolution
    padded_length = N + M - 1
    signal_padded = np.pad(signal_values, (0, padded_length - len(signal_values)))
    h_padded = np.pad(h_values, (0, padded_length - len(h_values)))

    # Perform FFT convolution
    signal_fft = fft(signal_padded)
    h_fft = fft(h_padded)
    result = ifft(signal_fft * h_fft).real

    # Create output with proper indices
    indices = range(len(result))
    return list(zip(indices, result))

def filtering(sampling_freq, cutoff_freq, stop_attenuation, transition_band, filter_type):
    # window_name=" "
    # transition_width=0.0
    # N=0.0
    # signal = read_signals('FIR test cases/Testcase 8/ecg400.txt')
    # indices = signal[0]  # List of indices
    # values = signal[1]  # List of values
    #
    # # Create a list of tuples (index, value)
    # signal = list(zip(indices, values))

    if stop_attenuation <= 21:
        window_name = "rectangular"
        N = (sampling_freq * 0.9) / transition_band
        # transition_width = 0.9/N

    elif stop_attenuation > 21 and stop_attenuation <= 44:
        window_name = "hanning"
        N = (sampling_freq * 3.1) / transition_band
        transition_width = 3.1 / N

    elif stop_attenuation > 44 and stop_attenuation <= 53:
        window_name = "hamming"
        N = (sampling_freq * 3.3) / transition_band
        transition_width = 3.3 / N

    else:
        window_name = "blackman"
        N = (sampling_freq * 5.5) / transition_band
        transition_width = 5.5 / N

    print(window_name)
    N_odd = round(N)
    if N_odd % 2 == 0:
        N_odd += 1
    if N > N_odd:
        N_odd += 2

    print(N_odd)
    steps = int(N_odd / 2)
    # print(steps)
    min = -steps
    max = steps
    indices = []
    hvalues = []
    w = []
    # fcc=cutoff_freq+transition_band/2
    # fc=fcc/sampling_freq
    # df=transition_band/sampling_freq
    # fcc=cutoff_freq+df/2
    # fc=fcc/sampling_freq
    # print(fc)
    if filter_type == "Low Pass":
        fc = (cutoff_freq + (transition_band / 2)) / sampling_freq
        h = []
        for n in range(min, max + 1):
            # Determine the window function
            if window_name == "rectangular":
                w = 1
            elif window_name == "hanning":
                w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "hamming":
                w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "blackman":
                w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N_odd - 1)) + 0.08 * math.cos((4 * math.pi * n) / (N_odd - 1))
            else:
                raise ValueError("Unknown window function")

            # Calculate impulse response
            if n == 0:
                hd = 2 * fc
            else:
                hd = 2 * fc * (math.sin(n * 2 * math.pi * fc) / (n * 2 * math.pi * fc))

            # Append the weighted impulse response
            h.append((n, hd * w))
            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
        # CompareSignal.CompareSignal("FIR test cases/Testcase 1/LPFCoefficients.txt", indices, samples)
        # with open('Low pass coefficients.txt', 'w') as f:
        #     f.write(str(h))
        # print(h)
        signal = read_signals('FIR test cases/Testcase 2/ecg400.txt')
        indices = signal[0]  # List of indices
        values = signal[1]  # List of values

        # Create a list of tuples (index, value)
        signal = list(zip(indices, values))

        conv_res = convolution(signal, h)
        indicesx = [x[0] for x in conv_res]
        samplesc = [x[1] for x in conv_res]
        print(samples)
        CompareSignal.CompareSignal("FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", indicesx, samplesc)

        conv_res = fast_convolution(signal, h)
        indicesc = [x[0] for x in conv_res]
        samplesc = [x[1] for x in conv_res]
        # print(samplesc)
        CompareSignal.CompareSignal("FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", indicesx, samplesc)


    elif filter_type == "High Pass":
        fc = (cutoff_freq - (transition_band / 2)) / sampling_freq
        h = []
        for n in range(min, max + 1):
            # Determine the window function
            if window_name == "rectangular":
                w = 1
            elif window_name == "hanning":
                w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "hamming":
                w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "blackman":
                w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N_odd - 1)) + 0.08 * math.cos((4 * math.pi * n) / (N_odd - 1))
            else:
                raise ValueError("Unknown window function")

            # Calculate impulse response
            if n == 0:
                hd = 1 - 2 * fc
            else:
                hd = -2 * fc * (math.sin(n * 2 * math.pi * fc) / (n * 2 * math.pi * fc))
            # Append the weighted impulse response
            h.append((n, hd * w))
            indices = [x[0] for x in h]
            samples = [x[1] for x in h]
        CompareSignal.CompareSignal("FIR test cases/Testcase 3/HPFCoefficients.txt", indices, samples)
        # with open('high pass coefficients.txt', 'w') as f:
        #     f.write(str(h))
        signal = read_signals('FIR test cases/Testcase 4/ecg400.txt')
        indices = signal[0]  # List of indices
        values = signal[1]  # List of values

        # Create a list of tuples (index, value)
        signal = list(zip(indices, values))
        print(signal)
        # print("*")
        conv_res = convolution(signal, h)
        indices = [x[0] for x in conv_res]
        samples = [x[1] for x in conv_res]
        CompareSignal.CompareSignal("FIR test cases/Testcase 4/ecg_high_pass_filtered.txt", indices, samples)

        conv_res = fast_convolution(signal, h)
        indicesc = [x[0] for x in conv_res]
        samplesc = [x[1] for x in conv_res]
        # print(samplesc)
        CompareSignal.CompareSignal("FIR test cases/Testcase 4/ecg_high_pass_filtered.txt", indices, samplesc)

def filtering_band(sampling_freq,  stop_attenuation, transition_band, filter_type, f1, f2):
    # window_name=" "
    # transition_width=0.0
    # N=0.0
    # signal = read_signals('FIR test cases/Testcase 8/ecg400.txt')
    # indices = signal[0]  # List of indices
    # values = signal[1]  # List of values
    #
    # # Create a list of tuples (index, value)
    # signal = list(zip(indices, values))

    if stop_attenuation <= 21:
        window_name="rectangular"
        N=(sampling_freq*0.9)/transition_band
        # transition_width = 0.9/N

    elif stop_attenuation > 21 and stop_attenuation<=44:
        window_name = "hanning"
        N=(sampling_freq*3.1)/transition_band
        transition_width = 3.1/N

    elif stop_attenuation > 44 and stop_attenuation <= 53:
        window_name="hamming"
        N=(sampling_freq*3.3)/transition_band
        transition_width = 3.3/N

    else:
        window_name="blackman"
        N=(sampling_freq*5.5)/transition_band
        transition_width = 5.5/N

    print(window_name)
    N_odd= round(N)
    if N_odd % 2 == 0:
        N_odd += 1
    if N>N_odd:
        N_odd+=2

    print(N_odd)
    steps = int(N_odd / 2)
    # print(steps)
    # steps-=0.5
    min=-steps
    max=steps
    indices=[]
    hvalues=[]
    w=[]

    if filter_type == "Band Pass":
        fc1 = (f1 - (transition_band / 2)) / sampling_freq
        fc2 = (f2 + (transition_band / 2)) / sampling_freq
        h = []
        for n in range(min, max + 1):
            # Determine the window function
            if window_name == "rectangular":
                w = 1
            elif window_name == "hanning":
                w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "hamming":
                w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "blackman":
                w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N_odd - 1)) + 0.08 * math.cos((4 * math.pi * n) / (N_odd - 1))
            else:
                raise ValueError("Unknown window function")

            # Calculate impulse response
            if n == 0:
                hd = 2 * (fc2-fc1)
            else:
                fcc2=2 * fc2 * (math.sin(n * 2 * math.pi * fc2) / (n * 2 * math.pi * fc2))
                fcc1=2 * fc1 * (math.sin(n * 2 * math.pi * fc1) / (n * 2 * math.pi * fc1))
                hd = fcc2 -fcc1

            # Append the weighted impulse response
            h.append((n, hd * w))

        indices = [x[0] for x in h]
        samples = [x[1] for x in h]
        CompareSignal.CompareSignal("FIR test cases/Testcase 5/BPFCoefficients.txt", indices, samples)
        with open('band pass coefficients.txt', 'w') as f:
            f.write(str(h))
        signal = read_signals('FIR test cases/Testcase 6/ecg400.txt')
        indices = signal[0]  # List of indices
        values = signal[1]  # List of values

        # Create a list of tuples (index, value)
        signal = list(zip(indices, values))

        conv_res = convolution(signal, h)
        indices = [x[0] for x in conv_res]
        samples = [x[1] for x in conv_res]
        CompareSignal.CompareSignal("FIR test cases/Testcase 6/ecg_band_pass_filtered.txt", indices, samples)

        conv_res = fast_convolution(signal, h)
        indicesc = [x[0] for x in conv_res]
        samplesc = [x[1] for x in conv_res]
        # print(samplesc)
        CompareSignal.CompareSignal("FIR test cases/Testcase 6/ecg_band_pass_filtered.txt", indices, samplesc)


    elif filter_type == "Band Stop":
        fc1 = (f1 + (transition_band / 2)) / sampling_freq
        fc2 = (f2 - (transition_band / 2)) / sampling_freq
        h = []
        for n in range(min, max + 1):
            # Determine the window function
            if window_name == "rectangular":
                w = 1
            elif window_name == "hanning":
                w = 0.5 + 0.5 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "hamming":
                w = 0.54 + 0.46 * math.cos((2 * math.pi * n) / N_odd)
            elif window_name == "blackman":
                w = 0.42 + 0.5 * math.cos((2 * math.pi * n) / (N_odd - 1)) + 0.08 * math.cos((4 * math.pi * n) / (N_odd - 1))
            else:
                raise ValueError("Unknown window function")

            # Calculate impulse response
            if n == 0:
                hd = 1 - 2 * (fc2-fc1)
            else:
                fcc1 = 2 * fc1 * (math.sin(n * 2 * math.pi * fc1) / (n * 2 * math.pi * fc1))
                fcc2 = 2 * fc2 * (math.sin(n * 2 * math.pi * fc2) / (n * 2 * math.pi * fc2))
                hd =fcc1-fcc2
            # Append the weighted impulse response
            h.append((n, hd * w))
        indices = [x[0] for x in h]
        samples = [x[1] for x in h]
        CompareSignal.CompareSignal("FIR test cases/Testcase 7/BSFCoefficients.txt", indices, samples)
        with open('band stop coefficients.txt', 'w') as f:
            f.write(str(h))
        signal = read_signals('FIR test cases/Testcase 8/ecg400.txt')
        indices = signal[0]  # List of indices
        values = signal[1]  # List of values

        # Create a list of tuples (index, value)
        signal = list(zip(indices, values))

        conv_res = convolution(signal, h)
        indices = [x[0] for x in conv_res]
        samples = [x[1] for x in conv_res]
        CompareSignal.CompareSignal("FIR test cases/Testcase 8/ecg_band_stop_filtered.txt", indices, samples)

        conv_res = fast_convolution(signal, h)
        indicesc = [x[0] for x in conv_res]
        samplesc = [x[1] for x in conv_res]
        # print(samplesc)
        CompareSignal.CompareSignal("FIR test cases/Testcase 8/ecg_band_stop_filtered.txt", indices, samplesc)



def on_filtering_button_click():
    window = tk.Toplevel()
    window.title("Filter Parameters")
    window.geometry("450x300")

    # Filter Type (First Selection)
    tk.Label(window, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5)
    filter_type = ttk.Combobox(window, values=["Low Pass", "High Pass", "Band Pass", "Band Stop"])
    filter_type.set("Low Pass")  # Set default value
    filter_type.grid(row=0, column=1, padx=10, pady=5)

    # Create frames for different filter types
    single_freq_frame = tk.Frame(window)
    dual_freq_frame = tk.Frame(window)

    # Common parameters with default values
    sampling_freq_label = tk.Label(window, text="Sampling Frequency (Hz):")
    sampling_freq_entry = tk.Entry(window)

    stop_atten_label = tk.Label(window, text="Stop Attenuation (dB):")
    stop_atten_entry = tk.Entry(window)

    trans_band_label = tk.Label(window, text="Transition Band (Hz):")
    trans_band_entry = tk.Entry(window)

    # Single frequency entries
    cutoff_freq_label = tk.Label(single_freq_frame, text="Cutoff Frequency (Hz):")
    cutoff_freq_entry = tk.Entry(single_freq_frame)

    # Dual frequency entries
    f1_label = tk.Label(dual_freq_frame, text="F1 (Hz):")
    f1_entry = tk.Entry(dual_freq_frame)

    f2_label = tk.Label(dual_freq_frame, text="F2 (Hz):")
    f2_entry = tk.Entry(dual_freq_frame)

    def update_frame(*args):
        # Clear previous entries
        single_freq_frame.grid_forget()
        dual_freq_frame.grid_forget()

        # Show common parameters
        sampling_freq_label.grid(row=1, column=0, padx=10, pady=5)
        sampling_freq_entry.grid(row=1, column=1, padx=10, pady=5)
        stop_atten_label.grid(row=4, column=0, padx=10, pady=5)
        stop_atten_entry.grid(row=4, column=1, padx=10, pady=5)
        trans_band_label.grid(row=5, column=0, padx=10, pady=5)
        trans_band_entry.grid(row=5, column=1, padx=10, pady=5)

        # Show relevant frequency entries
        if filter_type.get() in ["Low Pass", "High Pass"]:
            single_freq_frame.grid(row=2, column=0, columnspan=2)
            cutoff_freq_label.grid(row=0, column=0, padx=10, pady=5)
            cutoff_freq_entry.grid(row=0, column=1, padx=10, pady=5)
        else:
            dual_freq_frame.grid(row=2, column=0, columnspan=2)
            f1_label.grid(row=0, column=0, padx=10, pady=5)
            f1_entry.grid(row=0, column=1, padx=10, pady=5)
            f2_label.grid(row=1, column=0, padx=10, pady=5)
            f2_entry.grid(row=1, column=1, padx=10, pady=5)

    def validate_inputs():
        sampling_freq = float(sampling_freq_entry.get().strip())
        if sampling_freq <= 0:
            return False, "Sampling frequency must be positive"

        stop_attenuation = float(stop_atten_entry.get().strip())
        if stop_attenuation <= 0:
            return False, "Stop attenuation must be positive"

        transition_band = float(trans_band_entry.get().strip())
        if transition_band <= 0:
            return False, "Transition band must be positive"

        if filter_type.get() in ["Low Pass", "High Pass"]:
            cutoff_freq = float(cutoff_freq_entry.get().strip())
            if cutoff_freq <= 0 or cutoff_freq >= sampling_freq / 2:
                return False, "Cutoff frequency must be between 0 and Nyquist frequency"
        else:
            f1 = float(f1_entry.get().strip())
            f2 = float(f2_entry.get().strip())
            if f1 >= f2 or f1 <= 0 or f2 >= sampling_freq / 2:
                return False, "F1 must be less than F2, and both must be between 0 and Nyquist frequency"

        return True, ""

    def on_filter():
        is_valid, error_message = validate_inputs()
        if not is_valid:
            tk.messagebox.showerror("Error", error_message)
            return

        sampling_freq = float(sampling_freq_entry.get().strip())
        stop_attenuation = float(stop_atten_entry.get().strip())
        transition_band = float(trans_band_entry.get().strip())
        current_filter_type = filter_type.get()

        if current_filter_type in ["Low Pass", "High Pass"]:
            cutoff_freq = float(cutoff_freq_entry.get().strip())
            filtering(sampling_freq, cutoff_freq, stop_attenuation, transition_band, current_filter_type)
        else:
            fl = float(f1_entry.get().strip())
            f2 = float(f2_entry.get().strip())
            filtering_band(sampling_freq,  stop_attenuation, transition_band, current_filter_type, fl, f2)

        window.destroy()

    # Generate button
    tk.Button(window, text="Generate", command=on_filter).grid(row=6, column=1, padx=10, pady=15)

    # Bind the update function to filter type changes
    filter_type.bind('<<ComboboxSelected>>', update_frame)

    # Initial setup
    update_frame()
######################################################################



# GUI
root = tk.Tk()
root.title("DSP")
root.geometry("1550x1550")

# Load and set the background image
background_image = Image.open("bg.jpeg")
bg_image = ImageTk.PhotoImage(background_image)

# Create a label for the background image
background_label = tk.Label(root, image = bg_image)
background_label.place(relwidth = 1, relheight = 1)

title_label = tk.Label(root, text = "DSP-Task", font = ("Helvetica", 32), bg = 'lightgrey')
title_label.place(relx = 0.49, rely = 0.04, anchor = 'center')  # Center the title label

# Buttons
signal1_button = tk.Button(root, text = "Read Signal 1", command = on_signal1_button_click, width = 20, height = 2,
                           bg = 'lightgrey', relief = 'flat')
signal1_button.place(relx = 0.48, rely = 0.12, anchor = 'e')

displaysignal1_button = tk.Button(root, text = "Display Signal 1", command = on_displaysignal1_button_click, width = 20,
                                  height = 2, bg = 'lightgrey', relief = 'flat')
displaysignal1_button.place(relx = 0.48, rely = 0.19, anchor = 'e')

signal2_button = tk.Button(root, text = "Read Signal 2", command = on_signal2_button_click, width = 20, height = 2,
                           bg = 'lightgrey', relief = 'flat')
signal2_button.place(relx = 0.48, rely = 0.26, anchor = 'e')

displaysignal2_button = tk.Button(root, text = "Display Signal 2", command = on_displaysignal2_button_click, width = 20,
                                  height = 2, bg = 'lightgrey', relief = 'flat')
displaysignal2_button.place(relx = 0.48, rely = 0.33, anchor = 'e')

displaybothsignal_button = tk.Button(root, text = "Display Both Signals", command = on_displaybothsignals_button_click,
                                     width = 20, height = 2, bg = 'lightgrey', relief = 'flat')
displaybothsignal_button.place(relx = 0.48, rely = 0.4, anchor = 'e')

add_signals_button = tk.Button(root, text = "Add Signals", command = on_add_signals_button_click, width = 20,
                               height = 2, bg = 'lightgrey', relief = 'flat')
add_signals_button.place(relx = 0.5, rely = 0.12, anchor = 'w')

sub_signals_button = tk.Button(root, text = "Subtract Signals", command = on_sub_signals_button_click, width = 20,
                               height = 2, bg = 'lightgrey', relief = 'flat')
sub_signals_button.place(relx = 0.5, rely = 0.19, anchor = 'w')

multiply_signal1_button = tk.Button(root, text = "Multiply Signal", command = on_multiply_signal1_button_click,
                                    width = 20, height = 2, bg = 'lightgrey', relief = 'flat')
multiply_signal1_button.place(relx = 0.5, rely = 0.26, anchor = 'w')

fold_signal1_button = tk.Button(root, text = "Fold Signal", command = on_fold_signal1_button_click, width = 20,
                                height = 2, bg = 'lightgrey', relief = 'flat')
fold_signal1_button.place(relx = 0.5, rely = 0.33, anchor = 'w')

delay_advancing_signal1_button = tk.Button(root, text = "Delay/Advance Signal",
                                           command = on_delay_advancing_signal1_button_click, width = 20, height = 2,
                                           bg = 'lightgrey', relief = 'flat')
delay_advancing_signal1_button.place(relx = 0.5, rely = 0.4, anchor = 'w')

generate_signal_button = tk.Button(root, text = "Generate Signal", command = on_generate_signal_button_click,
                                   width = 20, height = 2, bg = 'lightgrey', relief = 'flat')
generate_signal_button.place(relx = 0.5, rely = 0.47, anchor = 'w')

quantize_button = tk.Button(root, text = "Quantize Signal", command = on_quantize_signal_button_click, width = 20,
                            height = 2, bg = 'lightgrey', relief = 'flat')
quantize_button.place(relx = 0.48, rely = 0.47, anchor = 'e')

compute_average_button = tk.Button(root, text = "Compute Signal Average", command = on_compute_average_button_click,
                                   width = 20, height = 2, bg = 'lightgrey', relief = 'flat')
compute_average_button.place(relx = 0.48, rely = 0.54, anchor = 'e')

sharpening_button = tk.Button(root, text = "Sharpen Signal", command = on_sharpening_button_click, width = 20,
                              height = 2, bg = 'lightgrey', relief = 'flat')
sharpening_button.place(relx = 0.5, rely = 0.54, anchor = 'w')

# sharpening_button = tk.Button(root, text="Sharpen Signal", command=on_sharpening_button_click, width=20, height=2, bg='lightgrey', relief='flat')
# sharpening_button.place(relx=0.5, rely=0.54, anchor='w')

conv_button = tk.Button(root, text="Convolve Signals", command=on_convolution_button_click, width=20, height=2, bg='lightgrey', relief='flat')
conv_button.place(relx=0.48, rely=0.61, anchor='e')

dft_button = tk.Button(root, text="DFT", command=on_dft_button_click, width=20, height=2, bg='lightgrey', relief='flat')
dft_button.place(relx=0.5, rely=0.61, anchor='w')

idft_button = tk.Button(root, text="IDFT", command=on_idft_button_click, width=20, height=2, bg='lightgrey', relief='flat')
idft_button.place(relx=0.48, rely=0.68, anchor='e')

corr_button = tk.Button(root, text="Correlation", command=on_corr_button_click, width=20, height=2, bg='lightgrey', relief='flat')
corr_button.place(relx=0.5, rely=0.68, anchor='w')

filter_button = tk.Button(root, text="Filtering", command=on_filtering_button_click, width=20, height=2, bg='lightgrey', relief='flat')
filter_button.place(relx=0.48, rely=0.75, anchor='e')

root.mainloop()

