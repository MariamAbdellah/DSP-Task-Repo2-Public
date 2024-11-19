<<<<<<< HEAD
def QuantizationTest2(file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
    expectedIntervalIndices=[]
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    expectedSampledError=[]
=======
import tkinter as tk
from tkinter import messagebox


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []

>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
<<<<<<< HEAD
            L=line.strip()
            if len(L.split(' '))==4:
                L=line.split(' ')
                V1=int(L[0])
                V2=str(L[1])
                V3=float(L[2])
                V4=float(L[3])
=======
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
<<<<<<< HEAD
    if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
     or len(Your_EncodedValues)!=len(expectedEncodedValues)
      or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
      or len(Your_SampledError)!=len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
            return
        
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one") 
            return
    print("QuantizationTest2 Test case passed successfully")
=======

    if (len(Your_IntervalIndices) != len(expectedIntervalIndices) or
            len(Your_EncodedValues) != len(expectedEncodedValues) or
            len(Your_QuantizedValues) != len(expectedQuantizedValues) or
            len(Your_SampledError) != len(expectedSampledError)):
        messagebox.showerror("Test Case Failed",
                             " Test case failed, your signal has a different length from the expected one.")

        return

    for i in range(len(Your_IntervalIndices)):
        if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your signal has different indices from the expected one.")
            return

    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your EncodedValues are different from the expected one.")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) >= 0.01:
            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your QuantizedValues are different from the expected one.")
            return

    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) >= 0.01:
            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your SampledError has different values from the expected one.")
            return

    messagebox.showinfo("Test Case Passed", " Test case passed successfully.")
>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
