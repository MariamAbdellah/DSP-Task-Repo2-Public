<<<<<<< HEAD
=======
import tkinter as tk
from tkinter import messagebox

>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
<<<<<<< HEAD
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
=======
        messagebox.showerror("Quantization Test Error",
                             " Test case failed: your signal has a different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            messagebox.showerror("Quantization Test Error",
                                 " Test case failed: your EncodedValues are different from the expected ones")
>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
<<<<<<< HEAD
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    print("QuantizationTest1 Test case passed successfully")
=======
            messagebox.showerror("Quantization Test Error",
                                 " Test case failed: your QuantizedValues differ from the expected ones")
            return
    messagebox.showinfo("Quantization Test Result", " Test case passed successfully")
>>>>>>> 59bcdbdb4bc6be27f2e11117cfe0d9729daf29fa
