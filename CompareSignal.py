import tkinter as tk
from tkinter import messagebox
def CompareSignal(file_name, Your_EncodedValues, Your_Values):
    expectedIndices=[]
    expectedValues=[]
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
                V2=int(L[0])
                V3=float(L[1])
                expectedIndices.append(V2)
                expectedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedIndices)) or (len(Your_Values) != len(expectedValues))):
        messagebox.showerror("Test Case Failed",
                             " Test case failed, your signal has a different length from the expected one.")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedIndices[i]):

            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your Values are different from the expected one.")
            return
    for i in range(len(expectedValues)):
        if abs(Your_Values[i] - expectedValues[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Test Case Failed",
                                 "Test case failed, your Values have different values from the expected one.")
            return
    messagebox.showinfo("Test Case Passed", " Test case passed successfully.")

