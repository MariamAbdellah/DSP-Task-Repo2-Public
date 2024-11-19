def CompareSignal(file_name, YourValues):
    expectedValues = []
    expectedQuantizedValues = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V2 = str(L[0])
                V3 = float(L[1])
                expectedQuantizedValues.append(V2)
                expectedValues.append(V3)
                line = f.readline()
            else:
                break
    if len(YourValues) != len(expectedValues):
        print("Test case failed, your signal have different length from the expected one")
        return
    # for i in range(len(YourValues)):
    #     if YourValues[i] != expectedValues[i]:
    #         print("Test case failed, your EncodedValues have different EncodedValues from the expected one")
    #         return
    for i in range(len(expectedQuantizedValues)):
        if abs(YourValues[i] - expectedValues[i]) < 0.01:
            continue
        else:
            print(
                "Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("Test case passed successfully")
