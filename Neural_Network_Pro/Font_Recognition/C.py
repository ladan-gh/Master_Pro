#-------train network with C---------------------------------------------------
#******train C1*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * C1[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_03 = 1

    elif y_in < 0:
        y_out_03 = -1

    else:
        y_out_03 = 0

    #-------------------------------
    if y_out_03 != t[0]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * C1[i][j] * t[0])
                b = b + (learning_rate * t[0])

    else:
        break

#******train C2*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * C2[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_03 = 1

    elif y_in < 0:
        y_out_03 = -1

    else:
        y_out_03 = 0

    #-------------------------------
    if y_out_03 != t[0]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * C2[i][j] * t[0])
                b = b + (learning_rate * t[0])

    else:
        break

#******train C3*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * C3[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_03 = 1

    elif y_in < 0:
        y_out_03 = -1

    else:
        y_out_03 = 0

    #-------------------------------
    if y_out_03 != t[0]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * C3[i][j] * t[0])
                b = b + (learning_rate * t[0])

    else:
        break

#***************train A1***********
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * A1[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_01 = 1

    elif y_in < 0:
        y_out_01 = -1

    else:
        y_out_01 = 0

    #-------------------------------
    if y_out_01 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] + (learning_rate * A1[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#************train A2**********************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * A2[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_01 = 1

    elif y_in < 0:
        y_out_01 = -1

    else:
        y_out_01 = 0

    #-------------------------------
    if y_out_01 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * A2[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#*********train A3*******************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * A3[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_01 = 1

    elif y_in < 0:
        y_out_01 = -1

    else:
        y_out_01 = 0

    #-------------------------------
    if y_out_01 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * A3[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train B1*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * B1[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_02 = 1

    elif y_in < 0:
        y_out_02 = -1

    else:
        y_out_02 = 0

    #-------------------------------
    if y_out_02 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * B1[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train B2*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * B2[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_02 = 1

    elif y_in < 0:
        y_out_02 = -1

    else:
        y_out_02 = 0

    #-------------------------------
    if y_out_02 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * B2[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train B3*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * B3[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_02 = 1

    elif y_in < 0:
        y_out_02 = -1

    else:
        y_out_02 = 0

    #-------------------------------
    if y_out_02 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * B3[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train E1*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * E1[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_04 = 1

    elif y_in < 0:
        y_out_04 = -1

    else:
        y_out_04 = 0

    #-------------------------------
    if y_out_04 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * E1[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train E2*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * E2[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_04 = 1

    elif y_in < 0:
        y_out_04 = -1

    else:
        y_out_04 = 0

    #-------------------------------
    if y_out_04 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * E2[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break

#******train E3*****************
while True:
    #col:7  and  row:9
    sum_ = 0
    for i in range(0, 9):
        for j in range(0, 7):
            sum_ += w_03[i][j] * E3[i][j]

    y_in  = b + sum_

    #-------------------------------
    if y_in > 0:
        y_out_04 = 1

    elif y_in < 0:
        y_out_04 = -1

    else:
        y_out_04 = 0

    #-------------------------------
    if y_out_04 != t[1]:# Occur an Error
        for i in range(0, 9):
            for j in range(0, 7):
                w_03[i][j] = w_03[i][j] +(learning_rate * E3[i][j] * t[1])
                b = b + (learning_rate * t[1])

    else:
        break
