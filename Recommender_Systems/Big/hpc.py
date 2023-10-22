import pandas as pd
x = [1,2,3,4]
df = pd.DataFrame(x)
df.to_csv('/home/401156007/Master/x.csv')

#-------------------------------------------------
import matplotlib.pyplot as plt

df = pd.read_csv('/home/401156007/Master/x.csv')

# Select the second column of the DataFrame
col = df.iloc[:, 1]

# Create a bar plot of the column
col.plot(kind='bar')

# Set the title of the plot
plt.title('Bar Plot for Second Column')

# Set the labels for the x-axis and y-axis
plt.xlabel('Index')
plt.ylabel('Value')

plt.savefig('/home/401156007/Master/photo.jpg')

# Display the plot
plt.show()