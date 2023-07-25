#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
fig, ax = plt.subplots()
people = ["Farrah", "Fred", "Felicia"]
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruits = ['apples', 'bananas', 'oranges', 'peaches']

bottom = np.zeros(len(people))

for i, row in enumerate(fruit):
    ax.bar(people, row, bottom=bottom,
           color=colors[i], label=fruits[i], width=0.5)
    bottom += row

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_ylim([0, 80])
ax.set_yticks(np.arange(0, 81, 10))
ax.legend()
plt.show()