from sklearn.tree import DecisionTreeRegressor

# Initialize an empty list to store the training data
X_train = []
y_train = []

# Create and train the decision tree model
model = DecisionTreeRegressor()

while True:
    # Get input data from the user
    number_to_predict = input("Enter a number between 0 and 36 (or -1 to stop): ")

    if number_to_predict == "-1":
        break

    if not number_to_predict.isdigit():
        print("Invalid input. Please enter a valid number.")
        continue

    number_to_predict = int(number_to_predict)

    if number_to_predict < 0 or number_to_predict > 36:
        print("Invalid input. Number must be between 0 and 36.")
        continue

    # Store the input data and corresponding output
    X_train.append([number_to_predict])
    y_train.append(number_to_predict)

    # Train the model with the updated data
    model.fit(X_train, y_train)

    # Predict the next possible numbers based on the updated model
    next_numbers = list(range(number_to_predict + 1, number_to_predict + 6))
    predictions = []
    for number in next_numbers:
        X_test = [[number]]
        predicted_number = model.predict(X_test)
        predictions.append(int(max(0, min(36, predicted_number))))

    print("Predicted next 5 numbers based on the updated model:", predictions)
    print()
