# matrix = [[0, 0], [0, 0]]
# actual_yes, actual_no, predicted_yes = 0, 0, 0

# for i in range(len(y_test)):
#     if y_test[i][0] == 1:
#         actual_yes += 1
#     elif y_test[i][0] == 0:
#         actual_no += 1
#     if pred_output[i][0] > 0.5:
#         predicted_yes += 1
#     x, y = 0, 0
#     if y_test[i][0] > 0.5:
#         x = 1
#     else:
#         x = 0
#     if pred_output[i][0] > 0.5:
#         y = 1
#     else:
#         y = 0
#     matrix[x][y] += 1

# TP = matrix[1][1]
# TN = matrix[0][0]
# FP = matrix[0][1]
# FN = matrix[1][0]

# total = len(y_test)
# accuracy = (TP + TN) / total
# misclassfication = (FP + FN) / total
# recall = TP / actual_yes
# specificity = TN / actual_no
# precision = TP / predicted_yes
# f_score = 2 * ((recall * precision) / (recall + precision))

# print("Confusion Matrix:", matrix)
# print("Accuracy: ", accuracy)
# print("Misclassfication Rate: ", misclassfication)
# print("True Positive Rate (Recall): ", recall)
# print("True Negative Rate (Specificity): ", specificity)
# print("Precision: ", precision)
# print("F Score: ", f_score)
