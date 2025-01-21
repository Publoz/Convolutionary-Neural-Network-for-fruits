# Convolutionary-Neural-Network-for-fruits
A university project that looked at CNN at identifying 3 similiar fruits

<img width="509" alt="Screenshot 2025-01-21 at 10 36 12 PM" src="https://github.com/user-attachments/assets/c4c2634d-3748-4b6a-ab32-0d6bee665654" />



## Extract from the report regarding results

4. Results 
 
### 4.1 Baseline MLP 
 
My baseline MLP had a simple structure consisting of: 
- An input layer. 
- One hidden layer of 256 nodes and a rectifier function. 
- An output layer with 3 nodes using the softmax activation function. 
This is a very simple network, but it was just to gain a gauge on how our model might perform. The 
training data was also not augmented to keep things as simple as possible. After training for 17 epochs 
and reaching a training accuracy of 70%, I decided to stop training it. The training loss was slowly going 
down, but the validation accuracy continued to fluctuate with no apparent relation to how the training 
accuracy was. This whole process took close to 20 minutes. 
 
This model had a training accuracy of 70% and a validation accuracy of 47%. I used this model to predict 
the values on my test set and got the following results. 
 
Table with number of correct predictions from each class in the test set - MLP 
 
Cherry Strawberry Tomato 
Correct (/150) 71 37 104 
 
This is an overall accuracy of 47%. We can clearly see that this model has overfitted majorly with a drop-off of over 20% between the training and test set accuracies. We can also see that this model performs 
well on the tomato class getting nearly 70% correct. This is likely because the model is not balanced and 
constantly guesses Tomato for every instance. This is reinforced by the incredibly low score on the 
strawberry class, which is less than 1/3, meaning a model randomly guessing would likely do far better 
on this class. 
 
Considering that the model only took 20 minutes to train and had such a simple structure shows that we 
would expect our CNN to perform very strongly. This model did not perform well, but it still had a score 
that indicates it was not randomly guessing and gave us good insight into how strong our model should 
be. 
 
### 4.2 Complete CNN 
Using the model from the 30th epoch gave me a training accuracy of 85% and a validation accuracy of 
82%. Each epoch took around 8 minutes to complete which means that our 30 epochs took around 4 
hours overall. 
I used my trained model on the 450 images of the training set and got the following results. 
 
Table with number of correct predictions from each class in the test set - CNN 
 Cherry Strawberry Tomato 
Correct (/150) 120 118 115 
 
This was an overall accuracy of 353/450 (78.44%). Our 78% is similar to our validation accuracy and 
training accuracy which indicates that we have not overfitted. Furthermore, we can see that the 
distribution of the correctly predicted classes in the test set is also similar. This highlights that our model 
is well balanced and does not favour a single class or just predict one class constantly. 
