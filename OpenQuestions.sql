insert into topic (name) values ('Programming Theory');
insert into topic (name) values ('Machine Learning');
insert into topic (name) values ('Databases');
insert into topic (name) values ('Computer Networks');
insert into topic (name) values ('Operating Systems');
insert into topic (name) values ('Data Structures');

-- PROGRAMMING THEORY

insert into question (id, text, topic_id)
values (1, 'What is the difference between a compiler and an interpreter?', 1);
insert into answer (text, question_id) 
values ('A compiler translates the entire source code into machine code before execution, while an interpreter translates and executes the source code line by line.', 1);
insert into answer (text, question_id) 
values ('A compiler generates an executable file that can be run independently, while an interpreter requires an interpreter program to execute the code.', 1);
insert into answer (text, question_id) 
values ('A compiler checks for syntax errors and generates error messages before execution, while an interpreter only generates error messages when it encounters an error during execution.', 1);
insert into answer (text, question_id) 
values ('A compiler produces faster code than an interpreter since the entire code is translated at once, while an interpreter is slower since it needs to translate and execute each line of code separately.', 1);
insert into answer (text, question_id) 
values ('A compiler is typically used for statically typed languages, while an interpreter is used for dynamically typed languages.', 1);

insert into question (id, text, topic_id)
values (2, 'What is object-oriented programming?', 1);
insert into answer (text, question_id) 
values ('Object-oriented programming (OOP) is a programming paradigm that focuses on creating objects that encapsulate data and behavior, and interact with each other through methods and messages.', 2);
insert into answer (text, question_id) 
values ('OOP is a programming methodology that emphasizes the use of classes and objects to model real-world entities, and allows for code reusability, modularity, and maintainability.', 2);
insert into answer (text, question_id) 
values ('OOP is a programming approach that enables developers to organize code into reusable, self-contained modules called classes, which can be instantiated to create objects with specific attributes and methods.', 2);
insert into answer (text, question_id) 
values ('OOP is a programming technique that promotes the use of inheritance, polymorphism, and encapsulation to build complex systems that are easy to extend, modify, and maintain.', 2);
insert into answer (text, question_id) 
values ('OOP is a programming philosophy that emphasizes the importance of data abstraction, encapsulation, and modularization to achieve code clarity, flexibility, and scalability.', 2);

insert into  question (id, text, topic_id)
values (3, 'What is version control?', 1);
insert into answer (text, question_id)
values ('Version control is the management of changes to documents, computer programs, large web sites, and other collections of information.', 3);
insert into answer (text, question_id)
values ('Version control is a system that records changes to a file or set of files over time so that you can recall specific versions later.', 3);
insert into answer (text, question_id)
values ('Version control is a software tool that helps developers manage changes to source code over time, allowing them to track and collaborate on code changes, merge changes from multiple contributors, and revert to previous versions if necessary.', 3);
insert into answer (text, question_id)
values ('version control refers to the process of managing different versions or drafts of a document, ensuring that the correct version is being used and that changes are tracked and recorded.s', 3);
insert into answer (text, question_id)
values ('version control can refer to the management of different versions of product designs or specifications, ensuring that the correct version is being used in production and that changes are properly documented and tracked.', 3);

insert into question (id, text, topic_id)
values (4, 'What do you understand by a class and a superclass?', 1);
insert into answer (text, question_id)
values ('A class is a blueprint or template for creating objects in object-oriented programming. It defines the properties and behaviors that objects of that class will have. A superclass, on the other hand, refers to a class that is higher in the inheritance hierarchy. It serves as a base class from which other classes, called subclasses, can inherit properties and behaviors.', 4);
insert into answer (text, question_id)
values ('In object-oriented programming, a class represents a collection of objects with similar characteristics and behaviors. It encapsulates data and methods that define the behavior of those objects. A superclass is a class that other classes can inherit from. It provides common attributes and behaviors that subclasses can reuse or override.', 4);
insert into answer (text, question_id)
values ('A class is a fundamental concept in object-oriented programming that defines a blueprint for creating objects. It includes attributes (data) and methods (functions) that describe the behavior of objects created from that class. A superclass is a class that acts as a parent or base class for other classes. It provides common attributes and behaviors that subclasses can inherit and extend.', 4);
insert into answer (text, question_id)
values ('In the context of object-oriented programming, a class represents a user-defined data type that encapsulates data and methods. It defines the structure and behavior of objects created from that class. A superclass refers to a class that is higher in the inheritance hierarchy. It provides a set of attributes and behaviors that subclasses can inherit and modify as needed.', 4);
insert into answer (text, question_id)
values ('A class is a blueprint for creating objects in object-oriented programming. It defines the properties (attributes) and behaviors (methods) that objects of that class will have. A superclass, also known as a parent class or base class, is a class from which other classes can inherit properties and behaviors. It allows for code reusability and promotes the concept of inheritance in object-oriented programming.', 4);

insert into question (id, text, topic_id)
values (5, 'Explain the concept of Big O notation and its significance in computer science.', 1);
insert into answer (text, question_id) values 
('Big O notation is a mathematical notation used to describe the efficiency of an algorithm by expressing its worst-case time complexity. It helps computer scientists analyze and compare algorithms based on their scalability and performance characteristics.', 5);
insert into answer (text, question_id) values 
('In computer science, Big O notation is a way of describing how the runtime or space requirements of an algorithm grow relative to the size of the input. It allows us to categorize algorithms into classes and understand their efficiency in terms of time and space usage.', 5);
insert into answer (text, question_id) values 
('The concept of Big O notation is crucial in computer science as it provides a standardized way to express the efficiency of algorithms. By using Big O notation, we can make informed decisions about which algorithms to choose based on their efficiency, enabling us to optimize our code and improve overall system performance.', 5);
insert into answer (text, question_id) values 
('Big O notation is a tool used in computer science to analyze and compare algorithms based on their time complexity. It allows us to understand how the runtime of an algorithm increases as the input size grows. This knowledge is essential for designing efficient algorithms and predicting their performance in real-world scenarios.', 5);
insert into answer (text, question_id) values 
('The significance of Big O notation in computer science lies in its ability to provide a language for discussing and reasoning about algorithmic efficiency. By using Big O notation, we can communicate and understand the scalability and performance characteristics of algorithms, enabling us to make informed decisions when solving computational problems.', 5);

insert into question (id, text, topic_id)
values (6, 'What is the difference between low-level and high-level programming languages?', 1);
insert into answer (text, question_id) values 
('Low-level programming languages, such as assembly language, are closer to machine code and provide direct control over hardware resources. High-level programming languages, such as Python or Java, are more abstract and provide built-in functions and libraries for easier development.', 6);
insert into answer (text, question_id) values 
('Low-level programming languages are more hardware-oriented and require a deep understanding of computer architecture, while high-level programming languages are more user-friendly and focus on solving problems without needing to worry about low-level details.', 6);
insert into answer (text, question_id) values 
('Low-level programming languages offer more control and efficiency in terms of memory management and performance optimization, whereas high-level programming languages prioritize productivity and ease of use by providing features like automatic memory management and simplified syntax.', 6);
insert into answer (text, question_id) values 
('Low-level programming languages allow for fine-grained control over the computer resources, making them suitable for tasks that require high performance or low-level system programming. High-level programming languages are better suited for general-purpose application development and offer higher productivity with their abstraction layers.', 6);
insert into answer (text, question_id) values 
('Low-level programming languages provide direct access to hardware components and system resources, allowing for precise control and optimization. High-level programming languages provide a higher level of abstraction, enabling developers to focus on problem-solving rather than low-level implementation details.', 6);

insert into question (id, text, topic_id)
values (7, 'What is software testing and why do we need it?', 1);
insert into answer (text, question_id) values 
('Software testing is the process of evaluating a software system to identify any defects, errors, or bugs in its functionality. We need software testing to ensure that the software meets the desired requirements, functions as intended, and provides a reliable and high-quality user experience.', 7);
insert into answer (text, question_id) values 
('Software testing is the practice of systematically verifying and validating a software application to ensure that it meets the specified requirements and performs as expected. We need software testing to minimize the risk of software failures, enhance the overall quality of the product, and increase user satisfaction by delivering a reliable and error-free software solution.', 7);
insert into answer (text, question_id) values 
('Software testing is the systematic process of assessing the functionality, performance, and usability of a software application to identify any issues or defects that may impact its performance or user experience. We need software testing to mitigate risks, enhance software reliability, and ensure that the product meets the expectations of end users and stakeholders.', 7);
insert into answer (text, question_id) values 
('Software testing is the practice of executing a software application with the intention of finding defects or errors in its functionality. We need software testing to ensure that the software meets the desired quality standards, functions correctly in different scenarios, and provides a stable and secure environment for users.', 7);
insert into answer (text, question_id) values 
('Software testing is an essential step in the software development lifecycle that involves executing a program or system to identify any discrepancies between expected and actual results. We need software testing to detect and fix defects early in the development process, improve software reliability, and deliver a robust and high-quality product to end users.', 7);

-- MACHINE LEARNING

insert into question (id, text, topic_id) 
values (8, 'What is a linear regression?', 2);
insert into answer (text, question_id) 
values ('Linear Regression is a supervised algorithm in Machine learning that supports finding the linear correlation among variables.', 8);
insert into answer (text, question_id) 
values ('Linear regression is a machine learning algorithm that solves regression problem', 8);
insert into answer (text, question_id) 
values ('Linear regression is a statistical method used to model the relationship between two variables by fitting a linear equation to the observed data.', 8);
insert into answer (text, question_id) 
values ('Linear regression is a statistical method used to analyze the relationship between two variables, where one variable is considered the dependent variable and the other is considered the independent variable. The method involves finding the line of best fit that represents the relationship between the two variables, allowing for predictions to be made about the dependent variable based on changes in the independent variable.', 8);
insert into answer (text, question_id) 
values ('Linear regression is the most basic and commonly used predictive analysis. Regression estimates are used to describe data and to explain the relationship.', 8);

insert into question (id, text, topic_id) 
values (9, 'What is a decision tree?', 2);
insert into answer (text, question_id)
values ('A decision tree is a graphical representation of all possible solutions to a decision based on certain conditions or variables.', 9);
insert into answer (text, question_id)
values ('A decision tree is a machine learning algorithm that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.', 9);
insert into answer (text, question_id)
values ('A decision tree is a statistical method used to model the relationship between two variables by fitting a tree-like structure to the observed data.', 9);
insert into answer (text, question_id)
values ('A decision tree is a tool used by lumberjacks to decide which trees to cut down in a forest.', 9);
insert into answer (text, question_id)
values ('A decision tree is a visual representation of a set of rules that can be used to make decisions based on data.', 9);

insert into question (id, text, topic_id) 
values (10, 'What is supervised and unsupervised machine learning?', 2);
insert into answer (text, question_id)
values ('Supervised machine learning is a type of machine learning where the model is trained on labeled data, meaning it is provided with input data and corresponding output labels. The model learns from this labeled data to make predictions or classifications on new, unseen data.', 10);
insert into answer (text, question_id)
values ('Unsupervised machine learning is a type of machine learning where the model is trained on unlabeled data, meaning it is provided with only input data without any corresponding output labels. The model learns patterns, structures, or relationships within the data to uncover hidden insights or clusters.', 10);
insert into answer (text, question_id)
values ('Supervised machine learning involves a training phase where the model learns from labeled examples and a prediction phase where it applies this learned knowledge to make predictions on new data. Unsupervised machine learning, on the other hand, focuses solely on finding patterns or structures within the input data without any predefined output labels.', 10);
insert into answer (text, question_id)
values ('In supervised machine learning, the model learns from a known dataset with labeled examples, enabling it to make predictions or classifications on new, unseen data. In unsupervised machine learning, the model explores the data itself to discover patterns, relationships, or groupings without any prior knowledge or guidance.', 10);
insert into answer (text, question_id)
values ('Supervised machine learning requires human intervention to provide labeled training data, whereas unsupervised machine learning does not rely on labeled data and instead relies on the inherent structure or patterns within the data to learn and make inferences.', 10);

insert into question (id, text, topic_id) 
values (11, 'What is gradient descent? How does it work?', 2);
insert into answer (text, question_id)
values ('Gradient descent is an optimization algorithm used in machine learning to minimize the error or cost function of a model. It works by iteratively adjusting the parameters of the model in the direction of steepest descent of the cost function, gradually reaching the optimal set of parameters that minimize the error.', 11);
insert into answer (text, question_id)
values ('Gradient descent is a mathematical algorithm that is commonly used in machine learning to optimize models. It works by calculating the gradient of the cost function with respect to the parameters of the model and then updating the parameters in the opposite direction of the gradient, iteratively moving towards the minimum of the cost function.', 11);
insert into answer (text, question_id)
values ('Gradient descent is a technique used to optimize models by iteratively adjusting the parameters of the model in order to minimize the error or loss function. It works by calculating the gradient of the loss function with respect to each parameter and updating the parameters in small steps, moving in the direction that reduces the loss.', 11);
insert into answer (text, question_id)
values ('Gradient descent is an iterative optimization algorithm used in machine learning to find the optimal values for a parameters of the model. It works by calculating the gradient of the cost function with respect to each parameter and updating the parameters in small increments, moving in the direction that minimizes the cost function.', 11);
insert into answer (text, question_id)
values ('Gradient descent is a widely used optimization algorithm in machine learning that aims to minimize the error or loss of a model. It works by calculating the gradients of the cost function with respect to the model parameters and updating these parameters in small steps, gradually approaching the optimal values that minimize the error.', 11);

insert into question (id, text, topic_id) 
values (12, 'What is the bias-variance trade-off?', 2);
insert into answer (text, question_id)
values ('The bias-variance trade-off refers to the relationship between the bias and variance of a model. Bias measures how far off the predictions of a model are from the true values, while variance measures the variability of the model predictions for different training sets. The trade-off occurs because reducing bias often increases variance, and vice versa. The goal is to find the right balance that minimizes both bias and variance, leading to a model that generalizes well to unseen data.', 12);
insert into answer (text, question_id)
values ('The bias-variance trade-off is a fundamental concept in machine learning that deals with the trade-off between a ability of the model to fit the training data and its ability to generalize to new, unseen data. Bias refers to the error introduced by approximating a real-world problem with a simplified model, while variance refers to the sensitivity of the model predictions to fluctuations in the training data. Finding the optimal trade-off is crucial for building models that are both accurate and robust.', 12);
insert into answer (text, question_id)
values ('The bias-variance trade-off refers to the trade-off between a model ability to capture complex patterns in the data (low bias) and its susceptibility to overfitting (high variance). A model with high bias may oversimplify the problem and underfit the data, while a model with high variance may overfit the data and fail to generalize well. Balancing bias and variance is essential for creating models that strike a good compromise between underfitting and overfitting.', 12);
insert into answer (text, question_id)
values ('The bias-variance trade-off is the delicate balance between a ability of the model to capture the true underlying patterns in the data (low bias) and its tendency to be overly influenced by random noise in the training set (high variance). Models with high bias may be too simplistic and miss important relationships, while models with high variance may be too complex and overfit the training data. Achieving an optimal trade-off is crucial for building models that are both accurate and reliable.', 12);
insert into answer (text, question_id)
values ('The bias-variance trade-off is a key concept in machine learning that involves finding the right level of model complexity. Bias refers to the error introduced by making overly simplistic assumptions about the underlying data, while variance refers to the error introduced by making complex models that are sensitive to small fluctuations in the training data. The trade-off arises because reducing bias often increases variance, and vice versa. Striking the right balance is important for developing models that generalize well to unseen data.', 12);

insert into question (id, text, topic_id) 
values (13, 'Why do we need to split our data into three parts: train, validation, and test?', 2);
insert into answer (text, question_id)
values ('We need to split our data into three parts - train, validation, and test - in order to properly evaluate and fine-tune our model. The training set is used to train the model, the validation set is used to tune hyperparameters and select the best model, and the test set is used to assess the final performance of the selected model on unseen data.', 13);
insert into answer (text, question_id)
values ('Splitting our data into train, validation, and test sets allows us to assess the performance of our model on unseen data. The training set is used to fit the model, the validation set is used to make adjustments and optimize the model performance, and the test set is used as a final evaluation to ensure that our model generalizes well.', 13);
insert into answer (text, question_id)
values ('By splitting our data into train, validation, and test sets, we can avoid overfitting our model to the training data. The training set is used to learn the parameters of the model, the validation set is used to assess its performance and make adjustments, and the test set is used as an unbiased evaluation of the final model performance on unseen data.', 13);
insert into answer (text, question_id)
values ('Splitting our data into three parts - train, validation, and test - allows us to assess the generalization ability of our model. The training set is used to train the model, the validation set is used to tune its hyperparameters and select the best configuration, and the test set is used to provide an unbiased estimate of how well our model will perform on new, unseen data.', 13);
insert into answer (text, question_id)
values ('The division of data into train, validation, and test sets helps us evaluate the performance and robustness of our model. The training set is used for model training, the validation set is used for model selection and tuning, and the test set is used for a final assessment of how well our model performs on completely new data. This separation ensures that we have an unbiased evaluation of our model performance.', 13);

insert into question (id, text, topic_id) 
values (14, 'Can you explain how cross-validation works?', 2); 
insert into answer (text, question_id)
values ('Cross-validation is a technique used to assess the performance of a model by dividing the data into multiple subsets or "folds." Each fold is used as a validation set while the remaining folds are used for training. This process is repeated multiple times, with each fold serving as the validation set once. The results are then averaged to provide a more robust evaluation of the model performance.', 14);
insert into answer (text, question_id)
values ('Cross-validation involves splitting the data into multiple subsets or folds. The model is trained on a combination of these folds and evaluated on the remaining fold. This process is repeated multiple times, with each fold serving as the validation set once. The performance metrics from each iteration are then averaged to obtain a more reliable estimate of the model performance.', 14);
insert into answer (text, question_id)
values ('Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting different training and validation set, in order to reduce the bias that you would have by selecting only one validation set.', 14);
insert into answer (text, question_id)
values ('Cross-validation works by splitting the data into multiple subsets or folds. One fold is held out as the validation set, and the remaining folds are used for training. This process is repeated multiple times, with each fold serving as the validation set once. The performance is then evaluated based on the average performance across all iterations, providing a more robust assessment of its generalization ability.', 14);
insert into answer (text, question_id)
values ('Cross-validation is a technique used to evaluate the performance of a model by repeatedly partitioning the data into training and validation sets. The model is trained on the training set and evaluated on the validation set, and this process is repeated multiple times with different partitions. The average performance across all iterations provides an estimate of how well the model will perform on unseen data.', 14);

insert into question (id, text, topic_id) 
values (15, 'What is sigmoid? What does it do?', 2); 
insert into answer (text, question_id)
values ('Sigmoid is a mathematical function that maps any real-valued number to a value between 0 and 1. It is commonly used as an activation function in neural networks. The sigmoid function is useful for tasks such as binary classification, where the output needs to be interpreted as a probability or a confidence score.', 15);
insert into answer (text, question_id)
values ('Sigmoid is a type of activation function used in machine learning and neural networks. It transforms the input into a range between 0 and 1, which can be interpreted as a probability. Sigmoid is particularly useful for tasks that require binary classification, as it can map the output to a probability of belonging to a certain class.', 15);
insert into answer (text, question_id)
values ('Sigmoid is a mathematical function that maps any real number to a value between 0 and 1. It is commonly used as an activation function in deep learning models. The sigmoid function is advantageous because it introduces non-linearity into the model, allowing it to learn more complex patterns and relationships in the data.', 15);
insert into answer (text, question_id)
values ('Sigmoid is a mathematical function that squashes the input values into a range between 0 and 1. It is often used as an activation function in artificial neural networks, where it helps in introducing non-linearity and enabling the network to learn complex patterns. The sigmoid function is especially useful in binary classification tasks, where it can output probabilities or confidence scores.', 15);
insert into answer (text, question_id)
values ('Sigmoid is a type of activation function that transforms the input values into a range between 0 and 1. It is commonly used in machine learning algorithms, particularly in logistic regression and artificial neural networks. The sigmoid function allows for non-linear transformations, making it suitable for capturing complex relationships between input variables and the output predictions.', 15);

insert into question (id, text, topic_id) 
values (16, 'What is classification? Which models would you use to solve a classification problem?', 2); 
insert into answer (text, question_id)
values ('Classification is a machine learning task that involves assigning input data into predefined categories or classes. Models commonly used for classification problems include logistic regression, support vector machines, decision trees, random forests, and neural networks.', 16);
insert into answer (text, question_id)
values ('Classification is the process of categorizing input data into distinct classes or categories based on their features. Popular models for solving classification problems include k-nearest neighbors, naive Bayes, decision trees, random forests, and gradient boosting algorithms like XGBoost or LightGBM.', 16);
insert into answer (text, question_id)
values ('Classification refers to the task of assigning input data points to predefined classes or categories. Some common models used for classification problems are logistic regression, support vector machines, k-nearest neighbors, decision trees, and ensemble methods like AdaBoost or RandomForest.', 16);
insert into answer (text, question_id)
values ('Classification problems are problems in which our prediction space is discrete, i.e. there is a finite number of values the output variable can be. Some models which can be used to solve classification problems are: logistic regression, decision tree, random forests, multi-layer perceptron, one-vs-all, amongst others.', 16);
insert into answer (text, question_id)
values ('Classification involves assigning input data into predefined classes or categories based on their characteristics. To solve a classification problem, one can utilize models such as logistic regression, support vector machines, k-nearest neighbors, decision trees, and ensemble methods like AdaBoost or RandomForest.', 16);

insert into question (id, text, topic_id) 
values (17, 'Tell me about the most useful machine learning evaluation metrics. Explain when do we use one and why.', 2); 
insert into answer (text, question_id)
values ('One of the most useful machine learning evaluation metrics is accuracy, which measures the percentage of correctly classified instances. It is commonly used when the class distribution is balanced and all classes are equally important. However, accuracy can be misleading when there is class imbalance or when the cost of misclassifying certain classes is higher than others.', 17);
insert into answer (text, question_id)
values ('Precision and recall are important metrics for evaluating machine learning models. Precision measures the proportion of correctly predicted positive instances out of all predicted positives, while recall measures the proportion of correctly predicted positive instances out of all actual positives. Precision is useful when the focus is on minimizing false positives, while recall is important when the goal is to minimize false negatives.', 17);
insert into answer (text, question_id)
values ('F1 score is a metric that combines precision and recall into a single value. It calculates the harmonic mean of precision and recall, providing a balanced evaluation of a model performance. The F1 score is commonly used when there is an uneven class distribution or when both false positives and false negatives need to be considered.', 17);
insert into answer (text, question_id)
values ('Area under the receiver operating characteristic curve (AUC-ROC) is a widely used metric for evaluating binary classification models. It measures the ability of the model to distinguish between positive and negative instances across different probability thresholds. AUC-ROC is particularly useful when the class distribution is imbalanced or when the cost of misclassification varies across different thresholds.', 17);
insert into answer (text, question_id)
values ('Mean squared error (MSE) is a commonly used evaluation metric for regression problems. It measures the average squared difference between the predicted and actual values. MSE is useful when the goal is to minimize the overall prediction error and when outliers have a significant impact on the model performance. However, MSE can be sensitive to outliers and may not provide a clear interpretation of the error in some cases.', 17);

insert into question (id, text, topic_id) 
values (18, 'What do we do with categorical variables? Explain your answer in detail', 2);
insert into answer (text, question_id)
values ('One approach to handling categorical variables is to encode them as numerical values using techniques such as one-hot encoding or label encoding. This is necessary when using machine learning algorithms that require numerical input, as categorical variables cannot be directly used. One-hot encoding creates binary variables for each category, while label encoding assigns a unique numerical value to each category. The choice between the two depends on the specific algorithm being used and the nature of the categorical variable.', 18);
insert into answer (text, question_id)
values ('Another option is to use target encoding, which replaces each category with the mean target value of the corresponding instances. This is useful when the categorical variable has a strong relationship with the target variable and encoding it as numerical values can provide more meaningful information to the model. However, target encoding may introduce bias if there are too few instances for certain categories.', 18);
insert into answer (text, question_id)
values ('In some cases, it may be appropriate to treat categorical variables as ordinal variables, where the categories have a natural order or hierarchy. This can be done by assigning numerical values to the categories based on their order. For example, in a survey question about satisfaction levels (e.g., "very unsatisfied," "unsatisfied," "neutral," "satisfied," "very satisfied"), assigning values from 1 to 5 can capture the ordinal nature of the variable. This approach should be used when there is a clear ordering among the categories.', 18);
insert into answer (text, question_id)
values ('When dealing with high-cardinality categorical variables (i.e., variables with a large number of unique categories), it may be necessary to group similar categories together to reduce dimensionality. This can be done through techniques such as binning or clustering. Binning involves grouping categories into broader intervals based on their similarity, while clustering uses unsupervised learning algorithms to automatically identify groups of similar categories. This is useful when the high-cardinality variable adds noise or complexity to the model.', 18);
insert into answer (text, question_id)
values ('In some cases, it may be appropriate to completely remove categorical variables from the model if they do not provide any meaningful information or if they introduce too much noise. This decision should be based on domain knowledge and the specific problem at hand. Removing categorical variables can simplify the model and reduce the risk of overfitting, but it should be done cautiously to ensure important information is not lost.', 18);

insert into question (id, text, topic_id) 
values (19, 'What is regularization? Why do we need it and which regularization techniques do you know?', 2);
insert into answer (text, question_id)
values ('Regularization is a technique used in machine learning to prevent overfitting, which occurs when a model becomes too complex and performs well on the training data but poorly on new, unseen data. Regularization adds a penalty term to the loss function, discouraging the model from fitting the training data too closely. This helps to generalize the model and improve its performance on unseen data. Regularization techniques include L1 regularization (Lasso), which adds the absolute values of the coefficients to the loss function, encouraging sparsity and feature selection; L2 regularization (Ridge), which adds the squared values of the coefficients to the loss function, encouraging smaller coefficients; and Elastic Net regularization, which combines both L1 and L2 regularization. These techniques help to control the complexity of the model and prevent overfitting.', 19);
insert into answer (text, question_id)
values ('Regularization is a method used in machine learning to reduce the complexity of a model and prevent overfitting. Overfitting occurs when a model becomes too specialized in fitting the training data and performs poorly on new data. Regularization techniques introduce a penalty term to the loss function, discouraging the model from learning complex patterns that may be specific to the training data but not generalizable. Some commonly used regularization techniques include dropout, which randomly sets a fraction of the input units to zero during training, forcing the model to learn redundant representations; early stopping, which stops training when the performance on a validation set starts to deteriorate, preventing the model from further overfitting; and weight decay, which adds a regularization term to the loss function that penalizes large weights, encouraging smaller and more generalizable weights.', 19);
insert into answer (text, question_id)
values ('Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. Overfitting occurs when a model learns to fit the noise or random fluctuations in the training data, leading to poor performance on new data. Regularization helps to control the complexity of the model and improve its generalization ability. Some common regularization techniques include ridge regression, which adds the squared sum of the coefficients to the loss function; lasso regression, which adds the sum of the absolute values of the coefficients; and elastic net regression, which combines both ridge and lasso regularization. These techniques help to shrink the coefficients and reduce the impact of irrelevant features, improving the model ability to generalize to new data.', 19);
insert into answer (text, question_id)
values ('Regularization is a technique used in machine learning to prevent overfitting and improve the generalization ability of a model. Overfitting occurs when a model becomes too complex and fits the noise or random variations in the training data, resulting in poor performance on new data. Regularization techniques add a penalty term to the loss function, discouraging the model from learning complex patterns that may be specific to the training data. Some commonly used regularization techniques include dropout, which randomly sets a fraction of the input units to zero during training, forcing the model to learn redundant representations; L1 regularization (Lasso), which adds the sum of the absolute values of the coefficients to the loss function, encouraging sparsity and feature selection; and L2 regularization (Ridge), which adds the squared sum of the coefficients to the loss function, encouraging smaller coefficients. These techniques help to control the complexity of the model and prevent overfitting.', 19);
insert into answer (text, question_id)
values ('Regularization is a technique used in machine learning to prevent overfitting and improve the generalization ability of a model. Overfitting occurs when a model becomes too complex and fits the noise or random variations in the training data, leading to poor performance on new data. Regularization techniques introduce a penalty term to the loss function, discouraging the model from learning overly complex patterns that may not generalize well. Some common regularization techniques include early stopping, which stops training when the performance on a validation set starts to deteriorate, preventing the model from further overfitting; weight decay, which adds a regularization term to the loss function that penalizes large weights, encouraging smaller and more generalizable weights; and dropout, which randomly sets a fraction of the input units to zero during training, forcing the model to learn redundant representations. These techniques help to control the complexity of the model and improve its ability to generalize to new data.', 19);

insert into question (id, text, topic_id) 
values (20, 'What is a random forest? How do we know how many trees will we need and how to select their depth?', 2);
insert into answer (text, question_id)
values ('A random forest is an ensemble learning method that combines multiple decision trees to make predictions. The number of trees needed in a random forest can be determined through techniques like cross-validation or grid search, where different numbers of trees are tested and evaluated based on their performance.', 20);
insert into answer (text, question_id)
values ('A random forest is a machine learning algorithm that uses a collection of decision trees to classify or regress data. The optimal number of trees in a random forest can be determined using techniques like out-of-bag error estimation or by analyzing the trade-off between model complexity and performance. The depth of the trees in a random forest can be selected based on the desired balance between bias and variance in the model.', 20);
insert into answer (text, question_id)
values ('Random forest is a popular ensemble learning technique that combines multiple decision trees to improve prediction accuracy. The number of trees required in a random forest can be determined using methods like cross-validation or by monitoring the performance metrics such as accuracy or mean squared error. The depth of the trees can be selected by considering factors like the complexity of the problem, available computational resources, and the risk of overfitting.', 20);
insert into answer (text, question_id)
values ('A random forest is a machine learning algorithm that utilizes an ensemble of decision trees to make predictions. The number of trees to be included in the random forest can be determined through methods like bootstrap aggregating (bagging) or by using techniques like the Breiman-Cutler proximity measure. The depth of the trees can be selected by considering factors such as the complexity of the dataset, desired model interpretability, and computational constraints.', 20);
insert into answer (text, question_id)
values ('Random forest is a powerful algorithm that combines multiple decision trees to make accurate predictions. The number of trees required in a random forest can be determined by evaluating the out-of-bag error or using techniques like k-fold cross-validation. The selection of tree depth can be done by considering factors such as the complexity of the problem, the amount of available training data, and the desired trade-off between model complexity and performance.', 20);

-- DATABASES

insert into question (id, text, topic_id) 
values (21, 'What is a database?', 3);
insert into answer (text, question_id)
values ('A database is a collection of data that is organized and stored in a way that enables efficient retrieval and manipulation.', 21);
insert into answer (text, question_id)
values ('A database is a software system that allows for the storage, organization, and management of large amounts of data.', 21);
insert into answer (text, question_id)
values ('A database is a structured set of data that is stored electronically and can be accessed, managed, and updated as needed.', 21);
insert into answer (text, question_id)
values ('A database is a tool for storing and organizing information in a way that makes it easy to search, analyze, and retrieve.', 21);
insert into answer (text, question_id)
values ('A database is a repository of information that can be used to support business operations, research, analysis, and decision-making.', 21);

insert into question (id, text, topic_id) 
values (22, 'What is the difference between rank() and dense_rank() window functions in PostgreSQL?', 3);
insert into answer (text, question_id)
values ('The rank() function assigns a unique rank to each row in the result set, with gaps between ranks if there are ties. The dense_rank() function, on the other hand, assigns a unique rank to each row without any gaps, even if there are ties.', 22);
insert into answer (text, question_id)
values ('The rank() function skips ranks when there are ties, resulting in non-consecutive rank numbers. In contrast, the dense_rank() function does not skip ranks and assigns consecutive rank numbers to rows with ties.', 22);
insert into answer (text, question_id)
values ('When using the rank() function, if two or more rows have the same values and are assigned the same rank, the next rank will be skipped. In contrast, the dense_rank() function does not skip ranks and assigns the next consecutive rank to rows with the same values.', 22);
insert into answer (text, question_id)
values ('The rank() function can produce gaps in the ranking sequence when there are ties. In contrast, the dense_rank() function does not produce any gaps and ensures a continuous ranking sequence.', 22);
insert into answer (text, question_id)
values ('The rank() function is useful when you want to identify the position of a row within a result set, including any ties. The dense_rank() function, on the other hand, is useful when you want to assign a unique rank to each row without any gaps, regardless of ties.', 22);

insert into question (id, text, topic_id) 
values (23, 'What are the pros and cons of indexing in database?', 3);
insert into answer (text, question_id)
values ('Pro: Indexing improves the efficiency of data retrieval, allowing for faster query execution and enhanced system performance. Con: Indexing increases storage space requirements, which can be a concern for large databases with limited capacity.', 23);
insert into answer (text, question_id)
values ('Pro: Indexing helps organize and structure data, enabling faster searching and sorting based on specific criteria, improving data accessibility and analysis. Con: Indexing can lead to increased maintenance overhead, as indexes need to be updated whenever the underlying data is modified.', 23);
insert into answer (text, question_id)
values ('Pro: Indexing aids in enforcing data integrity and uniqueness, preventing duplicate or conflicting data from being inserted into the database. Con: Indexing can slow down data modification operations, as indexes need to be updated whenever new data is added or existing data is modified.', 23);
insert into answer (text, question_id)
values ('Pro: Indexing supports faster joins and aggregations, allowing for efficient querying and analysis of complex datasets involving multiple tables. Con: Improperly designed indexes can degrade database performance, leading to slower query execution and increased resource consumption.', 23);
insert into answer (text, question_id)
values ('Pro: Indexing improves overall system performance by reducing the need to scan the entire dataset, resulting in faster data retrieval and query execution. Con: Over-indexing can lead to unnecessary overhead and increased storage requirements, negatively impacting database performance.', 23);

insert into question (id, text, topic_id) 
values (24, 'When do we use btree index and hash index? When hash index can be better than btree?', 3);
insert into answer (text, question_id)
values ('We use a B-tree index when we need to efficiently support range queries and ordered traversal of the data, in other words when we utilize <, <=, =, >=, > - operators, or LIKE and similar keywords. B-trees are commonly used when we need to support range queries and ordered traversal of the data. B-trees are well-suited for scenarios where the data is frequently updated and the index needs to be dynamically balanced. On the other hand, we use a hash index when we have equality-based queries, when we compare using = sign, and do not require range queries or ordered traversal. Hash indexes are faster for exact match lookups, but they do not support range queries.', 24);
insert into answer (text, question_id)
values ('B-tree indexes are commonly used when we need to support range queries (<, <=, =, >=, > - operators) and ordered traversal of the data. They are suitable for scenarios where the data is frequently updated and the index needs to be dynamically balanced. In contrast, hash indexes are preferable when we have equality-based queries and do not require range queries or ordered traversal. Hash indexes are faster for exact match lookups, but they do not support range queries.', 24);
insert into answer (text, question_id)
values ('B-tree indexes are used when we need to efficiently support range queries and ordered traversal of the data. They are well-suited for scenarios where the data is frequently updated and the index needs to be dynamically balanced. On the other hand, hash indexes are more appropriate when we have equality-based queries and do not require range queries or ordered traversal. Hash indexes excel in providing fast exact match lookups, but they do not support range queries.', 24);
insert into answer (text, question_id)
values ('When it comes to choosing between a B-tree index and a hash index, we use a B-tree index when we need to efficiently support range queries and ordered traversal of the data. B-trees are suitable for scenarios where the data is frequently updated and the index needs to be dynamically balanced. Conversely, a hash index is preferred when we have equality-based queries and do not require range queries or ordered traversal. Hash indexes provide faster exact match lookups, but they lack support for range queries.', 24);
insert into answer (text, question_id)
values ('The decision to use a B-tree index or a hash index depends on the specific requirements of the application. A B-tree index is used when we need to efficiently support range queries and ordered traversal of the data. B-trees are well-suited for scenarios with frequent data updates and dynamic balancing needs. On the other hand, a hash index is chosen when we have equality-based queries and do not require range queries or ordered traversal. Hash indexes excel in providing fast exact match lookups, but they do not support range queries.', 24);

insert into question (id, text, topic_id) 
values (25, 'Tell me something about GIN index and how can we access it in PostgreSQL?', 3);
insert into answer (text, question_id)
values ('A GIN index, short for Generalized Inverted Index, is a type of index in PostgreSQL that is specifically designed to handle complex data types such as arrays and full-text search. It allows efficient access to these types of data by creating an inverted index that maps each unique value to the rows that contain it. To access a GIN index in PostgreSQL, you can simply create the index using the CREATE INDEX statement, specifying the GIN method. Once the index is created, you can use it in your queries by referencing the indexed column in the WHERE clause.', 25);
insert into answer (text, question_id)
values ('The GIN index in PostgreSQL is a specialized index structure called the Generalized Inverted Index. It is used to efficiently handle complex data types like arrays and full-text search. The GIN index creates an inverted index that maps each unique value to the rows that contain it, allowing for fast access to these types of data. To access a GIN index in PostgreSQL, you need to create the index using the CREATE INDEX statement with the GIN method specified. Once the index is created, you can utilize it in your queries by referencing the indexed column in the WHERE clause.', 25);
insert into answer (text, question_id)
values ('In PostgreSQL, the GIN index, or Generalized Inverted Index, is a type of index that is specifically designed for handling complex data types such as arrays and full-text search. It provides efficient access to these types of data by creating an inverted index that maps each unique value to the rows that contain it. To access a GIN index in PostgreSQL, you can create it using the CREATE INDEX statement with the GIN method specified. Once the index is created, you can use it in your queries by referencing the indexed column in the WHERE clause.', 25);
insert into answer (text, question_id)
values ('The GIN index in PostgreSQL is a specialized index structure known as the Generalized Inverted Index. It is used to efficiently handle complex data types like arrays and full-text search. By creating an inverted index that maps each unique value to the rows that contain it, the GIN index allows for fast access to these types of data. To access a GIN index in PostgreSQL, you can create it using the CREATE INDEX statement with the GIN method specified. Once the index is created, you can access it in your queries by referencing the indexed column in the WHERE clause.', 25);
insert into answer (text, question_id)
values ('In PostgreSQL, the GIN index, or Generalized Inverted Index, is a type of index that is specifically designed for handling complex data types such as arrays and full-text search. It enables efficient access to these types of data by creating an inverted index that maps each unique value to the rows that contain it. To access a GIN index in PostgreSQL, you can create it using the CREATE INDEX statement with the GIN method specified. Once the index is created, you can utilize it in your queries by referencing the indexed column in the WHERE clause.', 25);

insert into question (id, text, topic_id) 
values (26, 'For example we have a table of customers with id and name, and another table of sales with id, cutomer_id, and product_name. Which type of joins we should use to get ALL customers from customers table whether they bought a product or not?', 3);
insert into answer (text, question_id)
values ('To get all customers from the customers table, whether they bought a product or not, we can use RIGHT or LEFT JOIN, depending on the table position in the query. This will return all rows from the customers table and the matching rows from the sales table. If a customer did not make any purchases, the corresponding columns from the sales table will be filled with NULL values.', 26);
insert into answer (text, question_id)
values ('To get all customers from the customers table, whether they bought a product or not, we can use a LEFT JOIN. This join type will return all rows from the customers table and the matching rows from the sales table. If a customer did not make any purchases, the columns from the sales table will contain NULL values.', 26);
insert into answer (text, question_id)
values ('To get all customers from the customers table, whether they bought a product or not, we can use a RIGHT JOIN. This type of join will return all rows from the customers table and the matching rows from the sales table. If a customer did not make any purchases, the corresponding columns from the sales table will be filled with NULL values.', 26);
insert into answer (text, question_id)
values ('To get all customers from the customers table, whether they bought a product or not, we can use a LEFT JOIN. This join type will return all rows from the customers table and the matching rows from the sales table. If a customer did not make any purchases, the columns from the sales table will contain NULL values.', 26);
insert into answer (text, question_id)
values ('To get all customers from the customers table, whether they bought a product or not, we can use a LEFT JOIN. This type of join will return all rows from the customers table and the matching rows from the sales table. If a customer did not make any purchases, the corresponding columns from the sales table will be filled with NULL values.', 26);

insert into question (id, text, topic_id) 
values (27, 'What are the various types of relationships in Database? Define them.', 3);
insert into answer (text, question_id)
values ('The various types of relationships in a database are:
   - One-to-One: This relationship occurs when each record in one table is related to only one record in another table.
   - One-to-Many: In this relationship, a record in one table can be associated with multiple records in another table, but a record in the second table can only be associated with one record in the first table.
   - Many-to-One: This is the reverse of a one-to-many relationship, where multiple records in one table can be associated with a single record in another table.
   - Many-to-Many: In this type of relationship, multiple records in one table can be associated with multiple records in another table. To represent this relationship, a junction table is used.', 27); 
insert into answer (text, question_id)
values ('The different types of relationships in a database are as follows:
   - One-to-One: Each record in one table is associated with only one record in another table.
   - One-to-Many: A record in one table can have multiple related records in another table, but a record in the second table can only be associated with one record in the first table.
   - Many-to-One: Multiple records in one table can be associated with a single record in another table.
   - Many-to-Many: Multiple records in one table can be associated with multiple records in another table. This relationship is represented using a junction or join table.', 27);
insert into answer (text, question_id)
values ('There are several types of relationships in a database:
   - One-to-One: Each record in one table is linked to only one record in another table.
   - One-to-Many: A record in one table can be associated with multiple records in another table, but a record in the second table can only be associated with one record in the first table.
   - Many-to-One: Multiple records in one table can be associated with a single record in another table.
   - Many-to-Many: Multiple records in one table can be associated with multiple records in another table. This relationship is represented using a junction or associative table.', 27);
insert into answer (text, question_id)
values ('The various types of relationships in a database are:
   - One-to-One: Each record in one table is related to only one record in another table.
   - One-to-Many: A record in one table can have multiple related records in another table, but a record in the second table can only be associated with one record in the first table.
   - Many-to-One: Multiple records in one table can be associated with a single record in another table.
   - Many-to-Many: Multiple records in one table can be associated with multiple records in another table. This relationship is represented using a junction or join table.', 27);
insert into answer (text, question_id)
values ('Different types of relationships in a database include:
   - One-to-One: Each record in one table is linked to only one record in another table.
   - One-to-Many: A record in one table can have multiple related records in another table, but a record in the second table can only be associated with one record in the first table.
   - Many-to-One: Multiple records in one table can be associated with a single record in another table.
   - Many-to-Many: Multiple records in one table can be associated with multiple records in another table. This relationship is represented using a junction or associative table.', 27);

insert into question (id, text, topic_id) 
values (28, 'What are the different types of Normalization? Explain them.', 3);
insert into answer (text, question_id)
values ('The different types of normalization in a database are:
   - First Normal Form (1NF): This ensures that each column in a table contains only atomic values, meaning no repeating groups or arrays.
   - Second Normal Form (2NF): In addition to meeting 1NF requirements, this form eliminates partial dependencies by ensuring that each non-key column is fully dependent on the entire primary key.
   - Third Normal Form (3NF): Building upon 2NF, this form eliminates transitive dependencies by ensuring that each non-key column is dependent only on the primary key and not on other non-key columns.
   - Boyce-Codd Normal Form (BCNF): This form further refines 3NF by eliminating non-trivial dependencies, ensuring that every determinant is a candidate key.
   - Fourth Normal Form (4NF): This form addresses multi-valued dependencies by eliminating them through the use of separate tables for each set of related attributes.', 28);
insert into answer (text, question_id)
values ('The following are the several types of database normalization:
   - First Normal Form: This requires removing duplicate data and making sure that each column in a database only includes atomic values.
   - Second Normal Form: avoids partial dependencies by assuring that each non-key column is fully dependent on the entire primary key.
   - Third Normal Form: Based on the second Normal Form, this form avoids transitive dependencies by guaranteeing that each non-key column is only depended on the primary key and not on other non-key columns.
   - Boyce-Codde: This form builds on 3NF by removing non-trivial dependencies and assuring that every determinant is a candidate key.
   - 4NF: this form eliminates multi-valued dependencies by using separate tables for each set of connected attributes.', 28);
insert into answer (text, question_id)
values ('The different types of normalization in a database include:
   - First Normal Form (1NF): This involves eliminating duplicate data and ensuring that each column in a table contains only atomic values.
   - Second Normal Form (2NF): In addition to meeting 1NF requirements, this form eliminates partial dependencies by ensuring that each non-key column is fully dependent on the entire primary key.
   - Third Normal Form (3NF): Building upon 2NF, this form eliminates transitive dependencies by ensuring that each non-key column is dependent only on the primary key and not on other non-key columns.
   - Boyce-Codd Normal Form (BCNF): This form refines 3NF by eliminating non-trivial dependencies, ensuring that every determinant is a candidate key.
   - Fourth Normal Form (4NF): This form addresses multi-valued dependencies by eliminating them through the use of separate tables for each set of related attributes.', 28);
insert into answer (text, question_id)
values ('The various kinds of normalization in a database include:
   - First Normal Form: This entails removing redundant information and making sure that each column in a database only includes atomic values.
   - Second Normal Form: This form ensures that each non-key column is entirely reliant on the entire main key, in addition to satisfying the constraints of 1NF.
   Building on the Second Normal Form, the Third Normal Form ensures that each non-key column is dependent only on the primary key and not on any other non-key columns, hence eliminating transitive dependencies.
   - Boyce-Code normal form: By removing non-trivial dependencies, this form improves 3NF by ensuring that every determinant is a candidate key.
   - Fourth normal form: By using separate tables for each set of linked attributes, this normal form eliminates multi-valued dependencies.', 28);
insert into answer (text, question_id)
values ('The different types of normalization in a database include:
   - First Normal Form (1NF): This involves eliminating duplicate data and ensuring that each column in a table contains only atomic values.
   - Second Normal Form (2NF): In addition to meeting 1NF requirements, this form eliminates partial dependencies by ensuring that each non-key column is fully dependent on the entire primary key.
   - Third Normal Form (3NF): Building upon 2NF, this form eliminates transitive dependencies by ensuring that each non-key column is dependent only on the primary key and not on other non-key columns.
   - Boyce-Codd Normal Form (BCNF): This form refines 3NF by eliminating non-trivial dependencies, ensuring that every determinant is a candidate key.
   - Fourth Normal Form (4NF): This form addresses multi-valued dependencies by eliminating them through the use of separate tables for each set of related attributes.', 28);

insert into question (id, text, topic_id) 
values (29, 'What is ACID?', 3);
insert into answer (text, question_id)
values ('ACID stands for Atomicity, Consistency, Isolation, and Durability. It is a set of properties that ensure reliability and integrity in database transactions. Atomicity ensures that a transaction is treated as a single, indivisible unit of work. Consistency ensures that a transaction brings the database from one valid state to another. Isolation ensures that concurrent transactions do not interfere with each other. Durability ensures that once a transaction is committed, its changes are permanent and will survive any subsequent failures.', 29);
insert into answer (text, question_id)
values ('ACID refers to the properties of a reliable database transaction. Atomicity ensures that a transaction is treated as a single, indivisible unit of work, either fully completed or fully rolled back. Consistency ensures that a transaction brings the database from one valid state to another, preserving data integrity and enforcing constraints. Isolation ensures that concurrent transactions do not interfere with each other, providing concurrency control. Durability ensures that once a transaction is committed, its changes are permanently stored and will survive any subsequent failures.', 29);
insert into answer (text, question_id)
values ('ACID is an acronym for Atomicity, Consistency, Isolation, and Durability. These properties define the reliability and integrity of a database transaction. Atomicity guarantees that a transaction is treated as a single, indivisible unit of work, ensuring that all changes are either fully committed or fully rolled back. Consistency ensures that a transaction brings the database from one valid state to another, maintaining data integrity and enforcing constraints. Isolation provides concurrency control by ensuring that concurrent transactions do not interfere with each other. Durability guarantees that once a transaction is committed, its changes are permanently stored and will survive any subsequent failures.', 29);
insert into answer (text, question_id)
values ('ACID stands for Atomicity, Consistency, Isolation, and Durability. These properties are essential for ensuring the reliability and integrity of database transactions. Atomicity ensures that a transaction is treated as a single, indivisible unit of work, meaning that all changes within the transaction are either fully committed or fully rolled back. Consistency guarantees that a transaction brings the database from one valid state to another, preserving data integrity and enforcing constraints. Isolation provides concurrency control by ensuring that concurrent transactions do not interfere with each other. Durability ensures that once a transaction is committed, its changes are permanently stored and will survive any subsequent failures.', 29);
insert into answer (text, question_id)
values ('ACID refers to the properties that define the reliability and integrity of a database transaction. Atomicity ensures that a transaction is treated as a single, indivisible unit of work, guaranteeing that all changes are either fully committed or fully rolled back. Consistency ensures that a transaction brings the database from one valid state to another, maintaining data integrity and enforcing constraints. Isolation provides concurrency control by ensuring that concurrent transactions do not interfere with each other. Durability guarantees that once a transaction is committed, its changes are permanently stored and will survive any subsequent failures.', 29);

insert into question (id, text, topic_id) 
values (30, 'What is the difference between RDMS and Nonrelational DBMS?', 3);
insert into answer (text, question_id)
values ('The main difference between a Relational Database Management System (RDBMS) and a Nonrelational Database Management System (Nonrelational DBMS) is in their data storage and organization. RDBMS stores data in tables with predefined schemas, while Nonrelational DBMS stores data in various formats such as key-value pairs, documents, graphs, or wide-column stores, without requiring a predefined schema.', 30);
insert into answer (text, question_id)
values ('RDMS and Nonrelational DBMS differ in their data modeling approach. RDBMS follows a structured and rigid data model based on tables, rows, and columns, while Nonrelational DBMS allows for flexible and dynamic data models that can adapt to changing requirements.', 30);
insert into answer (text, question_id)
values ('Another distinction between RDBMS and Nonrelational DBMS is their scalability. RDBMS typically scales vertically, meaning that it requires more powerful hardware to handle increased workloads. Nonrelational DBMS, on the other hand, is designed for horizontal scalability, allowing for distributed storage and processing across multiple servers.', 30);
insert into answer (text, question_id)
values ('RDBMS emphasizes data consistency and integrity through the use of ACID (Atomicity, Consistency, Isolation, Durability) properties, ensuring reliable and accurate data transactions. Nonrelational DBMS often sacrifices some of these ACID properties in favor of high scalability and performance, prioritizing availability and partition tolerance (CAP theorem).', 30);
insert into answer (text, question_id)
values ('RDBMS is generally better suited for structured data with complex relationships and strict consistency requirements, making it ideal for applications that rely heavily on transactions and complex queries. Nonrelational DBMS excels in handling unstructured or semi-structured data, providing high scalability and flexibility for applications that require rapid data ingestion and retrieval.', 30);

-- COMPUTER NETWORKS

insert into question (id, text, topic_id) 
values (31, 'What are the different layers of the OSI Model?', 4);
insert into answer (text, question_id)
values ('The different layers of the OSI Model are: 
   - Physical layer: Deals with the physical transmission of data over the network, including cables, connectors, and signaling.
   - Data link layer: Responsible for error-free transmission of data frames between two directly connected nodes, ensuring reliable communication.
   - Network layer: Handles logical addressing and routing of data packets across multiple networks, enabling end-to-end connectivity.
   - Transport layer: Ensures reliable delivery of data by providing error detection, flow control, and segmentation of data into manageable chunks.
   - Session layer: Establishes, maintains, and terminates communication sessions between applications on different devices.
   - Presentation layer: Handles data formatting, encryption, compression, and protocol conversion to ensure compatibility between different systems.
   - Application layer: Provides network services directly to user applications, such as email, web browsing, file transfer, etc.', 31);
insert into answer (text, question_id)
values ('The OSI Model is made up of the following layers:
   - Physical layer: Handles data transfer via physical means such as cables, connectors, and network interfaces.
   - Data link layer: In charge of detecting and correcting errors in sent data as well as managing access to shared media.
   - Network layer: Manages logical addressing and routing of data packets over several networks, allowing end-to-end communication.
   - Transport layer: Provides data segmentation, flow management, and error recovery methods to ensure reliable data transmission.
   - Session layer: Creates, manages, and ends sessions between programs operating on various devices.
   - Presentation layer: Handles data formatting, encryption, and compression to ensure system compatibility.
   - Application layer: Offers network services to user applications, allowing them to access network resources and communicate with one another.', 31);
insert into answer (text, question_id)
values ('The OSI Model is composed of the following layers:
   - Physical layer: Deals with the physical transmission of data over the network, including electrical signals, cables, and network interfaces.
   - Data link layer: Manages the reliable transmission of data frames between directly connected nodes, handling error detection and correction.
   - Network layer: Handles logical addressing and routing of data packets across multiple networks, enabling end-to-end communication.
   - Transport layer: Ensures reliable delivery of data by providing flow control, error recovery, and segmentation of data.
   - Session layer: Establishes, maintains, and terminates communication sessions between applications on different devices.
   - Presentation layer: Handles data representation, encryption, and compression to ensure compatibility between different systems.
   - Application layer: Provides network services directly to user applications, allowing them to access network resources and communicate with other applications.', 31);
insert into answer (text, question_id)
values ('The following layers make up the OSI Model:
   - Physical layer: This layer controls how data is physically sent across conduits like cables, connectors, and network interfaces.
   - Data link layer: In charge of controlling access to shared media as well as error detection and correction in the transmitted data.
   In order to enable end-to-end communication, the network layer manages the logical addressing and routing of data packets across various networks.
   - Transport layer: By supplying segmentation, flow management, and error recovery techniques, this layer makes sure that data is delivered reliably.
   - Session layer: Creates, maintains, and ends sessions between programs that are operating on various devices.
   - Presentation layer: Takes care of data formatting, encryption, and compression to ensure system compatibility.
   - Application layer: Offers network services to user apps so they can interact with one another and access network resources.', 31);
insert into answer (text, question_id)
values ('The OSI Model is structured into the following layers:
   - Physical layer: Deals with the physical transmission of data over the network, including the electrical, mechanical, and procedural aspects.
   - Data link layer: Manages the reliable transmission of data frames between directly connected nodes, ensuring error-free communication.
   - Network layer: Handles logical addressing and routing of data packets across different networks, enabling interconnectivity.
   - Transport layer: Ensures the reliable delivery of data by providing end-to-end error detection, flow control, and segmentation.
   - Session layer: Establishes, manages, and terminates sessions between applications on different devices, allowing for synchronized communication.
   - Presentation layer: Handles data representation, encryption, and compression to ensure compatibility between different systems.
   - Application layer: Provides network services directly to user applications, allowing them to access network resources and communicate with other applications.', 31);

insert into question (id, text, topic_id) 
values (32, 'What is the difference between IPv4 and IPv6?', 4);
insert into answer (text, question_id)
values ('IPv4 and IPv6 are both protocols used for internet communication, but they differ in terms of address length. IPv4 addresses are 32 bits long, allowing for approximately 4.3 billion unique addresses, while IPv6 addresses are 128 bits long, providing for a virtually unlimited number of unique addresses.', 32);
insert into answer (text, question_id)
values ('Another difference between IPv4 and IPv6 is the address format. IPv4 addresses are written in decimal format with four sets of numbers separated by periods (e.g., 192.168.0.1), while IPv6 addresses are written in hexadecimal format with eight sets of numbers separated by colons (e.g., 2001:0db8:85a3:0000:0000:8a2e:0370:7334).', 32);
insert into answer (text, question_id)
values ('IPv4 uses network address translation (NAT) to allow multiple devices to share a single public IP address, while IPv6 eliminates the need for NAT by providing a large enough address space to assign unique addresses to every device connected to the internet.', 32);
insert into answer (text, question_id)
values ('IPv4 has a limited number of available addresses, which has led to the depletion of available IPv4 addresses. In contrast, IPv6 was designed to address this issue and provide a solution for the future growth of the internet by offering a significantly larger pool of available addresses.', 32);
insert into answer (text, question_id)
values ('IPv6 includes built-in support for features such as auto-configuration, mobility, and better security compared to IPv4. These features enhance the efficiency and security of network communication in IPv6-enabled networks.', 32);

insert into question (id, text, topic_id) 
values (33, 'Explain in detail 3 way Handshaking.', 4);
insert into answer (text, question_id)
values ('Three-way handshake, also known as TCP handshake, is a method used by TCP (Transmission Control Protocol) to establish a reliable and secure connection between two devices over an IP network. It involves three steps: SYN(synchronize), SYN-ACK, and ACK(acknowledgement).', 33);
insert into answer (text, question_id)
values ('The first step of the three-way handshake is the SYN (synchronize) packet. The initiating device, often referred to as the client, sends a SYN packet to the receiving device, known as the server. This packet contains a sequence number that the client chooses randomly to start the connection.', 33);
insert into answer (text, question_id)
values ('Upon receiving the SYN packet, the server responds with a SYN-ACK (synchronize-acknowledge) packet. This packet acknowledges the client SYN packet by incrementing the client sequence number by one and choosing its own random sequence number. The SYN-ACK packet also serves as a synchronization request from the server to the client.', 33);
insert into answer (text, question_id)
values ('After receiving the SYN-ACK packet, the client sends an ACK (acknowledge) packet back to the server. This packet acknowledges the server SYN-ACK packet by incrementing the server sequence number by one. At this point, both devices have exchanged their initial sequence numbers and have agreed on the initial sequence number for the data transfer.', 33);
insert into answer (text, question_id)
values ('The three-way handshake is crucial for establishing a reliable connection because it ensures that both devices are ready and capable of receiving and transmitting data. It also helps in establishing initial parameters for reliable data transmission, such as window size and other congestion control mechanisms. By completing this handshake, both devices can start exchanging data packets with confidence in the reliability and integrity of the connection.', 33);

insert into question (id, text, topic_id) 
values (34, 'Tell me about DNS protocol.', 4);
insert into answer (text, question_id)
values ('DNS, or Domain Name System, is a decentralized system that translates domain names into IP addresses. It acts as a phonebook for the internet, allowing users to access websites and other online resources by typing in easy-to-remember domain names instead of complicated numerical IP addresses.', 34);
insert into answer (text, question_id)
values ('DNS is a hierarchical system that consists of multiple servers, known as DNS servers or name servers. These servers store and distribute information about domain names and their corresponding IP addresses. When a user enters a domain name in their web browser, the DNS system is queried to find the IP address associated with that domain.', 34);
insert into answer (text, question_id)
values ('DNS works by using a process called DNS resolution. When a user requests a website, their device first checks its local cache to see if it has previously resolved the domain name. If not, it sends a query to the nearest DNS server, which then recursively searches for the IP address by consulting other DNS servers until it finds the correct one.', 34);
insert into answer (text, question_id)
values ('DNS plays a crucial role in the functioning of the internet as it allows users to easily navigate and access websites without having to remember complex IP addresses. It also enables services like email delivery, where the DNS system is used to locate the mail server associated with a specific domain.', 34);
insert into answer (text, question_id)
values ('In addition to translating domain names into IP addresses, DNS also supports other types of records, such as MX records for email routing, TXT records for various purposes like domain verification, and SRV records for specifying specific services associated with a domain. This flexibility makes DNS a versatile system that can handle various needs and requirements of modern internet communication.', 34);

insert into question (id, text, topic_id) 
values (35, 'What is the primary purpose of a network router? List its key hardware components and their purposes.', 4);
insert into answer (text, question_id)
values ('The primary purpose of a network router is to forward data packets between different networks, ensuring efficient and reliable communication. Its key hardware components include:
   - Central Processing Unit (CPU): Responsible for executing routing protocols, managing network traffic, and making forwarding decisions.
   - Memory: Stores routing tables, packet buffers, and other essential data for efficient routing operations.
   - Network Interfaces: Connect the router to different networks and enable the transmission and reception of data packets.
   - Routing Table: Contains information about network addresses and the best paths to reach them, allowing the router to make intelligent forwarding decisions.
   - Switching Fabric: Facilitates the movement of data packets between input and output interfaces, ensuring proper routing and forwarding.', 35);
insert into answer (text, question_id)
values ('The primary purpose of a network router is to provide connectivity between different networks, allowing for the exchange of data. Its key hardware components include:
   - Input/Output Ports: Connect the router to various devices and networks, enabling the transfer of data packets.
   - Routing Processor: Executes routing protocols, calculates optimal paths for data transmission, and manages network traffic.
   - Forwarding Engine: Handles the actual forwarding of data packets based on the routing decisions made by the routing processor.
   - Memory: Stores routing tables, packet buffers, and other crucial information necessary for efficient routing operations.
   - Power Supply Unit (PSU): Provides electrical power to all the router components, ensuring uninterrupted operation.', 35);
insert into answer (text, question_id)
values ('The primary purpose of a network router is to direct data traffic between different networks, ensuring efficient and secure communication. Its key hardware components include:
   - Network Interfaces: Connect the router to various networks and devices, allowing for the transmission and reception of data packets.
   - Routing Processor: Executes routing protocols, determines optimal paths for data transmission, and manages network traffic.
   - Memory: Stores routing tables, packet buffers, and other vital information required for effective routing operations.
   - Forwarding Engine: Responsible for forwarding data packets based on the routing decisions made by the routing processor.
   - Firewall: Provides network security by filtering and inspecting incoming and outgoing data packets, protecting against unauthorized access.', 35);
insert into answer (text, question_id)
values ('The primary purpose of a network router is to facilitate the transfer of data between different networks, ensuring efficient and reliable communication. Its key hardware components include:
   - Network Interfaces: Connect the router to various networks and devices, enabling the transmission and reception of data packets.
   - Routing Processor: Executes routing protocols, determines the best paths for data transmission, and manages network traffic.
   - Memory: Stores routing tables, packet buffers, and other essential data for efficient routing operations.
   - Switching Fabric: Facilitates the movement of data packets between input and output interfaces, ensuring proper routing and forwarding.
   - Power Supply Unit (PSU): Provides electrical power to all the router components, ensuring uninterrupted operation.', 35);
insert into answer (text, question_id)
values ('The primary purpose of a network router is to route data packets between different networks, enabling effective communication. Its key hardware components include:
   - Network Interfaces: Connect the router to various networks and devices, allowing for the transmission and reception of data packets.
   - Routing Processor: Executes routing protocols, calculates optimal paths for data transmission, and manages network traffic.
   - Memory: Stores routing tables, packet buffers, and other crucial information necessary for efficient routing operations.
   - Forwarding Engine: Responsible for forwarding data packets based on the routing decisions made by the routing processor.
   - Power Supply Unit (PSU): Supplies electrical power to all the router components, ensuring continuous operation.', 35);

insert into question (id, text, topic_id) 
values (36, 'For each layer of the TCP/IP protocol stack, provide two protocol examples', 4);
insert into answer (text, question_id)
values ('For the Application Layer:
1. HTTP (Hypertext Transfer Protocol): Used for transmitting web pages and other resources over the internet.
2. SMTP (Simple Mail Transfer Protocol): Responsible for sending and receiving email messages.
For the Transport Layer:
1. TCP (Transmission Control Protocol): Provides reliable, connection-oriented communication between devices.
2. UDP (User Datagram Protocol): Offers connectionless, unreliable communication, ideal for real-time applications like video streaming or online gaming.', 36);
insert into answer (text, question_id)
values ('For the Network Layer:
1. IP (Internet Protocol): Handles the addressing and routing of data packets across networks.
2. ICMP (Internet Control Message Protocol): Used for error reporting and diagnostic functions, such as ping or traceroute.
For the Data Link Layer:
1. Ethernet: Commonly used for local area networks (LANs), it provides a physical and logical connection between devices.
2. Wi-Fi (802.11): Enables wireless communication between devices within a network.', 36);
insert into answer (text, question_id)
values ('For the Physical Layer:
1. Ethernet cables: Used for wired connections, providing the physical medium for data transmission.
2. Fiber optic cables: Transmit data using light signals, offering high-speed and long-distance communication.
For the Application Layer:
1. FTP (File Transfer Protocol): Used for transferring files between a client and a server.
2. DNS (Domain Name System): Converts domain names into IP addresses, allowing users to access websites.', 36);
insert into answer (text, question_id)
values ('For the Transport Layer:
1. SCTP (Stream Control Transmission Protocol): Provides reliable, message-oriented communication between devices.
2. DCCP (Datagram Congestion Control Protocol): Offers connectionless, congestion-controlled communication for multimedia streaming.
For the Network Layer:
1. IPv6 (Internet Protocol version 6): The latest version of IP, designed to address the limitations of IPv4 and accommodate the growing number of devices connected to the internet.
2. OSPF (Open Shortest Path First): A routing protocol that determines the best path for data packets to travel through a network.', 36);
insert into answer (text, question_id)
values ('For the Data Link Layer:
1. PPP (Point-to-Point Protocol): Enables the transmission of data packets over serial connections, such as dial-up or DSL.
2. ATM (Asynchronous Transfer Mode): A cell-based switching technology used for high-speed data transfer in both LAN and WAN environments.
For the Physical Layer:
1. RS-232 (Recommended Standard 232): A standard for serial communication between devices, commonly used for connecting computers to modems or printers.
2. Bluetooth: Enables wireless communication between devices within short distances, commonly used for connecting peripherals like keyboards or headphones to a computer or smartphone.', 36);

insert into question (id, text, topic_id) 
values (37, 'Which error detection techniques do you know? Briefly explain procedure of one of them.', 4);
insert into answer (text, question_id)
values ('Cyclic Redundancy Check (CRC): This technique involves the use of a polynomial division algorithm to generate a checksum for a data packet. The sender performs the division and appends the resulting checksum to the data packet. The receiver then performs the same division and compares the calculated checksum with the received checksum. If they match, no errors are detected. Otherwise, errors are present.', 37);
insert into answer (text, question_id)
values ('Checksum: In this technique, a simple mathematical sum or exclusive OR (XOR) operation is performed on all the bits in a data packet. The result is then appended to the packet as a checksum. The receiver performs the same operation on the received packet and compares the calculated checksum with the received checksum. If they match, no errors are detected. Otherwise, errors are present.', 37);
insert into answer (text, question_id)
values ('Parity Check: This technique involves adding an extra bit, called a parity bit, to each byte or group of bytes in a data packet. The parity bit is set to 0 or 1 in such a way that the total number of 1s in each byte (including the parity bit) is always even (even parity) or odd (odd parity). The receiver counts the number of 1s in each byte (including the received parity bit) and checks if it matches the expected parity. If not, errors are detected.', 37);
insert into answer (text, question_id)
values ('Hamming Code: Hamming code is an error detection and correction technique that adds redundant bits to a data packet based on a specific algorithm. These redundant bits allow the receiver to detect and correct single-bit errors in the received packet. The algorithm ensures that the redundant bits are placed at specific positions in the packet to maximize error detection and correction capabilities.', 37);
insert into answer (text, question_id)
values ('Forward Error Correction (FEC): FEC is an error detection and correction technique that adds extra redundant bits to a data packet in such a way that the receiver can reconstruct the original data even if some bits are corrupted during transmission. This technique is commonly used in situations where retransmission of lost or corrupted packets is not feasible, such as in real-time multimedia streaming.', 37);

-- OPERATING SYSTEMS

insert into question (id, text, topic_id) 
values (38, 'What is multithreading in Operating System?', 5);
insert into answer (text, question_id)
values ('Multithreading in an operating system refers to the ability of the system to execute multiple threads concurrently. It allows for parallel execution of tasks, where each thread can perform a separate set of instructions independently.', 38);
insert into answer (text, question_id)
values ('In an operating system, multithreading refers to the concept of executing multiple threads within a single process. It enables concurrent execution of tasks, where each thread can perform its own set of instructions simultaneously, leading to improved efficiency and responsiveness.', 38);
insert into answer (text, question_id)
values ('Multithreading is when system can manage many threads of execution within a single process. It enables concurrent task execution where each thread may independently carry out its own set of instructions, improving performance and resource utilization.', 38);
insert into answer (text, question_id)
values ('In the context of an operating system, multithreading refers to the capability of executing multiple threads within a single process. It enables concurrent execution of tasks, where each thread can run independently and share the resources of the process, leading to enhanced responsiveness and better utilization of system resources.', 38);
insert into answer (text, question_id)
values ('This ability to operate many threads concurrently within a single process is known as multithreading in operating systems. It allows to execute activities concurrently in which each thread may independently carry out its own set of instructions improves performance, responsiveness, and resource management.', 38);

insert into question (id, text, topic_id) 
values (39, 'What is a deadlock, and what are the four necessary conditions for it to occur?', 5);
insert into answer (text, question_id)
values ('A deadlock is a situation in computer science where two or more processes are unable to proceed because each is waiting for the other to finish. The four necessary conditions for a deadlock are mutual exclusion, hold and wait, no preemption, and circular wait.', 39);
insert into answer (text, question_id)
values ('A deadlock occurs when two or more processes are blocked and unable to proceed because they are waiting for each other to release resources. The four necessary conditions for a deadlock are resource allocation, hold and wait, no preemption, and circular wait.', 39);
insert into answer (text, question_id)
values ('A deadlock is a state in which two or more processes are unable to continue executing because each is waiting for a resource held by the other. The four necessary conditions for a deadlock are mutual exclusion, hold and wait, no preemption, and circular wait.', 39);
insert into answer (text, question_id)
values ('Deadlock is a situation that occurs in computer systems when two or more processes are blocked and unable to proceed because they are waiting for resources held by each other. The four necessary conditions for a deadlock are mutual exclusion, hold and wait, no preemption, and circular wait.', 39);
insert into answer (text, question_id)
values ('A deadlock is a state in which two or more processes are waiting for each other to release resources, resulting in a standstill. The four necessary conditions for a deadlock are mutual exclusion, hold and wait, no preemption, and circular wait.', 39);

insert into question (id, text, topic_id) 
values (40, 'What is the difference between Shortest Job First and Round Robin scheduling? Explain how these algorithms work.', 5);
insert into answer (text, question_id)
values ('The main difference between Shortest Job First and Round Robin scheduling is that SJF prioritizes processes with the shortest burst time, while RR assigns a fixed time slice to each process before moving on to the next one. SJF works by sorting the processes based on their burst time and executing them in that order, while RR works by repeatedly executing each process for a fixed time quantum before switching to the next process in the queue.', 40);
insert into answer (text, question_id)
values ('Shortest Job First and Round Robin scheduling differ in how they prioritize and allocate CPU time to processes. SJF prioritizes processes with the shortest burst time, while RR allocates a fixed time slice to each process before moving on to the next one. SJF works by selecting the process with the shortest remaining burst time and executing it first, while RR works by executing each process for a fixed time quantum and then switching to the next process in the queue.', 40);
insert into answer (text, question_id)
values ('Shortest Job First and Round Robin scheduling are two different algorithms used for CPU scheduling in operating systems. SJF prioritizes processes with the shortest burst time, while RR allocates a fixed time slice to each process before moving on to the next one. SJF works by selecting the process with the shortest remaining burst time and executing it first, while RR works by executing each process for a fixed time quantum and then switching to the next process in the queue.', 40);
insert into answer (text, question_id)
values ('The main difference between Shortest Job First and Round Robin scheduling is how they prioritize and allocate CPU time to processes. SJF prioritizes processes with the shortest burst time, while RR allocates a fixed time slice to each process before moving on to the next one. SJF works by selecting the process with the shortest remaining burst time and executing it first, while RR works by executing each process for a fixed time quantum and then switching to the next process in the queue.', 40);
insert into answer (text, question_id)
values ('Shortest Job First and Round Robin scheduling are two different algorithms used for CPU scheduling in operating systems. SJF prioritizes processes with the shortest burst time, while RR allocates a fixed time slice to each process before moving on to the next one. SJF works by selecting the process with the shortest remaining burst time and executing it first, while RR works by executing each process for a fixed time quantum and then switching to the next process in the queue.', 40);

insert into question (id, text, topic_id) 
values (41, 'What is MMU and how it utilizes TLB?', 5);
insert into answer (text, question_id)
values ('MMU stands for Memory Management Unit, which is a hardware component in a computer that manages memory access for processes. It utilizes TLB (Translation Lookaside Buffer) to store recently used memory mappings, allowing for faster access to memory.', 41);
insert into answer (text, question_id)
values ('The MMU is responsible for mapping virtual addresses used by processes to physical addresses in memory. It utilizes TLB to cache frequently accessed mappings, reducing the time needed to access memory.', 41);
insert into answer (text, question_id)
values ('The MMU is a hardware component that manages the translation of virtual memory addresses to physical memory addresses. It utilizes TLB to store recently used translations, improving the performance of memory access.', 41);
insert into answer (text, question_id)
values ('The MMU is a component in a computer that manages the translation of virtual memory addresses to physical memory addresses. It uses TLB to cache frequently accessed translations, speeding up the process of accessing memory.', 41);
insert into answer (text, question_id)
values ('MMU is a hardware component that manages the mapping of virtual addresses used by processes to physical addresses in memory. It utilizes TLB to store recently accessed mappings, improving the efficiency of memory access.', 41);

insert into question (id, text, topic_id) 
values (42, 'How one process can pass info to another?', 5);
insert into answer (text, question_id)
values ('One process can pass information to another by using interprocess communication mechanisms such as pipes, sockets, and message queues.', 42);
insert into answer (text, question_id)
values ('Information can be passed between processes through shared memory, where both processes have access to the same memory space.', 42);
insert into answer (text, question_id)
values ('Processes can communicate with each other through signals, where one process sends a signal to another process to indicate a certain event or action.', 42);
insert into answer (text, question_id)
values ('RPC (Remote Procedure Call) can be used to pass information between processes running on different machines over a network.', 42);
insert into answer (text, question_id)
values ('One process can pass information to another by writing to a file or database that both processes have access to.', 42);

insert into question (id, text, topic_id) 
values (43, 'What is an Inode and how is it structured?', 5);
insert into answer (text, question_id)
values ('An Inode is a data structure used by the file system to store information about a file, including its ownership, permissions, timestamps, and location on disk.', 43);
insert into answer (text, question_id)
values ('The Inode is structured as a fixed-size block of data that contains fields for storing the file metadata, such as its size, type, and location.', 43);
insert into answer (text, question_id)
values ('Each Inode is uniquely identified by a number, which is used by the file system to locate and access the file on disk.', 43);
insert into answer (text, question_id)
values ('Inodes are organized into a table or array within the file system, which allows the operating system to quickly locate and access files.', 43);
insert into answer (text, question_id)
values ('The Inode also contains pointers to the blocks of data that make up the file, allowing the operating system to read and write data to and from the file.', 43);

insert into question (id, text, topic_id) 
values (44, 'How can one recover from the deadlock?', 5);
insert into answer (text, question_id)
values ('One way to recover from a deadlock is to terminate one or more processes involved in the deadlock, freeing up the resources they were holding and allowing the remaining processes to continue.', 44);
insert into answer (text, question_id)
values ('Another approach is to use a timeout mechanism, where processes that are waiting for resources will give up after a certain amount of time and release any resources they were holding, allowing other processes to access them.', 44);
insert into answer (text, question_id)
values ('A third option is to use a preemption mechanism, where the operating system forcibly takes resources away from one process and gives them to another, breaking the deadlock and allowing the system to continue running.', 44);
insert into answer (text, question_id)
values ('A fourth approach is to use a resource allocation algorithm that can detect and prevent deadlocks from occurring in the first place, by carefully managing resource requests and releases.', 44);
insert into answer (text, question_id)
values ('Finally, it may be possible to resolve a deadlock by manually releasing resources or changing the order in which processes request resources, although this can be difficult and time-consuming.', 44);

-- DATA STRUCTURES

insert into question (id, text, topic_id) 
values (45, 'What is the difference between a stack and a queue?', 6);
insert into answer (text, question_id)
values ('A stack is a data structure that follows the Last-In-First-Out (LIFO) principle, while a queue follows the First-In-First-Out (FIFO) principle.', 45);
insert into answer (text, question_id)
values ('In a stack, elements are added and removed from the top, while in a queue, elements are added at the back and removed from the front.', 45);
insert into answer (text, question_id)
values ('Stacks are typically used for recursive function calls, while queues are often used in breadth-first search algorithms.', 45);
insert into answer (text, question_id)
values ('Stacks have a push() and pop() operation, while queues have an enqueue() and dequeue() operation.', 45);
insert into answer (text, question_id)
values ('A stack can be visualized as a vertical structure, while a queue can be visualized as a horizontal structure.', 45);

insert into question (id, text, topic_id) 
values (46, 'What is the difference between an array and a linked list?', 6);
insert into answer (text, question_id)
values ('An array is a data structure that stores elements in contiguous memory locations, while a linked list stores elements in non-contiguous memory locations connected by pointers.', 46);
insert into answer (text, question_id)
values ('An array is a contiguous block of memory used to store elements, while a linked list is a collection of nodes that are connected via pointers.', 46);
insert into answer (text, question_id)
values ('Arrays have a fixed size, while linked lists can dynamically grow or shrink in size.', 46);
insert into answer (text, question_id)
values ('Accessing an element in an array takes constant time, while accessing an element in a linked list takes linear time.', 46);
insert into answer (text, question_id)
values ('Inserting or deleting an element in an array requires shifting all subsequent elements, while in a linked list it only requires updating pointers.', 46);

insert into question (id, text, topic_id) 
values (47, 'What do you understand by sorting? Name some popular sorting techniques and in what scenario do they work best?', 6);
insert into answer (text, question_id)
values ('Sorting is the process of arranging data in a specific order, usually ascending or descending. It is an essential operation in computer science and is used in a wide range of applications, including database management, search algorithms, and data analysis.
Some popular sorting techniques are:
1. Bubble sort: This is a simple sorting algorithm that compares adjacent elements and swaps them if they are in the wrong order. It works best for small datasets but can be slow for large datasets.
2. Selection sort: This algorithm selects the smallest element in the dataset and swaps it with the first element. It then selects the second smallest element and swaps it with the second element, and so on. It is efficient for small datasets but can be slow for large datasets.
3. Insertion sort: This algorithm inserts each element of the dataset into its correct position in a sorted list. It is efficient for small datasets and can be faster than other algorithms for partially sorted datasets.
4. Merge sort: This algorithm divides the dataset into smaller sub-arrays, sorts them recursively, and then merges them back together. It is efficient for large datasets and has a time complexity of O(n log n).
5. Quick sort: This algorithm selects a pivot element, partitions the dataset around the pivot, and recursively sorts the sub-arrays. It is efficient for large datasets and has a time complexity of O(n log n) on average.
The choice of sorting technique depends on the size of the dataset, the distribution of data, and the available memory. For small datasets, simple algorithms like bubble sort and insertion sort may be sufficient, while for large datasets, more efficient algorithms like merge sort and quick sort are preferred. Additionally, if the data is already partially sorted, insertion sort may be faster than other algorithms.', 47);
insert into answer (text, question_id)
values ('Data must be arranged in a specified order, which is the process of sorting. It is a fundamental process in computer science and has a variety of uses, such as data analysis, search algorithms, and database management.
Some popular methods of sorting:
1. Bubble sort: This straightforward sorting method swaps neighboring elements if they are in the wrong order after comparing nearby elements. Although it can be slow for large datasets, it works best for small datasets.
2. Selection sort: algorithm chooses the tiniest element in the dataset and swaps it with element one. The next step is to choose the next-smallest element and swap it with the first element, and so forth. Although it can be slow for huge datasets, it is effective for small datasets.
3. Insertion sort: This algorithm places each dataset member in the appropriate place in a sorted list. It is effective for small datasets and can be quicker for partially sorted datasets than other algorithms.
4. Merge sort: This algorithm separates the dataset into smaller sub-arrays, recursively sorts them, and then combines them. It is effective for handling massive datasets and has an O(n log n) time complexity.
5. Quick sort algorithm divides the dataset around a pivot element it has chosen, then recursively sorts the sub-arrays. It is effective at handling huge datasets and, on average, has a time complexity of O(n log n).
The size of the dataset, how the data is distributed, and the amount of memory that is available all influence the sorting method that is chosen. Simple methods like bubble sort and insertion sort may be adequate for small datasets, however merge sort and rapid sort are preferable for large datasets due to their increased efficiency. Additionally, insertion sort could be quicker than other methods if the data has already been partially sorted.', 47);
insert into answer (text, question_id)
values ('Radix Sort is a non-comparative sorting algorithm that operates by distributing elements into buckets based on their radix. The radix refers to the base of the number system used to represent numbers. The elements are sorted by processing them digit by digit, from the least significant digit (LSD) to the most significant digit (MSD), or vice versa.
The algorithm begins by sorting elements based on the value of the unit place (LSD). Then, it sorts elements based on the value of the tenth place, and this process continues until the most significant place.
Radix sort is especially effective when sorting data that can be sorted lexicographically, such as integers and words.', 47);
insert into answer (text, question_id)
values ('Heap sort is a comparison-based sorting technique that uses a Binary Heap data structure.
Heap sort works by first creating a max heap from the input data. The max heap is a specialized tree-based data structure that satisfies the heap property. In a max heap, for any given node I, the value of I is greater than or equal to the values of its children. The largest element is the root of the heap. This element is placed at the end of the array, and the process is repeated for the remaining elements until the entire array is sorted
Heap sort is also an in-place algorithm, meaning it does not require extra memory space to sort.
In conclusion, heap sort is beneficial when you need to sort large datasets efficiently and quickly, especially when extra memory is not available or expensive. It is also useful when you need to find the largest or smallest element in a list quickly.', 47);
insert into answer (text, question_id)
values ('Counting sort is an integer sorting algorithm used in computer science to collect objects according to keys that are small positive integers. It operates by counting the number of objects that possess distinct key values, and applying prefix sum on those counts to determine the positions of each key value in the output sequence
Counting sort is efficient if the range of input data is not significantly greater than the number of objects to be sorted. Counting sort is frequently used as a subroutine in radix sort, a more efficient sorting method for larger keys.
Bucket sort, also known as bin sort, is a sorting algorithm that works by distributing the elements of an array into several buckets. Each bucket is then sorted individually, either using a different sorting algorithm or by recursively applying the bucket sorting algorithm itself
The steps involved in the bucket sort algorithm are:
    Scatter: Go through the original array, placing each object in its appropriate bucket.
    Sort: elements are sorted using a stable sorting algorithm.
    Gather: Return all elements to the original array after visiting the buckets in order.
The bucket sort algorithm is based on the assumption that the input is drawn from a uniform distribution. In terms of use cases, bucket sort is useful when input is uniformly distributed across a range. It is also useful when you want to sort floating point numbers that range from 0.0 to 1.0', 47);

insert into question (id, text, topic_id) 
values (48, 'What are the advantages of the heap over a stack?', 6);
insert into answer (text, question_id)
values ('The heap allows for dynamic memory allocation, meaning that memory can be allocated and deallocated as needed during runtime, while the stack has a fixed size that cannot be changed.', 48);
insert into answer (text, question_id)
values ('The heap can store larger amounts of data than the stack, as the heap can grow to fill available memory, while the stack is limited by its fixed size.', 48);
insert into answer (text, question_id)
values ('The heap allows for non-contiguous memory allocation, meaning that data can be stored in different locations in memory and still be accessed efficiently, while the stack requires contiguous memory allocation.', 48);
insert into answer (text, question_id)
values ('The heap allows for data to persist beyond the scope of a function or program, while the stack only stores data temporarily during a function call.', 48);
insert into answer (text, question_id)
values ('The heap allows for multiple threads to access and modify data simultaneously, while the stack is only accessible by the currently executing thread.', 48);

insert into question (id, text, topic_id) 
values (49, 'List the types of trees in data structures with a short description of each type.', 6);
insert into answer (text, question_id)
values ('Binary Tree: A data structure where each node has at most two children, referred to as the left child and the right child. It is commonly used for efficient searching and sorting operations.
AVL Tree: A self-balancing binary search tree, where the heights of the left and right subtrees of any node differ by at most one. This ensures faster search, insertion, and deletion operations.
B-tree: A self-balancing search tree that can have multiple children for each node. It is commonly used in databases and file systems due to its ability to efficiently store and retrieve large amounts of data.', 49);
insert into answer (text, question_id)
values ('Red-Black Tree: A self-balancing binary search tree where each node has an extra attribute, the color, which can be either red or black. It ensures that no path from the root to a leaf is more than twice as long as any other path, making it efficient for various operations.
Trie: A tree-like data structure used for efficient retrieval of strings or words. Each node represents a prefix or a complete word, and the paths from the root to the leaf nodes form all possible words in the structure. It is commonly used in applications like autocomplete and spell checkers.', 49);
insert into answer (text, question_id)
values ('N-ary Tree: In this sort of tree with a node, N is the maximum number of children. A binary tree is a two-year tree since each binary tree node has no more than two offspring. A full N-ary tree is one in which the children of each node are either 0 or N.
Binary Tree: This is a type of data structure where each node can have a maximum of two offspring, known as the left child and the right child. It is frequently employed for effective searching and sorting tasks.
AVL Tree: A self-balancing binary search tree in which each node has left and right subtrees that are at most one height apart. This guarantees quicker insertion, deletion, and search operations.', 49);
insert into answer (text, question_id)
values ('B-tree: A self-balancing search tree with the ability to have several children for each node. Due to its effectiveness in storing and retrieving massive volumes of data, it is frequently employed in databases and file systems.
Red-Black Tree: A self-balancing binary search tree with an additional attribute—color, which can be either red or black—that each node possesses. It makes it effective for a variety of tasks by ensuring that no path from the root to a leaf is more than twice as long as any other path.', 49);
insert into answer (text, question_id)
values ('Trie: A tree-like data structure for fast word or string retrieval. The paths from the root to the leaf nodes constitute all conceivable words in the structure, and each node represents a prefix or a full word. Applications like autocomplete and spell checkers frequently use it.
N-ary Tree: The maximum number of children in this type of tree with a node is N. Since each binary tree node can only produce two offspring, a binary tree is a two-year tree. The offspring of each node in a full N-ary tree are either 0 or N.', 49);

insert into question (id, text, topic_id) 
values (50, 'How dynamic memory allocations help in managing data?', 6);
insert into answer (text, question_id)
values ('Dynamic memory allocations allow for efficient memory management by allowing the allocation and deallocation of memory at runtime. This means that memory can be allocated as needed, reducing wastage and optimizing the use of available memory resources.', 50);
insert into answer (text, question_id)
values ('Dynamic memory allocations enable the creation of data structures with variable sizes, such as resizable arrays or linked lists. This flexibility allows for efficient storage and manipulation of data, as the size can be adjusted based on the current needs of the program.', 50);
insert into answer (text, question_id)
values ('Dynamic memory allocations facilitate the efficient management of large datasets. By allocating memory dynamically, programs can handle large amounts of data without requiring a fixed amount of memory from the start. This allows for more efficient use of system resources and improves overall performance.', 50);
insert into answer (text, question_id)
values ('Dynamic memory allocations support dynamic data structures, such as trees or graphs, where the number of elements and their relationships can change over time. With dynamic memory allocations, nodes or vertices can be created and removed as needed, enabling efficient data management in complex data structures.', 50);
insert into answer (text, question_id)
values ('Dynamic memory allocations help prevent memory leaks and improve memory utilization. By dynamically allocating memory and deallocating it when no longer needed, programs can avoid wasting memory resources and ensure that memory is freed up for other purposes. This helps maintain the overall stability and efficiency of the program.', 50);

-- SELECT topic.name as topic_name, question_id, question.text as question, answer.id as answer_id, answer.text as answer
-- FROM topic join question on topic.id = question.topic_id 
-- join answer on question.id = answer.question_id
-- order by question_id;

-- select * from topic;

with get_count as (select distinct topic.name, count(question.id) over(partition by topic.id)
from topic join question on topic.id = question.topic_id 
order by topic.name)
select *, sum(count) over()
from get_count;