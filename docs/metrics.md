# Common NLP Metrics & Applications

| Metric        | Description                                                                                                                                                                                                                  | Applications                                                                                                                                                                             |  |  |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|
| **Accuracy**  | Used when the output variable is categorical or discrete. It denotes the fraction of times the model makes correct predictions as compared to the total predictions it makes.                                                | Mainly used in classification tasks, such as `sentiment classification` (multiclass),  `natural language inference` (binary), `paraphrase detection` (binary), etc.                      |  |  |
| **Precision** | Shows how precise or exact the model’s predictions are, i.e., given all the positive (the class we care about) cases, how many can the model classify correctly?                                                             | Used in various classification tasks, especially in cases where mistakes in a positive class are more costly than mistakes in a negative class, e.g., disease predictions in healthcare. |  |  |
| **Recall**    | Recall is complementary to precision. It captures how well the model can recall positive class, i.e., given all the positive predictions it makes, how many of them are indeed positive?                                     | Used in classification tasks, especially where retrieving positive results is more important, e.g., e-commerce search and other information-retrieval tasks. F1                          |  |  |
| **F1 score**  | Combines precision and recall to give a single metric, which also captures the trade-off between precision and recall, i.e., completeness and exactness. F1 is defined as `(2 × Precision × Recall) / (Precision + Recall)`. | Used simultaneously with accuracy in most of the classification tasks. It is also used in sequence-labeling tasks, such as entity extraction, retrieval-based questions answering, etc.  |  |  |
| **AUC**       | Captures the count of positive predictions that are correct versus the count of positive predictions that are incorrect as we vary the threshold for prediction.                                                             | Used to measure the quality of a model independent of the prediction threshold. It is used to find the optimal prediction threshold for a classification task.                           |  |  |
| **MRR**       | Used to evaluate the responses retrieved given their probability of correctness. It is the mean of the reciprocal of the ranks of the retrieved results.                                                                     | Used heavily in all information-retrieval tasks, including article search, e-commerce search, etc.                                                                                       |  |  |
| **MAP**       | Used in ranked retrieval results, like MRR. It calculates the mean precision across each retrieved result.                                                                                                                   | Used in information-retrieval tasks                                                                                                                                                      |  |  |