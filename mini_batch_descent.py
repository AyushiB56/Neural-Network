def train(examples, learning_rate=0.01, batch_size=10, epochs=100):

  weights_ab = np.random.normal(0, 1, 3 + 1)
  print(weights_ab)
  feature= examples["features"]
  label=examples["label"]
  weights= weights_ab[:-1]
  bias= weights_ab[-1]
  n_samples=examples["features"].shape[0]

  for epoch in range(epochs):
    for start in range(0,n_samples, batch_size):
      end= start+batch_size
      x_batch=feature[start:end]
      y_batch= label[start:end]

      z=np.dot(x_batch,weights)+bias
      y_pred = 1/(1+np.exp(-z))
      error = y_pred-y_batch
      dw= np.dot(x_batch.T,error)/batch_size
      db= error/batch_size

      #update weight

      weights= weights- learning_rate*dw
      bias= bias- learning_rate*db
  print("Training complete. Final weights:", weights)



    

