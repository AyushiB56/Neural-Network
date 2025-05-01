def train_batch():
    n= 1000
    input_dim=10
    X= torch.randn(n,input_dim)#we have 1000 rows/ records with each row has 10 features
    y = torch.randint(0,2,(n,), dtype=torch.float)
    learning_rate=0.001

    W1= torch.tensor(np.random.uniform(0,1,(3,10)), dtype=torch.float, requires_grad=True) # first layer
    W2= torch.tensor(np.random.uniform(0,1,(2,3)), dtype=torch.float, requires_grad=True)
    W3=torch.tensor(np.random.uniform(0,1,(1,2)), dtype=torch.float, requires_grad=True)
    bias1= torch.tensor(np.random.uniform(0,1,(3,)), dtype=torch.float, requires_grad=True)
    bias2= torch.tensor(np.random.uniform(0,1,(2,)), dtype=torch.float, requires_grad=True)
    bias3= torch.tensor(np.random.uniform(0,1,(1,)), dtype=torch.float, requires_grad=True)
    avg_loss=[]
    W_list=[W1,W2,W3]
    dw=[]
    bias_db=[]

    bias_list=[bias1,bias2,bias3]

    for epoch in range(100):
      Loss_value=0
      tic= time.time()
      for i in range(len(y)):

        Xin =X[i]
        yin = y[i].unsqueeze(0)  # unsqueeze(0) adds a dimension of size 1 at position 0, making the shape torch.Size([1])
        result = forward_pass(Xin,W_list,bias_list)
        Loss= nn.BCELoss()
        Loss_value+= Loss(result,yin)
        #avg_loss.append(Loss_value.item())
      Loss_value=Loss_value/n

      Loss_value.backward()
      for gradient in W_list:
          dw.append(gradient.grad)

      for gradient in bias_list:
          bias_db.append(gradient.grad)




      with torch.no_grad():
          for idx in range(len(W_list)):
            W_list[idx] -= learning_rate * dw[idx]
            bias_list[idx] -= learning_rate * bias_db[idx]


      for j in range(len(W_list)):
          W_list[j].grad.data.zero_()

      for j in range(len(bias_list)):
          bias_list[j].grad.data.zero_()

      toc= time.time()
      print(f"The average loss per epoch - {epoch}: {Loss_value}")
      print("time taken"+ str(1000*(toc-tic)) + "ms")

  # suppose the number of neurons are 3 then. total params to train are 10*3+3(bias)= 33

