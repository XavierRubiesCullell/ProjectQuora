import utils

def batch_generator(data, target, batch_size):
    data = np.array(data)
    target = np.array(target)
    nsamples = len(data)
    perm = np.random.permutation(nsamples)
    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i+batch_size]
        if target is not None:
            yield data[batch_idx,:], target[batch_idx]
        else:
            yield data[batch_idx], None

def training(model, train_data, train_target, args):

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = AdamW(model.parameters(), lr=args['learning_rate'], correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['num_warmup_steps'],
                                                num_training_steps=args['num_training_steps'])
    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']

    for X, y in batch_generator(train_data, train_target, batch_size):
        #model.train()
        X_i, X_s, X_p, y = utils.ToTensor(X,y)
        
        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p, labels=y)[1]
        
        loss = criterion(out, y)
        loss.backward()
        total_loss += loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])  # Gradient clipping 
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        out = F.softmax(out, dim=1)

        ncorrect += (torch.max(out, 1)[1] == y).sum().item()
        print("train:", loss.item())
    total_loss /= len(train_data)
    acc = ncorrect/len(train_data) * 100
    return acc, loss

def validation(model, eval_data, eval_target, args):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']

    for X, y in batch_generator(eval_data, eval_target, batch_size):
        #model.train()
        X_i, X_s, X_p, y = utils.ToTensor(X,y)
        
        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p, labels=y)[1]
        
        loss = criterion(out, y)   
        total_loss += loss
        out = F.softmax(out, dim=1)
        ncorrect += (torch.max(out, 1)[1] == y).sum().item()
        print("validation:", loss.item())

    total_loss /= len(eval_data)
    acc = ncorrect/len(eval_data) * 100
    return acc, loss

def build(learn_data, model_class, pretrained_model, args):
    model = model_class.from_pretrained(pretrained_model, num_labels=2)
    print("Model loaded")

    epochs = args['epochs']

    X_train, X_val, y_train, y_val = train_test_split(learn_data.iloc[:,:-1], learn_data.iloc[:,-1],                                                            test_size=0.2, random_state=47)

    train_acc = [None]*epochs
    train_loss = [None]*epochs
    val_acc = [None]*epochs
    val_los = [None]*epochs
    for epoch in range(epochs):
        t_acc, t_loss = training(model, X_train, y_train, args)
             
        train_acc[epoch] = t_acc
        train_loss[epoch] = t_loss
        
        v_acc, v_loss = validation(model, X_val, y_val, args)
        val_acc[epoch] = v_acc
        val_loss[epoch] = v_loss

def test(model, test_data, args):
    ncorrect = 0
    total_loss = 0

    batch_size = args['batch_size']

    for X, _ in batch_generator(test_data, batch_size):
        #model.eval()
        X_i, X_s, X_p = utils.ToTensor(X,y=None)
        
        out = model(input_ids=X_i, token_type_ids=X_s, attention_mask=X_p)[0]
        out = F.softmax(out, dim=1)

        ncorrect += (torch.max(out, 1)[1] == y).sum().item()

    acc = ncorrect/len(test_data) * 100
    return acc