import numpy as np
model = Model(vocab_size, d_model, num_heads, num_layers, max_seq_length).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
model.train()
for epoch in range(10):
    running_loss=[]
    optimizer.zero_grad()
    for data in tqdm(loader):
        batch = tuple(t.to(device) for t in data)
        inputs, labels = batch
        output = model(inputs)
        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {np.average(running_loss)}")
