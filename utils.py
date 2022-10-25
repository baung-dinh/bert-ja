from lib import *
from config import *

# Save and Load functions
def save_checkpoint(save_path, model, valid_loss):
    if save_path is None:
        return
    
    state_dict = {
        			'model_state_dict': model.state_dict(),
                  	'valid_loss': valid_loss
    			}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model, device):
    if load_path is None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    print(state_dict['valid_loss'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return
    
    state_dict = {
        			'train_loss_list': train_loss_list,
        			'valid_loss_list': valid_loss_list,
        			'global_steps_list': global_steps_list
    			}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')
   
def load_metrics(load_path, device):
    if load_path is None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'],state_dict['global_steps_list']


# Training function
def train_model(device, 
         model,
         optimizer,
         train_loader,
         valid_loader,
         eval_every,
         criterion=nn.BCELoss(),
         num_epochs=5,
         file_path=destination_folder,
         best_valid_loss=float('Inf')):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, title, text, titletext), _ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titletext = titletext.type(torch.LongTensor)
            titletext = titletext.to(device)
            output = model(titletext, labels)
            loss, _ = output
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update running values
            running_loss += loss.item()
            global_step +=1 
            
            # evaluation step
            if global_step % eval_every == 5:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for(labels, title, text, titletext), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        titletext = titletext.type(torch.LongTensor)
                        titletext = titletext.to(device)
                        output = model(titletext, labels)
                        loss, _ = output
                        
                        valid_running_loss += loss.item()
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / eval_every
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                
                # reset running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()
                
                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train loss: {:.4f}, Valid loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader), average_train_loss, average_valid_loss))
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
