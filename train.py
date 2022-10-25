from lib import *
from utils import train_model
from config import *
from module import *

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    global_step = 0
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese')
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    # Model parameter
    MAX_SEQ_LEN = 10
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

    fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

    # Tabular Dataset
    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                            test='test.csv', format='CSV', fields=fields, skip_header=True)

    # Iterator
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    global_step = 0
    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    train_model(device=device,model=model, optimizer=optimizer, train_loader=train_iter, valid_loader=valid_iter, eval_every=len(train_iter)//2)
    #load_metrics(destination_folder + '/metrics.pt')
    #load_checkpoint(destination_folder + '/model.pt', model) 
if __name__ == "__main__":
    main()