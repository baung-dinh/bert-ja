
from lib import BertForSequenceClassification, nn
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        options_name = 'cl-tohoku/bert-base-japanese'
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)
	
    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea