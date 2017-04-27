entries = ['english data filename', 'english data vocab size', 'english data division difference',
           'english division number', 'farsi data filename', 'farsi data vocab size', 'translator encoder layer number',
           'translator encoder hidden size', 'translator attention hidden size', 'translator decoder layer number',
           'translator decoder hidden size', 'translator initial learning rate', 'translator name', 'epoch size',
           'batch size', 'print period', 'save period']
f = open('config.cfg', 'w')
for ent in entries:
    f.write(ent + '=' + input(ent + ':') + '\n')
f.close()

