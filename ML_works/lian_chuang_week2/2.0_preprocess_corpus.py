


def get_stopwords():
    stopwords = []
    stopwords = [' '+line.strip()+' ' for line in open(
        R'C:\Users\xiong35\Desktop\corpus\stopped_word.txt').readlines()]
    return stopwords


def preprocess(fname,new_file):
    f = open(fname, 'r')
    lines = f.readlines()
    new_f = open(new_file,'a')
    stopwords = get_stopwords()
    to_replace = [',','.','-','_','?','!','”','“','’','‘','[',']','–',';']
    for i,line in enumerate(lines):
        if i%10 == 0:
            print('{} is done'.format(i))
            new_f.write('\n')
        line = line.strip('\n').lower()
        for word in stopwords:
            if word in line:
                line = line.replace(word,' ')
        for mark in to_replace:
            line = line.replace(mark,' ')
        for num in '0123456789':
            line = line.replace(num,'')
        new_f.write(line)
    f.close()
    new_f.close()

fname = R'C:\Users\xiong35\Desktop\corpus\Holmose.txt'
new_file = R'C:\Users\xiong35\Desktop\corpus\new_h.txt'

preprocess(fname,new_file)