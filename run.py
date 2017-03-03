# -*- encoding:utf-8 -*-

from csv import DictReader

def data(filePath, hashSize, hashSalt):
    ''' generator for data using hash trick
    '''

    for idx, row in enumerate(DictReader(
        open(filePath), delimiter=',')):
        id = row['id']
        del row['id']
        # lable of sample
        y = 0
        if 'click' in row:
            if row['click'].strip() == '1':
                y = 1.
            del row['click']
        # date of sample
        date = int(row['hour'][4:6])
        row['hour'] = row['hour'][6:]
        # hash features of sample
        x = []
        for key, value in row.iteritems():
            key = hashSalt + key + '_' + value
            feature_hash = abs(hash(key)) % hashSize + 1
            x.append(feature_hash)
        yield idx, date, id, x, y

def train(epoch, trainPath):
    pass

def main():
    pass

if __name__ == '__main__':
    main()

