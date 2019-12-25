import csv

class R_logger:
    def __init__(self, filename):
        ''' Constructor for this class. '''
        # Create some member animals
        self.filename = filename        
        self.fileopen = False

    def clear(self):
        self.f = open(self.filename, 'w')
  
    def open(self):
        self.f = open(self.filename, 'r')

    def write_line(self, data):
        if (self.fileopen == False):
            self.f = open(self.filename, 'a')
           # self.fileopen = True        
        self.f.writelines(data)
        self.f.close()
    
    def load_dictionary(self):
        my_dictionary = {}

        with open(self.filename, mode='r') as infile:
            reader = csv.reader(infile)
            my_dictionary = { rows[2]:rows[1] for rows in reader }

        return my_dictionary
    
    def close(self):
        self.f.close()
