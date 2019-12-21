import csv

class R_logger:
    def __init__(self, filename):
        ''' Constructor for this class. '''
        # Create some member animals
        self.filename = filename        
        self.fileopen = False

    def clear(self):
        self.f = open(self.filename, 'w')
  
    def write_line(self, data):
        if (self.fileopen == False):
            self.f = open(self.filename, 'a')
           # self.fileopen = True        
        self.f.writelines(data)
        self.f.close()
    
    def close(self):
        self.f.close()
