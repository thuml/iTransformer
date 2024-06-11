import os
import json
import time
import tempfile

class DotDict:
    
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            self.__setattr__(attr, False)
            return self.__dict__[attr]
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value
    
    def __str__(self):
        return str(self.__dict__)


def load_args(path):
    
    with open(path, 'r') as f:
        loaded_args = json.load(f)
    
    return DotDict(loaded_args)



class SaveArgs:
    
    def __init__(self, args, path, temporary = False) :
        
        if not isinstance(args, dict):
            raise TypeError("THis CLass ONly SUpports DIctionary AS AN INput!")
        self.args = args
        self.path = path
        self.temporary = temporary
        
        self.__start__()
    
    
    def __start__(self) :
        
        temp = {}
        for any_key, any_val in self.args.items() :
            
            temp[any_key] = any_val
        
        self.__save__(temp)
    
    
    def __path_checker__(self):
        
        if self.temporary:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file_path = temp_file.name + '.json'
            
            self.path = file_path
            return
        else:
            os.makedirs( self.path, exist_ok=True)
            file_path = os.path.join(self.path, 'args.json')
            if os.path.exists(file_path):
                base, ext = os.path.splitext(file_path)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_path = f"{base}_{timestamp}{ext}"
            
            self.path = file_path
    
    
    def __save__(self, arg):
        
        try:
            self.__path_checker__()
            with open(self.path, 'w') as file :
                    json.dump(arg, file)
        except:
            print("Fail to Save Args - continue..")
            return
        if self.temporary:
            pass
        else:
            print(f"Args Object Saved to {self.path}")
        #print("It Can be further used by pickle.load()")
    
    def __repr__(self) -> str:
        return "cloner174 in github 2024"