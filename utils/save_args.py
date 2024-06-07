# cloner174
# enhanced version files

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


class SaveArgs:
    
    def __init__(self, args, path) :
        
        if not isinstance(args, dict):
            raise TypeError("THis CLass ONly SUpports DIctionary AS AN INput!")
        self.args = args
        self.path = path
        self.arg_creator = DotDict
        
        self.__start__()
    
    def __start__(self) :
        
        temp = {}
        
        for any_key, any_val in self.args.items() :
            
            temp[any_key] = any_val
            
        self.__modify__(temp)
    
    def __modify__(self, dict):
        
        try:
            
            arg_new = self.arg_creator(dict)
        except:
            print("Fail to Save Args")
            return
        
        self.__save__(arg_new)
    
    def __path_checker__(self):
        
        try:
            import os
            if os.path.exists(self.path):
                pass
            else:
                try:
                    os.makedirs(self.path)
                except:
                    self.path = 'Args.pkl'
                    return True
            self.path = os.path.join(self.path, 'Args.pkl')
            return True
        except:
            return False
    
    def __save__(self, arg):
        
        try:
            import pickle
            if self.__path_checker__():
                with open(self.path, 'wb') as file :
                    pickle.dump(arg, file)
            else:
                print("Fail to Save Args")
                return
        except:
            print("Fail to Save Args")
            return
        print(f"Args Object Saved to {self.path}")
        print("It Can be further used by pickle.load()")
        
    
    def __repr__(self) -> str:
        return "cloner174 in github 2024"
    
#end#