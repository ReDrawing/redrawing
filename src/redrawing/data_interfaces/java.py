import numpy as np

class Data2Java():

    def __init__(self):
        self.data_list = []

    def add_data_class(self, data_class):
        self.data_list.append(data_class)

    def toJavaType(self, obj):
        

        if type(obj) in self.data_list:
            return type(obj).__name__

        if type(obj) == int:
            return "int"
        if type(obj) == float:
            return "float"
        if type(obj) == str:
            return "String"
        if type(obj) == bool:
            return "boolean"

        if type(obj) == np.ndarray:
            obj = obj.tolist()

        if type(obj) == list:
            item_type = self.toJavaType(obj[0])
            
            ndim = 1

            item = obj[0]

            while isinstance(item, list):
                ndim += 1
                item = item[0]

            dim = "[]" *ndim

            return item_type+dim


        if type(obj) == dict:
           
            keys = list(obj.keys())
            values = list(obj.values())

            key_type = self.toJavaType(keys[0])
            value_type = self.toJavaType(values[0])

            for key in keys:
                if self.toJavaType(key) !=  key_type:
                    key_type = "Object"
                    break
            for value in values:
                if self.toJavaType(value) != value_type:
                    value_type = "Object"
                    break

            return "Map<"+key_type+","+value_type+">"
        
        raise Exception("Can't handle type"+str(type(obj)))
    
    def createJavaClass(self, data_class):
        java_class = "public class "+data_class.__name__+"{\n"

        data = data_class()

        names = list(data.__dict__.keys())
        objs = list(data.__dict__.values())

        for i in range(len(names)):
            javaType = self.toJavaType(objs[i])

            name = names[i]

            while name[0] == '_':
                name = name[1:]

            java_class += "\tpublic "
            java_class += javaType
            java_class += " "
            java_class += name
            java_class += ";\n"
        
        java_class += "}\n"

        return java_class
