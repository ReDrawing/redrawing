import inspect

import numpy as np

import redrawing.data_interfaces as data_interfaces


class Data2Java():
    '''!
        Converts redrawing.Data classes to Java class files
    '''

    def __init__(self, package="br.campinas.redrawing.data"):
        '''!
            Initializes the converter

            Parameters:
                @param package - The name of the Java Package
        '''

        self.data_list = []
        self.package = package

    def add_data_class(self, data_class):
        '''!
            Add a class to be converted

            Parameters:
                @param data_class - Data class
        '''
        self.data_list.append(data_class)

    def toJavaType(self, obj):
        '''!
            Converts a object type to Java type

            Parameters:
                @param obj - Object which class will be converted

            Returns:
                @returns string with Java class name
        '''

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
            
            ndim = 1

            item = obj[0]

            while isinstance(item, list):
                ndim += 1
                item = item[0]
            
            item_type = self.toJavaType(item)

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
        '''!
            Create the java class from a Data class

            Parameters:
                @param data_class - data class
            
            Returns:
                @returns string with the content of Java file
        '''

        java_class = "package "+self.package+";\n\n"
        java_class += "import java.util.Map;\n\n"

        java_class += "public class "+data_class.__name__+"{\n"

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

        return java_class, data_class.__name__

    def add_redrawing(self):
        '''!
            Adds all the Data class of redrawing.data_interfaces to be converted.
        '''
        for name, obj in inspect.getmembers(data_interfaces):
            if inspect.isclass(obj):
                if issubclass(obj, data_interfaces.Data):
                    if obj != data_interfaces.Data:
                        self.add_data_class(obj)

    def convert_all(self):
        '''!
            Converts all the add classes.

            Creates files in the executing path.
        '''

        for data_class in self.data_list:
            java_class, name = self.createJavaClass(data_class)

            with open(name+".java","w") as f:
                f.write(java_class)

if __name__ == '__main__':
    dj2 = Data2Java()
    dj2.add_redrawing()
    dj2.convert_all()