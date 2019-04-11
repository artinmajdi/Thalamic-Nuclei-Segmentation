import numpy as np
import os, sys
sys.path.append('/array/ssd/msmajdi/code/thalamus/keras')
# sys.path.append( os.path.dirname(os.path.dirname(__file__)) )
import otherFuncs.smallFuncs as smallFuncs
# import Parameters.UserInfo as UserInfo
# import Parameters.paramFunc as paramFunc
# import keras
# import h5py
# import nibabel as nib

# params = paramFunc.Run(UserInfo.__dict__, terminal=True)
# K = smallFuncs.gpuSetting(params.WhichExperiment.HardParams.Machine.GPU_Index)

class Person:
    def __init__(self, name, surname, number):
        self.name = name
        self.surname = surname
        self.number = number

    def funcTest(self, aa):
        print('yeap', aa)
class Student(Person):
    UNDERGRADUATE, POSTGRADUATE = range(2)

    def __init__(self, student_type, canDance ,  *args, **kwargs):
        self.student_type = student_type
        self.canDance = canDance
        self.classes = []
        super().__init__(*args, **kwargs)

b = smallFuncs.Nuclei_Class(1,'Cascade').All_Nuclei().Indexes
# c = b # .All_Nuclei()


a = Student(Student.UNDERGRADUATE, 'yes' ,  'artin' ,surname='majdi' , number='5202442240')
a.funcTest('ssss')
print('--')
