import unittest
from cmath import pi
from Network import overallModel
from DualQuaternion import transformations
from DualQuaternion.DualQuaternion import DualQuaternion
import numpy
import time
class TestQuaternionMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestQuaternionMethods, self).__init__(*args, **kwargs)
        self.quaternion_list = list()
        self.matrix_list = list()
        self.quaternion_list.append(numpy.array([1,1,1,1,1,1,1,1]))
        self.quaternion_list.append(numpy.array([1, 1, 1, 1, 1, 1, 1, 1]))

        self.matrix_list.append(numpy.array([[0,0,8,1],[8,0,0,1],[0,8,0,1],[0,0,0,1]]))
        self.matrix_list.append(numpy.array([[0,0,.5,1],[.5,0,0,1],[0,0.5,0,1],[0,0,0,1]]))


    def testReverse(self):
        s = ""
        for angle in range(0,20):
            for r0 in numpy.linspace(0,1,5):
                print(time.clock())
                for r1 in numpy.linspace(0,1,5):
                    for r2 in numpy.linspace(0,1,5):
                        for t0 in numpy.linspace(0, 1, 5):
                            for t1 in numpy.linspace(0, 1, 5):
                                for t2 in numpy.linspace(0, 1, 5):
                                    if(r0+r1+r2>0):
                                        s=str(r0) + ", " + str(r1) + ", " + str(r2) + ", " + str(t0) + ", " + str(t1) + ", " + str(t2) + ", " + str(angle)
                                        numpy.testing.assert_allclose(transformations.rotation_matrix(angle, (r0,r1,r2), (t0,t1,t2)), transformations.dual_quaternion_matrix(DualQuaternion(transformations.rotation_matrix(angle, (r0,r1,r2), (t0,t1,t2)))),atol=1e-10,err_msg=s)


if __name__ == '__main__':
    #overallModel.createRotationMatrix((1,0,0),pi)
    unittest.main()