
"""
DualQuaternionClass.py
Copyright 2016 Hurchel Young
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
import numbers
from collections import Sequence

import numpy as np

from . import transformations

#from .transformations import *


#a Cartesian vector class for easy use with quaternions
#this class will allow access by XYZ or by index
class Vector(Sequence):
    def __init__(self, x=0, y=0, z=0):
        if isinstance(x, numbers.Real):
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)

        elif isinstance(x, (list, np.ndarray)):
            self._x = float(x[0])
            self._y = float(x[1])
            self._z = float(x[2])

        self._list = [self._x, self._y, self._z]
        super(Vector, self).__init__()

    #allow access to the vector values in list form
    def __getitem__(self, i):
        return self._list[i]

    #allow setting of the vector values in list form
    def __setitem__(self, i, value):
        if i < 3:
            self._list[i] = float(value)
            self._x = self._list[0]
            self._y = self._list[1]
            self._z = self._list[2]
        else:
            raise Exception('index out of range')

    # x set and get
    def _getX(self):
        return self._x
    def _setX(self, value):
        self._x = float(value)
        self._list[0] = self._x

    # y set and get
    def _getY(self):
        return self._y
    def _setY(self, value):
        self._y = float(value)
        self._list[1] = self._y

    # z set and get
    def _getZ(self):
        return self._z
    def _setZ(self, value):
        self._z = float(value)
        self._list[2] = self._z

    #define the properties functions for xyz
    x = property(_getX, _setX)
    y = property(_getY, _setY)
    z = property(_getZ, _setZ)

    #define the length of the vector, which is always 3
    def __len__(self):
        return 3

    #define the string representatnion
    def __repr__(self):
        return repr(self._list)

    #multiplication of a vector by another vector, or by a scaler
    #multiplication by vector returns the cross product
    def __mul__(self, v2):
        if isinstance(v2, (Vector, list)):
            return Vector(self[1]*v2[2] - self[2]*v2[1], self[2]*v2[0] - self[0]*v2[2], self[0]*v2[1] - self[1]*v2[0])
        elif isinstance(v2, numbers.Real):
            return Vector(self.x * v2, self.y * v2, self.z * v2)
        else:
            raise Exception('cannot multiply vector by ' + str(type(v2)))

    #vector addition
    def __add__(self, v2):
        if isinstance(v2, (Vector, list)):
            return Vector(self[0]+v2[0], self[1]+v2[1], self[2]+v2[2])
        else:
            raise Exception('cannot add vector to ' + str(type(v2)))

    #vector subtraction
    def __sub__(self, v2):
        if isinstance(v2, (Vector, list)):
            return Vector(self[0]-v2[0], self[1]-v2[1], self[2]-v2[2])
        else:
            raise Exception('cannot add vector to ' + str(type(v2)))

    @property
    def magnitude(v):
        #future, we may want to raise exception for magnitude of zero
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @property
    def normalize(v):
        vMagnitudeInverse = (1.0 / v.magnitude) if (v.magnitude > 0) else 0.0
        return Vector(v[0]*vMagnitudeInverse, v[1]*vMagnitudeInverse, v[2]*vMagnitudeInverse)

    #return the dot product
    def dot(self, v2):
        return self.x*v2[0] + self.y*v2[1] + self.z*v2[2]





#this class will allow a Quaternion to be referenced by wxyz properties or by the 0-3 index
#Quaternions will be in the form q = (w,x,y,z)
class Quaternion(Sequence):
    #initialize a quaternion either as an identity quaternion or with specified values
    #x can be a Real, a list, or a Vector
    def __init__(self, w=1, x=0, y=0, z=0):

        if isinstance(w, numbers.Real):
	        self._w = float(w)

        elif isinstance(w, Quaternion):
            self._w = float(w.w)
            self._x = float(w.x)
            self._y = float(w.y)
            self._z = float(w.z)

        if isinstance(x, numbers.Real) and isinstance(w, numbers.Real):
            self._x = float(x)
            self._y = float(y)
            self._z = float(z)

        #even though we can accept our custom Vector we must treat it as a list
        elif isinstance(x, (list, Vector)):
            self._x = float(x[0])
            self._y = float(x[1])
            self._z = float(x[2])

        self._list = [self._w, self._x, self._y, self._z]
        super(Quaternion, self).__init__()

    #allow access to the quaternion values in list form
    def __getitem__(self, i):
        return self._list[i]

    #allow setting of the quaternion values in list form
    def __setitem__(self, i, value):
        if i < 4:
            self._list[i] = float(value)
            self._w = self._list[0]
            self._x = self._list[1]
            self._y = self._list[2]
            self._z = self._list[3]
        else:
            raise Exception('index out of range')

    # w set and get
    def _getW(self):
        return self._w
    def _setW(self, value):
        self._w = float(value)
        self._list[0] = self._w

    # x set and get
    def _getX(self):
        return self._x
    def _setX(self, value):
        self._x = float(value)
        self._list[1] = self._x

    # y set and get
    def _getY(self):
        return self._y
    def _setY(self, value):
        self._y = float(value)
        self._list[2] = self._y

    # z set and get
    def _getZ(self):
        return self._z
    def _setZ(self, value):
        self._z = float(value)
        self._list[3] = self._z

    #define the properties functions for wxyz
    w = property(_getW, _setW)
    x = property(_getX, _setX)
    y = property(_getY, _setY)
    z = property(_getZ, _setZ)

    #define the length of the quaternion, which is always 4
    def __len__(self):
        return 4

    #define the string representatnion
    def __repr__(self):
        return repr(self._list)

    #multiplication of a quaternion by another quaternion, or by a scaler
    def __mul__(self, q2):
        if isinstance(q2, Quaternion):
            return Quaternion(-self.x * q2.x - self.y * q2.y - self.z * q2.z + self.w * q2.w,
                              self.x * q2.w + self.y * q2.z - self.z * q2.y + self.w * q2.x,
                              -self.x * q2.z + self.y * q2.w + self.z * q2.x + self.w * q2.y,
                              self.x * q2.y - self.y * q2.x + self.z * q2.w + self.w * q2.z)
        elif isinstance(q2, numbers.Real):
            return Quaternion(self.w * q2, self.x * q2, self.y * q2, self.z * q2)
        else:
            raise Exception('cannot multiply quaternion by ' + str(type(q2)))

    #addition of quaternions
    def __add__(self, q2):
        if isinstance(q2, Quaternion):
            return Quaternion(self.w + q2.w,
                              self.x + q2.x,
                              self.y + q2.y,
                              self.z + q2.z)
        else:
              raise Exception('cannot add ' + str(type(q2)) + ' to a quaternion')

    @property
    def magnitude(q):
        return math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)

    @property
    def normalize(q):
        qMagnitudeInverse = (1.0 / q.magnitude) if q.magnitude > 0 else 0.0
        return q * qMagnitudeInverse

    @property
    def conjugate(q):
        return Quaternion(q.w,-q.x,-q.y,-q.z)

    #extract the angle of rotation
    @property
    def getRotationAngle(q):
        return 2.0 * math.acos(q.w)

    @property
    def inverse(q):
        return q.conjugate * (1.0 / q.magnitude**2)

    def dot(q1, q2):
        return q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    #return a unit quaternion for rotation about an axis
    @staticmethod
    def quaternionAboutAxis(radians, v):
        v = Vector(v[0], v[1], v[2]).normalize
        newQ = Quaternion(0, v.x, v.y, v.z)
        #since the w of newQ has not been set yet we can multiply newQ by
        #the result of sin(n/theta) and it will only be applyed to each of the vector values
        newQ = newQ * math.sin(radians*0.5)
        #finally set the w value
        newQ.w = math.cos(radians*0.5)
        return newQ.normalize




#all dual quaternions returned by this class will be normalized
class DualQuaternion(object):
    #this init will take no args and return an identy dual quaternion
    # or 2*quaternions if you have them prepared
    # or a quaternion and a vector (to represent the translation)
    def __init__(self, q1 = Quaternion(1,0,0,0), q2 = Quaternion(0,0,0,0)):

        if isinstance(q1, Quaternion):
            self.mReal = q1.normalize
        elif isinstance(q1, list) and len(q1)==4:
            self.mReal = Quaternion(w=q1[0], x=q1[1], y=q1[2], z=q1[3]).normalize
        elif isinstance(q1, np.ndarray):
            matrix = q1.copy()
            w,x,y,z = transformations.quaternion_from_matrix(matrix)
            self.mReal = Quaternion(w,x,y,z)
            dx, dy, dz = transformations.translation_from_matrix(matrix)

            w,x,y,z = transformations.quaternion_multiply(Quaternion(0, dx / 2, dy / 2, dz / 2), self.mReal)

            self.mDual=Quaternion(w,x,y,z)
            q2=self.mDual
        else:
            raise Exception('could not create dual Quaternion from first arg')

        if isinstance(q2, Quaternion):
            self.mDual = q2
        elif isinstance(q2, (list, Vector)):
            self.mDual = (Quaternion(0, q2[0],q2[1],q2[2]) * self.mReal) * 0.5

        else:
            raise Exception('could not create dual Quaternion from second arg')


    #return the formated string value for printing
    def __repr__(self):
        return repr(self.mReal) + '\n' + repr(self.mDual)

    #define multiplication for dual quaternions
    def __mul__(self, dq2):
        #if we are multiplying against another dualQuaternion
        if isinstance(dq2, DualQuaternion):
            return DualQuaternion(dq2.mReal * self.mReal,
                                  dq2.mDual * self.mReal + dq2.mReal * self.mDual)
        #if we are multiplying against a number as a scalar then we use quaternion multiplication
        elif isinstance(dq2, numbers.Real):
            #by returning a new quaternion the real part will be normalized
            return DualQuaternion(self.mReal * dq2,
                                  self.mDual * dq2)
        else:
            raise Exception('cannot multiply quaternion by ' + str(type(dq2)))

    #define addition for dual quaternions
    def __add__(self, dq2):
        if isinstance(dq2, DualQuaternion):
            return DualQuaternion(self.mReal + dq2.mReal, self.mDual + dq2.mDual)
        else:
            raise Exception('cannot add ' + str(type(dq2)) + ' to quaternion')

    def toArray(self):
        return np.array([self.mReal.w,self.mReal.x,self.mReal.y,self.mReal.z, self.mDual.w,self.mDual.x,self.mDual.y,self.mDual.z])
    #normalization of a dual quaternion
    @property
    def normalize(dq1):
         magInverse = (1.0 / Quaternion.dot( dq1.mReal, dq1.mReal )) if Quaternion.dot( dq1.mReal, dq1.mReal ) > 0 else 0.0
         newDQ = DualQuaternion()
         newDQ.mReal = dq1.mReal * magInverse
         newDQ.mDual = dq1.mDual * magInverse
         return newDQ

    #conjugate of a dual quaternion
    @property
    def conjugate(dq1):
        return DualQuaternion(dq1.mReal.conjugate, dq1.mDual.conjugate)

    #extract the rotation of a dual quaternion
    @property
    def getRotation(dq1):
        return dq1.mReal

    #extract the translation of a dual quaternion
    @property
    def getTranslation(dq1):
        q3 = (dq1.mDual * 2.0) * dq1.mReal.conjugate
        return Vector(q3.x, q3.y, q3.z)

    #extract the orientation vector
    @property
    def getOrientationVector(dq1):
        return Vector(dq1.mReal.x, dq1.mReal.y, dq1.mReal.z).normalize

    #extract the angle of rotation
    @property
    def getRotationAngle(dq1):
        return 2.0 * math.acos(dq1.mReal.w)

    #dot product of two dual quaternions
    def dot(dq1, dq2):
        return Quaternion.dot(dq1.mReal, dq2.mReal)

    #convert a dual quaternion to a 4x4 matrix
    #this will contain rotation and translation information
    def dualQuaternionToMatrix(dq1):
        #make sure the dq is normalized
        newDQ = dq1.normalize
        #pull out the real quaternion parameters for easy use
        w = newDQ.mReal.w
        x = newDQ.mReal.x
        y = newDQ.mReal.y
        z = newDQ.mReal.z
        #create a 4x4 identity matrix
        mat = np.identity(4)
        #Extract rotational information into the new matrix
        mat[0][0] = w*w + x*x - y*y - z*z
        mat[0][1] = 2*x*y + 2*w*z
        mat[0][2] = 2*x*z - 2*w*y
        mat[1][0] = 2*x*y - 2*w*z
        mat[1][1] = w*w + y*y - x*x - z*z
        mat[1][2] = 2*y*z + 2*w*x
        mat[2][0] = 2*x*z + 2*w*y
        mat[2][1] = 2*y*z - 2*w*x
        mat[2][2] = w*w + z*z - x*x - y*y
        #Extract translation information into the new matrix
        q1 = (newDQ.mDual * 2.0) * newDQ.mReal.conjugate
        mat[3][0] = q1.x
        mat[3][1] = q1.y
        mat[3][2] = q1.z
        #return the new matrix
        return mat

    def normalizeDual(self, factor):
        self.mDual.x *= factor
        self.mDual.y *= factor
        self.mDual.z *= factor

    def getDualMaximum(self):
        np.max([self.mDual.x, self.mDual.y, self.mDual.z])
