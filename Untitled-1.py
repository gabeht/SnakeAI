from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import keyboard as kb
import cv2
import copy
import random
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

resolution = (512,512)

class enviroment(py_environment.PyEnvironment):
    _episode_ended = False
    _action_spec = array_spec.BoundedArraySpec(shape=(),dtype=np.int32,minimum=0,maximum=4,name="action")
    _observation_spec =array_spec.BoundedArraySpec(shape=(5+15*15,),dtype=np.float32,minimum=0.0,maximum=1.0,name="observation")
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def _step(self,action):
        if self._episode_ended:
            return self.reset()
        if action >= 0 and action <= 3:
            self._state,reward,self._episode_ended = self.runFrame(action)
        else:
            print(action)
            raise ValueError("your moma got too fat")
        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype= np.float32),reward)
        else:
            return ts.transition(np.array(self._state, dtype= np.float32),reward,discount = 0.9)
    def _reset(self):
        self._state = self.reset()
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype= np.float32))
            
    
    width = 15
    height = 15
    area = height*width
    def reset(self):
        self.score = 0
        self.snakeBone = [[3,1],[2,1],[1,1]]
        self.snakeHead = [4,1]
        self.fruit = [self.snakeBone[0][0]+4,self.snakeBone[0][1]]
        self.direction = 0
        spaceFruit = [self.fruit[0] - self.snakeHead[0], self.fruit[1] - self.snakeHead[1]]
        if self.drawSelf:
            image = env.drawImage()
            cv2.imshow("Snake game that is super duper rad",image)
            cv2.waitKey(100)
        return self.combineData(spaceFruit)       
    def __init__(self,drawSelf=False):
        self.drawSelf = drawSelf
         
    def runFrame(self,action):
        reward = 0
        #direction right = 0, up = 1, left = 2 ,down = 3,
        if (self.direction - action)%2 == 1:
            self.direction = action
        self.snakeBone.insert(0,copy.deepcopy(self.snakeHead))
        angle = self.direction * (np.pi / 2)
        self.snakeHead[0]+=round(np.cos(angle))
        self.snakeHead[1]+=round(np.sin(angle))
        if self.drawSelf:
            image = env.drawImage()
            cv2.imshow("Snake game that is super duper rad",image)
            cv2.waitKey(100)
        if(self.snakeHead == self.fruit):
            self.score+=1
            reward += 10
            self.fruit = self.generateFruit()
        else:
            self.snakeBone.pop()
        spaceFruit = [self.fruit[0] - self.snakeHead[0], self.fruit[1] - self.snakeHead[1]]
        if(self.snakeHead[0]<=0)or(self.snakeHead[0]>self.width)or(self.snakeHead[1]<=0)or(self.snakeHead[1]>self.height):
            reward -= 50
            return self.combineData(spaceFruit),reward, True
        if(self.snakeHead in self.snakeBone):
            return self.combineData(spaceFruit),reward,True
        return self.combineData(spaceFruit),reward,False
            
    def combineData(self,spaceFruit):
        finalList =[]
        finalList.append((spaceFruit[0]+15)/30)
        finalList.append((spaceFruit[1]+15)/30)
        finalList.append(((self.direction-0)/3))
        finalList.append((self.snakeHead[0]-1/14))
        finalList.append((self.snakeHead[1]-1/14))
        for i in range(1,16):
            for j in range(1,16):
                if self.inSnake([i,j]):
                    finalList.append(1)
                else:
                    finalList.append(0)
        return finalList
                    
                
    
    def inSnake(self,cord):
        for j in self.snakeBone:
                if abs(cord[0]-j[0]) <=0.001 and abs(cord[1]-j[1]) <=0.001:
                    return True
        if abs(cord[0]-self.snakeHead[0]) <=0.001 and abs(cord[1]-self.snakeHead[1]) <=0.001:
            return True
        return False
                
        
        
    
    def generateFruit(self):
        freeSpace = []
        for i in range(1,16):
            for j in range(1,16):
                freeSpace.append([i,j])
        finalFreeSpace = self.contain(freeSpace,self.snakeBone)
        finalFreeSpace = self.contain(finalFreeSpace,[self.snakeHead])
        return random.choice(finalFreeSpace)
    
    def contain(self,list1,list2):
        list3 =[]
        for i in list1:
            inList = False
            for j in list2:
                if abs(i[0]-j[0]) <=0.001 and abs(i[1]-j[1]) <=0.001:
                    inList = True
                    break
            if not inList:
                list3.append(i)
        return list3
            
    
    def drawImage(self):
        image = np.zeros((resolution[1],resolution[0],3),dtype=np.uint8)
        pt1,pt2 = self.ConvertPix(self.snakeHead)
        image = cv2.rectangle(image,pt1,pt2,(0,255,1),-1)
        for bone in self.snakeBone:
            pt1,pt2 = self.ConvertPix(bone)
            image = cv2.rectangle(image,pt1,pt2,(0,255,1),-1)
        pt1,pt2 = self.ConvertPix(self.fruit)
        image = cv2.rectangle(image,pt1,pt2,(0,0,255),-1)
        return image
        
        
    def ConvertPix(self,location):
        normalized = [location[0]/(self.width+1),1-(location[1]/(self.height+1))]
        pt1 = [round((normalized[0]-(0.5/(self.width+1)))*resolution[0]),round((normalized[1]+(0.5/(self.height+1)))*resolution[1])]
        pt2 = [round((normalized[0]+(0.5/(self.width+1)))*resolution[0]),round((normalized[1]-(0.5/(self.height+1)))*resolution[1])]
        return pt1,pt2
if __name__ == "__main__":
    env = enviroment(True)
    env.reset()
    while not kb.is_pressed("q"):
        action = env.direction
        if kb.is_pressed("w"):
            action = 1
        elif kb.is_pressed("d"):
            action = 0
        elif kb.is_pressed("s"):
            action = 3
        elif kb.is_pressed("a"):
            action = 2
        _,_,_episode_ended = env.runFrame(action)
        if _episode_ended:
            env.reset()
        
    cv2.destroyAllWindows()
        
        
            
    
        
        
            
        
        
    
    
    


