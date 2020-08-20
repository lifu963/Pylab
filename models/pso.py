import numpy as np
from tqdm import tqdm


class PSO_BaseModel(object):
    def __init__(self,dim,size,max_iter,x_max,max_vel,func,early_stop,tol,answer,answer_tol):
        self.dim = dim
        self.size = size
        self.max_iter = max_iter
        self.x_max = x_max
        self.max_vel = max_vel
        self.func = func
        
        self.early_stop = early_stop
        self.tol = tol
        self.answer = answer
        self.answer_tol = answer_tol
        
        self.X = np.random.uniform(low=-self.x_max,high=self.x_max,size=(self.size,self.dim))
        self.V = np.random.uniform(low=-self.max_vel,high=self.max_vel,size=(self.size,self.dim))
        self.Y = self.cal_y()
        
        self.pbest_x = self.X.copy()
        self.pbest_y = self.Y.copy()
        self.gbest_x = np.zeros((1,self.dim))
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        
        self.recode_mode = False
        self.recode_value = {'X':[],'V':[],'Y':[]}
        
    def cal_y(self):
        self.Y = self.func(self.X).reshape(-1,1)
        return self.Y
    
    def update_V(self,iter_num):
        pass
    
    def update_X(self):
        self.X = self.X + self.V
    
    def recoder(self):
        if not self.recode_mode:
            return
        self.recode_value['X'].append(self.X)
        self.recode_value['V'].append(self.V)
        self.recode_value['Y'].append(self.Y)
    
    def update_gbest(self):
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(),:].copy()
            self.gbest_y = self.Y.min()
    
    def update_pbest(self):
        self.pbest_x = np.where(self.pbest_y>self.Y,self.X,self.pbest_x)
        self.pbest_y = np.where(self.pbest_y>self.Y,self.Y,self.pbest_y)
    
    def run(self,max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in tqdm(range(self.max_iter),ascii=True):
            self.update_V(iter_num)
            self.recoder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            
            self.gbest_y_hist.append(self.gbest_y)
            
            if self.answer is not None:
                assert self.answer_tol is not None
                if np.abs(self.gbest_y_hist[-1]-self.answer)<self.answer_tol:
                    break
            
            if self.early_stop is not None:
                assert self.tol is not None
                if self.early_stop<=iter_num:
                    if np.abs(self.gbest_y_hist[-self.early_stop] - self.gbest_y_hist[-1])<self.tol:
                        break
                   
        return self  




class IPSO(PSO_BaseModel):
    def __init__(self,func,dim=2,size=50,max_iter=1000,x_max=2,max_vel=0.8,w_min=0.3,w_max=1.3,c1_s=2.4,c1_e=1.4,c2_s=2.3,c2_e=1.3,early_stop=1000,tol=1e-3,answer=None,answer_tol=None):
        super(IPSO,self).__init__(dim,size,max_iter,x_max,max_vel,func,early_stop,tol,answer,answer_tol)
        self.w_min = w_min
        self.w_max = w_max
        self.c1_s = c1_s
        self.c1_e = c1_e
        self.c2_s = c2_s
        self.c2_e = c2_e
    
    def update_V(self,iter_num):
        w = self.w_max - iter_num/self.max_iter*(self.w_max-self.w_min)
        c1 = self.c1_s-(self.c1_s-self.c1_e)*iter_num/self.max_iter
        c2 = self.c2_s-(self.c2_s-self.c2_e)*iter_num/self.max_iter
        
        r1 = np.random.rand(self.size,self.dim)
        r2 = np.random.rand(self.size,self.dim)
        self.V = w*self.V+c1*r1*(self.pbest_x-self.X)+c2*r2*(self.gbest_x-self.X)
        self.V = np.clip(self.V,-self.max_vel,self.max_vel)




class PSO(PSO_BaseModel):
    def __init__(self,func,dim=2,size=50,max_iter=1000,x_max=2,max_vel=0.8,w=0.7,c1=1.6,c2=1.6,early_stop=1000,tol=1e-3,answer=None,answer_tol=None):
        super(PSO,self).__init__(dim,size,max_iter,x_max,max_vel,func,early_stop,tol,answer,answer_tol)
        self.w = w
        self.c1 = c1
        self.c2 = c2
    
    def update_V(self,iter_num):
        r1 = np.random.rand(self.size,self.dim)
        r2 = np.random.rand(self.size,self.dim)
        self.V = self.w*self.V+self.c1*r1*(self.pbest_x-self.X)+self.c2*r2*(self.gbest_x-self.X)
        self.V = np.clip(self.V,-self.max_vel,self.max_vel)



class PSO_X(PSO_BaseModel):
    def __init__(self,func,dim=2,size=50,max_iter=1000,x_max=2,max_vel=0.8,c1=2.05,c2=2.05,early_stop=1000,tol=1e-3,answer=None,answer_tol=None):
        super(PSO_X,self).__init__(dim,size,max_iter,x_max,max_vel,func,early_stop,tol,answer,answer_tol)
        self.c1 = c1
        self.c2 = c2
        self.C = c1+c2
        assert self.C>4
        self.gamma = 2/np.abs(2-self.C-np.sqrt(self.C**2-4*self.C))
    
    def update_V(self,iter_num):
        r1 = np.random.rand(self.size,self.dim)
        r2 = np.random.rand(self.size,self.dim)
        
        self.V = self.gamma*(self.V+self.c1*r1*(self.pbest_x-self.X)+self.c2*r2*(self.gbest_x-self.X))
        self.V = np.clip(self.V,-self.max_vel,self.max_vel)

