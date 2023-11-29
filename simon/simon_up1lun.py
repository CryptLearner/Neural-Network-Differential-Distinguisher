

import numpy as np
from os import urandom  

def WORD_SIZE():
    return(16);   

a_circle = 1   
b_circle = 8
c_circle = 2

MASK_VAL = 2 ** WORD_SIZE() - 1; 

Z = (1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0); #62比特固定数组

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));   

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));  

def enc_one_round(p, k):   
    c0, c1 = p[0], p[1];
    temp = c0;
    c0 = c1 ^ ( rol(c0, a_circle) & rol(c0, b_circle) ) ^ rol(c0, c_circle) ^ k;  
    c1 = temp;
    return(c0,c1);   

def dec_one_round(c,k):  
    c0, c1 = c[0], c[1];
    temp = c1;   
    c1 = c0 ^ ( rol(temp, a_circle) & rol(temp, b_circle) ) ^ rol(temp, c_circle) ^ k;  
    c0 = temp;
    return(c0, c1);  

def expand_key(k, t):    
    ks = [0 for i in range(t)];   
    ks[0] = k[0];
    ks[1] = k[1];
    ks[2] = k[2];
    ks[3] = k[3];
 
    for i in range(4,t):   
        temp = ror(ks[i-1],3)  
        temp = temp ^ ks[i-3];
        temp = temp ^ ror(temp,1);
        ks[i] = ( (~ks[i-4]) & MASK_VAL) ^ temp ^ Z[i-4] ^ 3;   
    return(ks);   


def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:  
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):   
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():  
  key = (0x0302,0x0b0a,0x1312,0x1b1a);  
  pt = (0x6f72, 0x6e69);  
  ks = expand_key(key, 32) ; 
  ct = encrypt(pt, ks)  ;
  if (ct == (0x3a5d, 0xc612)):
    print("Testvector verified.加密验证成功")
    return(True);
  else:
    print("Testvector not verified.加密验证失败")
    return(False);

def check_deco():  
  key = (0x0302,0x0b0a,0x1312,0x1b1a);  
  ct = (0x3a5d, 0xc612);  
  ks = expand_key(key, 32) ; 
  pt = decrypt(ct, ks)  ;
  if (pt == (0x6f72, 0x6e69)):
    print("Testvector verified.解密验证成功")
    return(True);
  else:
    print("Testvector not verified.解密验证失败")
    return(False);


def convert_to_binary(l):  
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return(X)


def readcsv(datei):  
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator  #生成训练数据
def make_train_data_up1lun(n, nr, diff=(0x200,0x880)): 
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);  
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  
  delta_ctdata0l = ctdata0l ^ ctdata1l
  delta_ctdata0r = ctdata0r ^ ctdata1r
 
  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l   
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l   
    
  delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
    
  thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,a_circle) & rol(secondLast_ctdata0r,b_circle) ^ rol(secondLast_ctdata0r,c_circle)
  thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,a_circle) & rol(secondLast_ctdata1r,b_circle) ^ rol(secondLast_ctdata1r,c_circle)    
        
  delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    
  X = convert_to_binary([delta_ctdata0l,delta_ctdata0r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_ctdata0r,delta_secondLast_ctdata0r]);  #这样的数据是8*16=128比特  只向上推导了一轮

  return(X,Y);






def make_train_data_PDID_IDFT_1(n, nr, diff=(0,0x0040,0x200,0x0880)): 
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);  
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 
  plain2l = plain0l ^ diff[2]; plain2r = plain0r ^ diff[3]; 

  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key(keys, nr);  
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);


  delta_ctdata01l = ctdata0l ^ ctdata1l
  delta_ctdata01r = ctdata0r ^ ctdata1r
  delta_ctdata02l = ctdata0l ^ ctdata2l
  delta_ctdata02r = ctdata0r ^ ctdata2r

  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l   
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l   
  secondLast_ctdata2r = rol(ctdata2r, a_circle) & rol(ctdata2r, b_circle) ^ rol(ctdata2r, c_circle) ^ ctdata2l   


  delta_secondLast_ctdata01r =  secondLast_ctdata0r ^ secondLast_ctdata1r;  
  delta_secondLast_ctdata02r =  secondLast_ctdata0r ^ secondLast_ctdata2r;  

  X = convert_to_binary([delta_ctdata01l , delta_ctdata01r , delta_ctdata02l , delta_ctdata02r , ctdata0l , ctdata0r ,  delta_ctdata01r , delta_secondLast_ctdata01r , delta_ctdata02r , delta_secondLast_ctdata02r]);  #返回去修改convert_to_binary
  return(X,Y);    





def make_train_data_PDID_IDFT_2(n, nr, diff=(0,0x0040,0x200,0x0880)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1); 
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 
  plain2l = plain0l ^ diff[2]; plain2r = plain0r ^ diff[3]; 

  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key(keys, nr); 
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);


  delta_ctdata01l = ctdata0l ^ ctdata1l
  delta_ctdata01r = ctdata0r ^ ctdata1r
  delta_ctdata02l = ctdata0l ^ ctdata2l
  delta_ctdata02r = ctdata0r ^ ctdata2r

  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l   
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l  
  secondLast_ctdata2r = rol(ctdata2r, a_circle) & rol(ctdata2r, b_circle) ^ rol(ctdata2r, c_circle) ^ ctdata2l   

  delta_secondLast_ctdata01r =  secondLast_ctdata0r ^ secondLast_ctdata1r;  
  delta_secondLast_ctdata02r =  secondLast_ctdata0r ^ secondLast_ctdata2r;  

  X = convert_to_binary([delta_ctdata01l , delta_ctdata01r ,ctdata0l, ctdata0r, ctdata1l, ctdata1r, delta_ctdata01r , delta_secondLast_ctdata01r , delta_ctdata02l , delta_ctdata02r ,ctdata0l, ctdata0r, ctdata2l, ctdata2r, delta_ctdata02r , delta_secondLast_ctdata02r ]);  #返回去修改convert_to_binary
  return(X,Y);    


def make_train_data_PDID_IDFT_3(n, nr, diff=(0,0x0040,0x200,0x0880)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);  
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 
  plain2l = plain0l ^ diff[2]; plain2r = plain0r ^ diff[3]; 

  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key(keys, nr); 
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);


  delta_ctdata01l = ctdata0l ^ ctdata1l
  delta_ctdata01r = ctdata0r ^ ctdata1r
  delta_ctdata02l = ctdata0l ^ ctdata2l
  delta_ctdata02r = ctdata0r ^ ctdata2r

  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l   
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l   
  secondLast_ctdata2r = rol(ctdata2r, a_circle) & rol(ctdata2r, b_circle) ^ rol(ctdata2r, c_circle) ^ ctdata2l   

  delta_secondLast_ctdata01r =  secondLast_ctdata0r ^ secondLast_ctdata1r;  
  delta_secondLast_ctdata02r =  secondLast_ctdata0r ^ secondLast_ctdata2r;  

  X = convert_to_binary([delta_ctdata01l , delta_ctdata01r ,ctdata0l, ctdata0r, ctdata1l, ctdata1r, delta_ctdata01r , delta_secondLast_ctdata01r  ]);  


def make_train_data_PDID_IDDS_ORI(n, nr, diff=(0,0x0040,0x200,0x0880)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);  
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 
  plain2l = plain0l ^ diff[2]; plain2r = plain0r ^ diff[3]; 

  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key(keys, nr);  
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);

  X = convert_to_binary([ctdata0l, ctdata0r ,ctdata1l, ctdata1r, ctdata2l, ctdata2r]);  
  return(X,Y);    

def make_train_data_IDDS(n, nr, diff=(0,0x0040)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);  
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 


  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key(keys, nr);  
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);

  X = convert_to_binary([ctdata0l, ctdata0r ,ctdata1l ,ctdata1r]);  
  return(X,Y);    



check_testvector();  #加密已经验证成功
check_deco();

