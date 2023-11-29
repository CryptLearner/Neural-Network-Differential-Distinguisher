import numpy as np
from os import urandom


a_circle = 5
b_circle = 0
c_circle = 1

def WORD_SIZE():
    return(16)
MASK_VAL = 2 ** WORD_SIZE() - 1
# const_simeck32
const_simeck = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffc,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffc, 0xfffd, 0xfffc, 0xfffc,
                0xfffc, 0xfffc, 0xfffd, 0xfffc,
                0xfffc, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffd]

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def enc_one_round_simeck(p, k):
    c1 = p[0] 
    c0 = (rol(p[0], a_circle) & rol(p[0],b_circle)) ^ rol(p[0],c_circle) ^ p[1] ^ k
    return(c0,c1)

def dec_one_round_simeck(c, k):
    c0 = c[0]
    c1 = c[1] 
    p0 = c1
    p1 = (rol(c1, a_circle) & rol(c1,b_circle)) ^ rol(c1,c_circle) ^ c0 ^ k
    return(p0,p1)

def decrypt_simeck(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x,y = dec_one_round_simeck((x,y), k)
    return(x, y)

def expand_key_simeck(k, t):
    ks = [0 for i in range(t)]
    ks_tmp = [0,0,0,0]
    ks_tmp[0] = k[3]
    ks_tmp[1] = k[2]
    ks_tmp[2] = k[1]
    ks_tmp[3] = k[0]
    ks[0] = ks_tmp[0]
    for i in range(1, t):
        ks[i] = ks_tmp[1]
        tmp = (rol(ks_tmp[1], a_circle) & rol(ks_tmp[1], b_circle)) ^ rol(ks_tmp[1], c_circle) ^ ks[i-1] ^ const_simeck[i-1]
        ks_tmp[1] = ks_tmp[2]
        ks_tmp[2] = ks_tmp[3]
        ks_tmp[3] = tmp
    return(ks)

def encrypt_simeck(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round_simeck((x,y), k)
    return(x, y)

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

def make_train_data(n, nr, diff=(0,0x0040)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key_simeck(keys, nr);  
  ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);





####IDDS 
def make_train_data_IDDS(n, nr, diff=(0,0x0040)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key_simeck(keys, nr);  
  ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);


##IDFT 
def make_train_data_IDFT(n, nr, diff=(0,0x0040)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key_simeck(keys, nr);  
  ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks);

  delta_ctdata0l = ctdata0l ^ ctdata1l
  delta_ctdata0r = ctdata0r ^ ctdata1r
 
  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l
    
  delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r

  X = convert_to_binary([delta_ctdata0l,delta_ctdata0r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_ctdata0r,delta_secondLast_ctdata0r]); 
  return(X,Y);


##PDID 
def make_train_data_PDID(n, nr, diff=(0,0x0040,0x200,0x0880)):  
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1; 
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1); 
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16); 
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1]; 
  plain2l = plain0l ^ diff[2]; plain2r = plain0r ^ diff[3]; # 

  num_rand_samples = np.sum(Y==0); 
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16); 

  ks = expand_key_simeck(keys, nr);  
  ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt_simeck((plain2l, plain2r), ks);

  X = convert_to_binary([ctdata0l, ctdata0r ,ctdata1l, ctdata1r, ctdata2l, ctdata2r]); 
  return(X,Y);    


##PDID+IDFT 
def make_train_data_PDID_IDFT(n, nr, diff=(0,0x0040,0x200,0x0880)):  
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

  ks = expand_key_simeck(keys, nr);  
  ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt_simeck((plain2l, plain2r), ks);


  delta_ctdata01l = ctdata0l ^ ctdata1l
  delta_ctdata01r = ctdata0r ^ ctdata1r
  delta_ctdata02l = ctdata0l ^ ctdata2l
  delta_ctdata02r = ctdata0r ^ ctdata2r

  secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l  
  secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l   
  secondLast_ctdata2r = rol(ctdata2r, a_circle) & rol(ctdata2r, b_circle) ^ rol(ctdata2r, c_circle) ^ ctdata2l   


  delta_secondLast_ctdata01r =  secondLast_ctdata0r ^ secondLast_ctdata1r;  
  delta_secondLast_ctdata02r =  secondLast_ctdata0r ^ secondLast_ctdata2r;  

  X = convert_to_binary([delta_ctdata01l , delta_ctdata01r , delta_ctdata02l , delta_ctdata02r , ctdata0l , ctdata0r ,  delta_ctdata01r , delta_secondLast_ctdata01r , delta_ctdata02r , delta_secondLast_ctdata02r]);  
  return(X,Y);   
