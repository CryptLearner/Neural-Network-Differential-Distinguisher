
import train_nets_type_idds as tn_idds
import train_nets_type_idft as tn_idft
import train_nets_type_pdid as tn_pdid
import train_nets_type_pdid_idft as tn_pdid_idft



#20230914 PDID(c1,c2,c3） (0x0, 0x40)  (0x8000, 0x1)  (0x0, 0x1) (0x0, 0x4)  10-11轮
print("Start PDID-0914")
tn_pdid.train_simon_distinguisher(20,num_rounds=5,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_simon_distinguisher(20,num_rounds=6,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_simon_distinguisher(20,num_rounds=7,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_simon_distinguisher(20,num_rounds=8,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_simon_distinguisher(20,num_rounds=9,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_simon_distinguisher(20,num_rounds=10,depth=1, Diff=(0x0, 0x40, 0x8000, 0x1),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出