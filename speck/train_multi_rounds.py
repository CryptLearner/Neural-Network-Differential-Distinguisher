
import train_nets_type_idds as tn_idds
#import train_nets_type_idft as tn_idft
import train_nets_type_pdid as tn_pdid
#import train_nets_type_pdid_idft as tn_pdid_idft


tn_pdid.train_speck_distinguisher(20,num_rounds=5,depth=1, Diff=(0x2,0x400, 0x4000,0x8080),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_speck_distinguisher(20,num_rounds=6,depth=1, Diff=(0x2,0x400, 0x4000,0x8080),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_speck_distinguisher(20,num_rounds=7,depth=1, Diff=(0x2,0x400, 0x4000,0x8080),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出
tn_pdid.train_speck_distinguisher(20,num_rounds=8,depth=1, Diff=(0x2,0x400, 0x4000,0x8080),BS=5000);   #tn程序已经改为可变差分和可变BS 并且输出





