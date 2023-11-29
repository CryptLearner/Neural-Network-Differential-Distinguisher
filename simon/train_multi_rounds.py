
#import train_nets_idds_up1lun as tn
#import train_nets_idds_up2lun as tn
import train_nets_type_test as tn_test
import train_nets_type_original as tn_ori
import train_nets_type_idds as tn_idds
import train_nets_type_up1lun as tn_up1


tn_ori.train_simon_distinguisher(30,num_rounds=8,depth=1, Diff=(0x0,0x1,0x100,0x400),BS=5000);   

tn_ori.train_simon_distinguisher(30,num_rounds=9,depth=1, Diff=(0x0,0x1,0x100,0x400),BS=5000);   

tn_ori.train_simon_distinguisher(30,num_rounds=10,depth=1, Diff=(0x0,0x1,0x100,0x400),BS=5000); 

tn_ori.train_simon_distinguisher(30,num_rounds=11,depth=1, Diff=(0x0,0x1,0x100,0x400),BS=5000);  
