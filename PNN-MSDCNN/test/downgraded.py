import numpy as np
import scipy.ndimage
import scipy.misc as misc



def downgraded(I_MS,I_PAN,ratio):
           
           I_MS = np.double(I_MS)
           I_PAN = np.double(I_PAN)
           I_PAN = np.squeeze(I_PAN)
           ratio=np.double(ratio)



           I_MS_LP = np.zeros((I_MS.shape[0],int(np.round(I_MS.shape[1]/ratio)+ratio),int(np.round(I_MS.shape[2]/ratio)+ratio)))

           for idim in range(I_MS.shape[0]):
                      imslp_pad = np.pad(I_MS[idim,:,:],int(2*ratio),'symmetric')
                      I_MS_LP[idim,:,:]=misc.imresize(imslp_pad,1/ratio,'bicubic',mode='F')
            
           I_MS_LR = I_MS_LP[:,2:-2,2:-2]
       
           I_PAN_pad=np.pad(I_PAN,int(2*ratio),'symmetric')
           I_PAN_LR=misc.imresize(I_PAN_pad,1/ratio,'bicubic',mode='F')
           I_PAN_LR=I_PAN_LR[2:-2,2:-2]


           I_PAN_LR = np.expand_dims(I_PAN_LR,axis=0)









           return I_MS_LR,I_PAN_LR


           
