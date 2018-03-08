import chainer
import six
import numpy as np

class DeepFashionLandmarkDatasets(chainer.dataset.DatasetMixin):
    """docstring for DeepFashionLandmarkDatasets"""
    def __init__(self, category_file, landmark_file,root='.'):
        super(DeepFashionLandmarkDatasets, self).__init__()
        if isinstance(landmark_file, six.string_types) and isinstance(category_file,six.string_types):
            
            with open(landmark_file) as  l_f:
                landmark=[]
                visible=[]
                for i,line in enumerate(l_f):
                    l_line=line
                    pair2=l_line.strip().split()
                    #landmark shape: (12,)
                    pair2=np.array(pair2[1:],dtype=np.int32)
                    if pair2[0]==2:
                        # 6*2 
                        visible.append([pair2[[1,4,7,10,1,4]]])
                        landmark.append([pair2[[2,3,5,6,8,9,11,12,2,3,5,6]]])  
                    else:
                        visible.append([pair2[[1,4,7,10,13,16]]])
                        landmark.append([pair2[[2,3,5,6,8,9,11,12,14,15,17,18]]])
                
                self._category = chainer.datasets.LabeledImageDataset(category_file, root)
                self._landmark=np.array(landmark,dtype=np.float32)
                self._landmark=np.reshape(self._landmark,(-1,12))
                self._visible=np.array(visible,dtype=np.int32)
                self._visible=np.reshape(self._visible,(-1,6))
                
    def __len__(self):
        return len(self._category)
    
    def get_example(self,i):
        image,label=self._category[i]
        landmark=self._landmark[i]
        visible=self._visible[i]
        
        return image,visible,landmark,label

def test():
    dataset=DeepFashionLandmarkDatasets(category_file='category_img145.txt', landmark_file='landmarks145.txt')
    
    print('landmark shape:\t',dataset._landmark.shape)
    print('visible shape:\t',dataset._visible.shape)
    print('landmark dtype:\t',dataset._landmark.dtype)
    print('visible dtype:\t',dataset._visible.dtype)

if __name__=='__main__':
    test()     
