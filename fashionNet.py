import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

class PoseLayers(chainer.Chain):
    def __init__(self):
        super(PoseLayers,self).__init__()
        with self.init_scope():
            self.fc6_pose=L.Linear(None,1024)
            self.fc7_pose=L.Linear(None,1024)
            self._vis=[L.Linear(None,2) for i in range(6)]
            self._landmark=L.Linear(None,12)

    def __call__(self, pose):
        pose=self.fc6_pose(pose)
        pose=F.relu(pose)
        visible=[F.softmax(vis(pose)).data for vis in self._vis]
        landmark=self._landmark(pose)
#         visible=np.array(visible).transpose(1,0,2)
        return visible,landmark


class LocalLayers(chainer.Chain):
    def __init__(self):
        super(LocalLayers,self).__init__()
        with self.init_scope():
            self.fc6_local=L.Linear(None,1024)

    def gated_landmark(self,landmark, visible):
        pred_landmark=F.reshape(landmark,(-1,6,2))
#         print(visible)
        visible=np.array(visible,dtype=np.float32)
        
        visible=F.reshape(visible,(-1,6,2))

        gates=F.argmax(visible,axis=2)
        gates=F.reshape(gates,(-1,6,1))
        
        gates=gates.data
        pred_landmark=pred_landmark.data
        
        gates=np.multiply(gates,pred_landmark)
#         gates=[np.multiply(gate,pl) for gate,pl in zip(gates,pred_landmark)]
        gates=np.array(gates,dtype=np.float32)
#         print('gates shape:\t',gates.shape)
#         gates=F.reshape(gates,(-1,6,2))

        return gates

    def get_rois(self,landmark):
        pred_landmark=F.reshape(landmark,(-1,6,2))
        x=np.ones((pred_landmark.shape[0],6,2),dtype=np.float32)/20
        pred_landmark=F.concat((pred_landmark,x),axis=2)
        pred_landmark=F.reshape(pred_landmark,(-1,6,4))
        return pred_landmark

    def __call__(self,conv4,landmark,visible):
        gated=self.gated_landmark(landmark,visible)
        rois=self.get_rois(gated)
        
        new_rois=[]
        for i,roi in enumerate(rois):
            indices=np.zeros((rois.shape[1],1),dtype=np.float32)+(i)
            new_rois.append(F.concat((indices,roi),axis=1))
            
        rois=F.concat(new_rois,axis=0)     
#         print('conv4 shape:\t',conv4.shape)
#         print('rois shape:\t',rois.shape)
        
        #there may be something unexpected in roi_indices
        #conv4.shape[0]==3 can infer the rois[][0]'s index bound ?? 
        roi=F.roi_pooling_2d(conv4,rois,2,2,1)
        roi=F.reshape(roi,(-1,6,roi.shape[1],roi.shape[2],roi.shape[3]))
#         print('roi shape:\t',roi.shape)
        
        h=self.fc6_local(roi)
        h=F.relu(h)
        return h

class GlobalBranch(chainer.Chain):
    def __init__(self):
        super(GlobalBranch, self).__init__()
        with self.init_scope():
            self.conv5_1=L.Convolution2D(None, out_channels=512, ksize=(3, 3), pad=(1,1))
            self.bn5_1=L.BatchNormalization(512)
            
            self.conv5_2=L.Convolution2D(None, out_channels=512, ksize=(3, 3), pad=(1,1))
            self.bn5_2=L.BatchNormalization(512)
            
            self.conv5_3=L.Convolution2D(None, out_channels=512, ksize=(3, 3), pad=(1,1))
            self.bn5_3=L.BatchNormalization(512)
            
            self.fc6_global=L.Linear(None,4096)
            self.bn6=L.BatchNormalization(4096)
            
    def __call__(self,pool4):
#         MaxPooling2D((2,2), strides=(2,2))(input_red)
#         print('pool4 shape:\t',pool4.shape)
        h=F.max_pooling_2d(pool4, ksize=(2,2), stride=(2,2), pad=0)
        
#         red = ZeroPadding2D((1,1))(red)
#         red = Convolution2D(512, (3, 3), activation='relu')(red)
        h=self.conv5_1(h)
        h=self.bn5_1(h)
        h=F.relu(h)
        
        h=self.conv5_2(h)
        h=self.bn5_2(h)
        h=F.relu(h)
        
        h=self.conv5_3(h)
        h=self.bn5_3(h)
        h=F.relu(h)
        
        h=F.max_pooling_2d(h, ksize=(2,2), stride=(2,2), pad=0)
        h=self.fc6_global(h)
        h=self.bn6(h)
        h=F.relu(h)
        
        return h
        

class FashionNet(chainer.Chain):
    def __init__(self,alpha=5,beta=1):
        super(FashionNet,self).__init__()
        from chainer.links import VGG16Layers
        self.vgg_extractor=VGG16Layers()
        self.alpha=alpha
        self.beta=beta
        with self.init_scope():
            self.pose=PoseLayers()
            self.local=LocalLayers()
            self._global=GlobalBranch()
            self.cat=L.Linear(None,50)

    def setTrain_1():
        self.alpha=5
        self.beta=1
        
    def setTrain_2():
        self.alpha=1
        self.beta=5
    
    def __call__(self, imgs, gt_mb_visibles,gt_mb_landmarks,gt_mb_labels):
        vgg_ex=self.vgg_extractor.extract(imgs,layers=['pool5','pool4','conv4_3'])
#         pool5 shape: (3, 512, 7, 7) dtype: float32
#         pool4 shape: (3, 512, 14, 14) dtype: float32
#         conv4_3 shape: (3, 512, 28, 28) dtype: float32
        pose=vgg_ex['pool5']
        visible,landmark=self.pose(pose)
        
        conv4=vgg_ex['conv4_3']
        local=self.local(conv4,landmark,visible)
        
        pool4=vgg_ex['pool4']
        global_out=self._global(pool4)
        
        fc7_fusion=F.concat((local,global_out),axis=1)
        
        cat=self.cat(fc7_fusion)    
        
        if gt_mb_visibles.shape[1]==6:
            gt_mb_visibles=gt_mb_visibles.transpose(1,0)
        
        visible=np.array(visible,dtype=np.float32)
        visible=F.reshape(visible,(6,-1,2))
        
        print('vis shape:\t',visible.shape,gt_mb_visibles.shape)
        v_loss=[F.softmax_cross_entropy(vis,gt) for vis,gt in zip(visible,gt_mb_visibles)]   
        
        l_loss=F.mean_squared_error(landmark,gt_mb_landmarks)
        
        gt_mb_labels=np.array(gt_mb_labels)
        gt_mb_labels=gt_mb_labels.astype(np.int32)
        print('gt_mb_labels shape:\t',(gt_mb_labels.shape))
        c_loss=F.softmax_cross_entropy(cat,gt_mb_labels)
        
        loss=(np.array(v_loss).sum()+l_loss)*self.alpha+c_loss*self.beta
        
        accuracy=F.accuracy(cat,gt_mb_labels)

        from chainer import reporter
        reporter.report({'loss',loss},self)

        reporter.report({'accuracy',accuracy},self)
        
        
class FashionNet_Easy_train(chainer.Chain):
    def __init__(self,alpha=5,beta=1):
        super(FashionNet_Easy_train,self).__init__()
        self.alpha=alpha
        self.beta=beta
        with self.init_scope():
            self.pose=PoseLayers()
            self.local=LocalLayers()
            self._global=GlobalBranch()
#             self.atts=L.Linear(None,1000)
            self.cat=L.Linear(None,50)
    def setTrain_1():
        self.alpha=5
        self.beta=1
        
    def setTrain_2():
        self.alpha=1
        self.beta=5

    def __call__(self, pose,conv4,pool4, gt_mb_visibles,gt_mb_landmarks,gt_mb_labels):
#         , gt_mb_visibles,gt_mb_landmarks,gt_mb_labels
        
        #visible : softmax_cross_entropy(x, t)
        #cat : softmax_cross_entropy(x, t)
        
        #landmark: mean_squared_error(x0, x1) 
        
        visible,landmark=self.pose(pose)
#         print('visible shape:\t',visible.shape)
        
        local=self.local(conv4,landmark,visible)
        
        global_out=self._global(pool4)
        
        print('local shape:\t',local.shape)
        print('global shape:\t',global_out.shape)
        
        fc7_fusion=F.concat((local,global_out),axis=1)
        
#         atts=self.atts(fc7_fusion)
        cat=self.cat(fc7_fusion)    
        
        if gt_mb_visibles.shape[1]==6:
            gt_mb_visibles=gt_mb_visibles.transpose(1,0)
        
        visible=np.array(visible,dtype=np.float32)
        visible=F.reshape(visible,(6,-1,2))
        
        print('vis shape:\t',visible.shape,gt_mb_visibles.shape)
        v_loss=[F.softmax_cross_entropy(vis,gt) for vis,gt in zip(visible,gt_mb_visibles)]   
        
        l_loss=F.mean_squared_error(landmark,gt_mb_landmarks)
        
        gt_mb_labels=np.array(gt_mb_labels)
        gt_mb_labels=gt_mb_labels.astype(np.int32)
        print('gt_mb_labels shape:\t',(gt_mb_labels.shape))
        c_loss=F.softmax_cross_entropy(cat,gt_mb_labels)
        
        loss=(np.array(v_loss).sum()+l_loss)*self.alpha+c_loss*self.beta

        
def test_loss_FashionNet_Easy_train():
    from DeepfashionDatasets import DeepFashionLandmarkDatasets
    dataset=DeepFashionLandmarkDatasets(category_file='category_img145.txt', landmark_file='landmarks145.txt',root='./fashion_data/Img')
    model=FashionNet_Easy_train()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    conv4=np.random.randn(3, 512, 28, 28).astype(np.float32)
    pool4=np.random.randn(3, 512, 14, 14).astype(np.float32)
    
    category=np.random.randn(3).astype(np.int32)
    model(pose,conv4,pool4, dataset._visible[:3],dataset._landmark[:3],category)
    
        

def test_FashionNet_Easy_train():
    model=FashionNet_Easy_train()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    conv4=np.random.randn(3, 512, 28, 28).astype(np.float32)
    pool4=np.random.randn(3, 512, 14, 14).astype(np.float32)
    model(pose,conv4,pool4)
    
        
def test():
    from DeepfashionDatasets import DeepFashionLandmarkDatasets
    model=FashionNet()
    dataset=DeepFashionLandmarkDatasets(category_file='category_img145.txt', landmark_file='landmarks145.txt',root='./fashion_data/Img')
    category=np.random.randn(3).astype(np.int32)
    imgs=np.random.randn(3, 3, 224, 224).astype(np.float32)
    model(imgs,dataset._visible[:3],dataset._landmark[:3],category)
        
def test_global():
    pool4=np.random.randn(3, 512, 14, 14).astype(np.float32)
    
    model=GlobalBranch()
    res=model(pool4)
    print('res shape:\t',res.shape)
        
def test_LocalLayers():
    model=PoseLayers()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    conv4=np.random.randn(3, 512, 28, 28).astype(np.float32)
    res=model(pose)
    vis,landmark=res
    
    model=LocalLayers()
    res=model(conv4,landmark,vis)
    print('PoseLayers outshape:\t',res.shape)
        
        
def test_get_rois():
    model=PoseLayers()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    res=model(pose)
    vis,landmark=res
    
    model=LocalLayers()
    pred_landmark=model.get_rois(landmark)
    print('pred_landmark shape:\t',pred_landmark.shape)
        
        
def test_gated_landmark():
    model=PoseLayers()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    res=model(pose)
    vis,landmark=res
    
    model=LocalLayers()
    gates=model.gated_landmark(landmark, vis)
    print('gates shape:\t',gates.shape)
        
def test_poseLayer():
    model=PoseLayers()
    pose=np.random.randn(3, 512, 7, 7).astype(np.float32)
    res=model(pose)
    vis,landmark=res
#     print('vis : \t',(vis))
    print('landmark: \t',landmark)
    
        
def test_vggextractor():
    model=FashionNet()
    imgs=np.random.randn(3, 3, 224, 224).astype(np.float32)
    vgg_ex=model.vgg_extractor.extract(imgs,layers=['pool5','pool4','conv4_3'])
    
    #get the pool5, pool4, conv4_3 layers shape and dtype
    print('pool5 shape: {} dtype: {}'.format(vgg_ex['pool5'].shape,vgg_ex['pool5'].dtype))
    print('pool4 shape: {} dtype: {}'.format(vgg_ex['pool4'].shape,vgg_ex['pool4'].dtype))
    print('conv4_3 shape: {} dtype: {}'.format(vgg_ex['conv4_3'].shape,vgg_ex['conv4_3'].dtype))
    

if __name__=='__main__':
    test()


