import os
import sys
import cv2
import time
import numpy as np
from PIL import Image
import glob
import torch, copy
from torchvision.utils import save_image
import torch.nn.functional as F

from ui.deepfashion.ui_util.ui import Ui_Form
from ui.deepfashion.ui_util.mouse_event import GraphicsScene
import util.util as util

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
import torchvision.transforms as transforms

color_list = [QColor(0, 0, 0), QColor(204, 0, 0), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), 
                QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0), QColor(204, 204, 204)]

class Ex(QWidget, Ui_Form):
    def __init__(self, model=None, opt=None):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.opt = opt

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.sketch = None
        self.sketch_m = None
        self.img = None
        self.ref_img = None
        self.noise_ins = None
        self.noise = None
        self.inst_id = -1
        self.ref_inst_id = -1
        self.save_results = {}
        self.ref_inst = None
        self.inst = None

        # init the graphics
        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.sk_scene = GraphicsScene(19, 2)
        self.graphicsView_2.setScene(self.sk_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.tag_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.tag_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.mousePressEvent = self.get_inst_id
 
        self.result_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.result_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # self.ref_scene = GraphicsScene(20, 2)
        self.ref_scene = QGraphicsScene()
        self.graphicsView_5.setScene(self.ref_scene)
        self.graphicsView_5.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_5.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_5.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.graphicsView_5.mousePressEvent = self.get_ref_inst_id

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

        # init others
        self.img_path = 'datasets/demo_deepfashion/img'
        self.mask_path = 'datasets/demo_deepfashion/label'
        self.sketch_path = 'datasets/demo_deepfashion/edgeD'
        self.inst_path = 'datasets/demo_deepfashion/inst'
        self.save_path = os.path.join(opt.results_dir, opt.name, 'demo')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.img_path)
        self.img_name = os.path.basename(fileName)
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName).resize((256,256),Image.ANTIALIAS)
            self.img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            tag_image = image.scaled(self.graphicsView_3.size(), Qt.IgnoreAspectRatio)
            if len(self.tag_scene.items())>0:
                self.tag_scene.removeItem(self.tag_scene.items()[-1])
            self.tag_scene.addPixmap(tag_image)
            if len(self.result_scene.items())>0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image.scaled(self.graphicsView_4.size(), Qt.IgnoreAspectRatio))

            # load mask and sketch
            mask_filename = os.path.join(self.mask_path, os.path.basename(fileName)[:-4]+'.png')
            self.open_mask(mask_filename)
            if self.opt.add_sketch:
                sketch_filename = os.path.join(self.sketch_path, os.path.basename(fileName)[:-4]+'.png')
                self.open_sketch(sketch_filename)
            inst_filename = os.path.join(self.inst_path, os.path.basename(fileName)[:-4]+'.png')
            mat_img = cv2.imread(inst_filename)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.inst = mat_img.copy()

    def open_ref(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.img_path)
        if fileName:
            image = QPixmap(fileName)
            mat_img2 = Image.open(fileName).resize((256,256),Image.ANTIALIAS)
            self.ref_img = mat_img2.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView_5.size(), Qt.IgnoreAspectRatio)
        
            if len(self.ref_scene.items())>0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)

            # load ref mask 
            mask_filename = os.path.join(self.mask_path, os.path.basename(fileName)[:-4]+'.png')
            mat_img = cv2.imread(mask_filename)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.ref_mask = mat_img.copy()
            # load ref inst
            inst_filename = os.path.join(self.inst_path, os.path.basename(fileName)[:-4]+'.png')
            mat_img = cv2.imread(inst_filename)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.ref_inst = mat_img.copy()

    def open_mask(self, fileName=None):
        if fileName is None:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.mask_path)
        if fileName:    
            mat_img = cv2.imread(fileName)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            self.mask = mat_img.copy()
            self.mask_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 256, 256, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(256):
                for j in range(256):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.image = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items())>0:
                self.scene.reset_items() 
            self.scene.addPixmap(self.image)

    def open_sketch(self, fileName=None):
        if fileName is None:
            fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.sketch_path)
        if fileName:    
            mat_img = cv2.imread(fileName)
            mat_img = cv2.resize(mat_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            mat_img[mat_img==255] = 19
            self.sketch = mat_img.copy()
            self.sketch_m = mat_img       
            mat_img = mat_img.copy()
            image = QImage(mat_img, 256, 256, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return    

            for i in range(256):
                for j in range(256):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb()) 
           
            pixmap = QPixmap()
            pixmap.convertFromImage(image)  
            self.sk_image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
            self.sk_scene.reset()   
            if len(self.sk_scene.items())>0:
                self.sk_scene.reset_items() 
            self.sk_scene.addPixmap(self.sk_image)

    def get_inst_id(self, event):
        x = event.pos().x() * 2
        y = event.pos().y() * 2
        if self.inst is not None:
            self.inst_id = self.inst[y, x, 0]
        self.label0.setText('Tag Instance Id: '+str(self.inst_id))
        if self.ref_inst is not None:
            # get the reference inst id
            class_id = self.mask[y, x, 0]
            ref_mask_copy = self.ref_mask.copy()[:,:,0] # [h, w]
            ref_inst_indexes = np.where(ref_mask_copy==class_id)
            if len(ref_inst_indexes[0]) == 0:
                print("reference mask does not contain the semantic object of "+str(class_id))
                self.ref_inst_id = -1
                self.label1.setText('Ref Instance Id: '+str(self.ref_inst_id))
            else:
                new_y = ref_inst_indexes[0][0]
                new_x = ref_inst_indexes[1][0]
                self.ref_inst_id = self.ref_inst[new_y, new_x, 0]
                self.label1.setText('Ref Instance Id: '+str(self.ref_inst_id))


    def bg_mode(self):
        self.scene.mode = 0
        self.sk_scene.mode = 0

    def hair_mode(self):
        self.scene.mode = 1

    def face_mode(self):
        self.scene.mode = 2

    def clothes_mode(self):
        self.scene.mode = 3

    def pants_mode(self):
        self.scene.mode = 5

    def body_mode(self):
        self.scene.mode = 4

    def shoes_mode(self):
        self.scene.mode = 6

    def sketch_mode(self):
        self.sk_scene.mode = 19

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1
            self.sk_scene.size += 1
    
    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1 
            self.sk_scene.size -= 1

    def edit(self):
        # process mask
        for i in range(19):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        # process sketch
        # self.sketch_m = self.sketch_m * 0
        # self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[19], self.sk_scene.size_points[19], 19)
        if self.opt.add_sketch:
            for i in [0, 19]:
                self.sketch_m = self.make_mask(self.sketch_m, self.sk_scene.mask_points[i], self.sk_scene.size_points[i], i)

        # process input for model
        assert type(self.img_name), "img_name should not be None!"
        label_tensor = torch.from_numpy(self.mask_m[None, None, ...][...,0]).float() # [1, 1, H, W]
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        if self.opt.add_sketch:
            sketch_tensor = torch.from_numpy(self.sketch_m[None, None, ...][...,0]).float()
            sketch_tensor[sketch_tensor==19] = 1
        else:
            sketch_tensor = torch.tensor(0)

        instance = np.array(Image.open(os.path.join(self.inst_path, self.img_name[:-4]+'.png')))
        instance_tensor = torch.from_numpy(instance[None, None, ...])
        instance_tensor = F.interpolate(instance_tensor, size=label_tensor.size()[2:], mode='nearest')
        instance_tensor = instance_tensor.long()

        if self.ref_img is not None:
            ref_label_tensor = torch.from_numpy(self.ref_mask[None, None, ...][...,0]).float() # [1, 1, H, W]
            ref_label_tensor[ref_label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
            ref_instance_tensor = torch.from_numpy(self.ref_inst[None, None, ...][...,0])
            ref_instance_tensor = ref_instance_tensor.long()
            
            transform_list = []
            transform_list += [transforms.ToTensor()]
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            img_transform = transforms.Compose(transform_list)
            ref_image_tensor = img_transform(self.ref_img).unsqueeze(0)
            image_tensor = img_transform(self.img).unsqueeze(0)
        else:
            ref_label_tensor = torch.tensor(0)
            ref_instance_tensor = torch.tensor(0)
            ref_image_tensor = torch.tensor(0)
            image_tensor = torch.tensor(0)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'sketch': sketch_tensor,
                      'sketch_mask': torch.tensor(0),
                      'path': self.img_name,
                      'ref_label': ref_label_tensor,
                      'ref_instance': ref_instance_tensor,
                      'ref_image': ref_image_tensor,
                      'ref_inst_id': self.ref_inst_id,
                      'inst_id': self.inst_id,
                      'is_ref_inst': self.clickButtion4.isChecked()
                      }
        
        if self.noise is None or self.clickButtion2.isChecked():
            self.noise = torch.randn([1, instance_tensor.max()+1, 2, self.opt.noise_nc]).cuda()
        if self.noise_ins is None or self.clickButtion2.isChecked():
            self.noise_ins = torch.randn(1, self.opt.z_dim, dtype=torch.float32).cuda()
        if self.noise is not None and self.clickButtion3.isChecked():
            assert self.inst_id != -1, "inst_id should not be -1"
            id_noise = torch.randn([1, 1, 2, self.opt.noise_nc]).cuda()
            self.noise[:, self.inst_id, :, :] = id_noise
        if self.noise is not None and self.clickButtion4.isChecked():
            assert self.inst_id != -1, "inst_id should not be -1"
            id_noise = torch.randn([1, 1, 2, self.opt.noise_nc]).cuda()
            self.noise[:, self.inst_id, :, :] = id_noise

        generated = self.model(input_dict, mode='demo', noise=self.noise, noise_ins=self.noise_ins)
  
        # #save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        result = generated.permute(0, 2, 3, 1)
        result = result.cpu().numpy()
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        save_result = Image.fromarray(copy.deepcopy(result))
        result = Image.fromarray(result)
        # qim = QImage(result.data, result.shape[1], result.shape[0], result.shape[0] * 3, QImage.Format_RGB888)
        # pixmap = QPixmap()
        pixmap = result.toqpixmap()
        if len(self.result_scene.items())>0: 
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(pixmap)

        # post-process for save results
        input_dict['fake_image'] = save_result
        self.inputs = input_dict

        print('Done !')
    
    def post_process(self):
        # fake image
        self.save_results['fake_image'] = self.inputs['fake_image']
        # label
        label_t = self.inputs['label'][0,...].cpu().float()
        label_t = util.Colorize(self.opt.label_nc + 2)(label_t)
        label_np = np.transpose(label_t.numpy(), (1, 2, 0))
        self.save_results['label'] = Image.fromarray(np.uint8(label_np))
        # instance
        instance_t= self.inputs['instance'][0,...].cpu().float()
        instance_t = util.Colorize(self.opt.label_nc + 2)(instance_t)
        instance_np = np.transpose(instance_t.numpy(), (1, 2, 0))
        self.save_results['instance'] = Image.fromarray(np.uint8(instance_np))
        # sketch
        if self.opt.add_sketch:
            sketch_np = self.inputs['sketch'][0,0,...].cpu().float().numpy()
            sketch_np[sketch_np==0] = 255
            sketch_np[sketch_np==1] = 0
            self.save_results['sketch'] = Image.fromarray(np.uint8(sketch_np))
        # name
        basename = os.path.basename(self.inputs['path'])
        self.save_results['name'] = basename


    def make_mask(self, mask, pts, sizes, color):
        if len(pts)>0:
            for idx, pt in enumerate(pts):
                cv2.line(mask,pt['prev'],pt['curr'],(color,color,color),sizes[idx])
        return mask

    def save_img(self):
        self.post_process()

        folderName, _ = QFileDialog.getSaveFileName(self, "Save File", self.save_path)
        curr_save_path = os.path.join(self.save_path, self.save_results['name'][:-4]+'_'+os.path.basename(folderName))

        if not os.path.exists(curr_save_path):
            os.mkdir(curr_save_path)
        # save the different data
        self.save_results['fake_image'].save(os.path.join(curr_save_path, 'fake_image.png'))
        self.save_results['label'].save(os.path.join(curr_save_path, 'label.png'))
        self.save_results['instance'].save(os.path.join(curr_save_path, 'instance.png'))
        if self.opt.add_sketch:
            self.save_results['sketch'].save(os.path.join(curr_save_path, 'sketch.png'))
        
        print('Save !')

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()
        self.sketch_m = self.sketch.copy()
    
        self.scene.reset_items()
        self.scene.reset()
        self.sk_scene.reset_items()
        self.sk_scene.reset()
        if type(self.image):
            self.scene.addPixmap(self.image)
        if type(self.sk_image):
            self.sk_scene.addPixmap(self.sk_image)


def demo_for_deepfashion(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    envpath = '/home/tzt/include/anaconda3/envs/pytorch1.8/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

    # define the model
    model = Pix2PixModel(opt)
    model.eval()

    app = QApplication(sys.argv)
    ex = Ex(model, opt)
    sys.exit(app.exec_())