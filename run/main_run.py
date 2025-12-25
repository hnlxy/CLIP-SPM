import torch
import numpy as np
import os
import random
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from utils.utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy, verify_checkpoint_dir, task_confusion, visualize_grad_cam
from torch.optim import lr_scheduler
from video_reader import VideoDataset
from torch.utils.tensorboard import SummaryWriter

# torch version >= 2.3
from torch import autocast, GradScaler


# 1.6 <= torch version < 2.3
# from torch import autocast
# from torch.cuda.amp import GradScaler
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def getWIFN(seed):
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
    return worker_init_fn

class Learner:
    def __init__(self, cfg):
        self.cfg = self.parse_config(cfg)

        self.checkpoint_dir, self.logfile, self.test_checkpoint_path, self.resume_checkpoint_path \
            = get_log_files(cfg)

        print_and_log(self.logfile, "Options: %s\n" % self.cfg)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)
        if cfg.TEST.ONLY_TEST:
            print_and_log(self.logfile, "ONLY_TEST::Checkpoint Path: %s\n" % self.test_checkpoint_path)

        self.train_episodes = cfg.TRAIN.TRAIN_EPISODES
        self.test_episodes = cfg.TEST.TEST_EPISODES

        #self.writer = SummaryWriter()
        mode = 'test' if cfg.TEST.ONLY_TEST else 'train'
        ######################################################################################
        log_dir = './logs/'
        if cfg.INFO == '':
            info = f"{cfg.MODEL.NAME}_{mode}_{cfg.DATA.DATASET}::{cfg.MODEL.BACKBONE}_{cfg.TRAIN.WAY}-{cfg.TRAIN.SHOT}_{cfg.TRAIN.QUERY_PER_CLASS}"
        else:
            info = f"{cfg.INFO}_{mode}_{cfg.DATA.DATASET}::{cfg.MODEL.BACKBONE}_{cfg.TRAIN.WAY}-{cfg.TRAIN.SHOT}_{cfg.TRAIN.QUERY_PER_CLASS}"
        if cfg.DEBUG:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, f'debug_{cfg.MODEL.NAME}'),flush_secs = 30)
        else:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, f"{info}=>{datetime.now().strftime('%Y|%m|%d-%H:%M:%S')}"),flush_secs = 30)
        ######################################################################################
        
        #str_device = 'cuda:0'
        self.str_device = cfg.DEVICE.DEVICE
        self.device = torch.device(self.str_device if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        
        print("Random Seed: ", cfg.MODEL.SEED)
        np.random.seed(cfg.MODEL.SEED)
        random.seed(cfg.MODEL.SEED)
        torch.manual_seed(cfg.MODEL.SEED)
        torch.cuda.manual_seed(cfg.MODEL.SEED)
        torch.cuda.manual_seed_all(cfg.MODEL.SEED)
        torch.backends.cudnn.deterministic = True

        self.model = self.init_model()
        self.vd = VideoDataset(self.cfg)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.cfg.DATA.NUM_WORKERS, shuffle=False, worker_init_fn=getWIFN(cfg.MODEL.SEED), pin_memory=True)

        self.use_amp = self.cfg.USE_AMP

        # torch version >= 2.3
        self.scaler = GradScaler(self.str_device, enabled=self.use_amp)  # USE_AMP：Mixed Precision

        # torch version < 2.3
        # self.scaler = GradScaler(enabled=self.use_amp)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.cfg.SOLVER.OPTIM_METHOD == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr=self.cfg.SOLVER.LR, 
                                              betas=(0.5, 0.999), 
                                              weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
            # self.optimizer = torch.optim.AdamW(self.model.parameters(),
            #                                     lr=self.cfg.SOLVER.LR,
            #                                     betas=(0.5, 0.999),
            #                                     weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        elif self.cfg.SOLVER.OPTIM_METHOD == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.cfg.SOLVER.LR, 
                                             momentum=self.cfg.SOLVER.MOMENTUM, 
                                             weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)
        self.test_accuracies = TestAccuracies([self.cfg.DATA.DATASET])
        
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.cfg.SOLVER.LR_SCH], gamma=0.1)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.SOLVER.LR_SCH, gamma=0.9)
        
        self.start_iteration = 0
        self.best_acc = 0.0
        if self.cfg.CHECKPOINT.RESUME_FROM_CHECKPOINT or self.cfg.TEST.ONLY_TEST:
            self.load_checkpoint()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def init_model(self):
        if self.cfg.MODEL.NAME == 'trx':
            from models.model_TRX import CNN_TRX as CNN
        elif self.cfg.MODEL.NAME == 'ta2n':
            from models.model_ta2n import CNN
        elif self.cfg.MODEL.NAME == 'strm':
            from models.model_strm import CNN_STRM as CNN
        elif self.cfg.MODEL.NAME == 'molo':
            from models.model_molo import CNN_BiMHM_MoLo as CNN
        elif self.cfg.MODEL.NAME == 'soap':
            from models.model_soap import CNN_SOAP as CNN
        elif self.cfg.MODEL.NAME == 'otam':
            from models.model_otam import CNN_OTAM as CNN
        elif self.cfg.MODEL.NAME == 'clipfsar':
            from models.model_clipfsar import CNN_OTAM_CLIPFSAR as CNN
        elif self.cfg.MODEL.NAME == 'cpm2c':
            from models.model_cpm2c import CLIP_CPMMC_FSAR as CNN
        elif self.cfg.MODEL.NAME == 'sten':
            from models.model_sten import CNN_OTAM_CLIPFSAR as CNN
        elif self.cfg.MODEL.NAME == 'clipspm':
            from models.model_clipspm import CNN as CNN
        model = CNN(self.cfg)
        model = model.to(self.device)
        if self.cfg.DEVICE.NUM_GPUS > 1:
            model.distribute_model()

        print(f'inited model: {self.cfg.MODEL.NAME}\n')
        return model


    """
    Command line parser
    """
    def parse_config(self, cfg):
        

        print('learning rate decay scheduler', cfg.SOLVER.LR_SCH)

        if cfg.CHECKPOINT.CHECKPOINT_DIR == None:
            print("need to specify a checkpoint dir")
            exit(1)

        # if (args.backbone == "resnet50") or (args.backbone == "resnet34"):
        #     args.img_size = 224
        if cfg.MODEL.BACKBONE == "resnet50":
            cfg.trans_linear_in_dim = 2048
        else:
            cfg.trans_linear_in_dim = 512
        
        cfg.trans_linear_out_dim = cfg.MODEL.TRANS_LINEAR_OUT_DIM

        if cfg.DATA.DATASET == "ssv2":
            cfg.traintestlist = os.path.join("splits/ssv2_OTAM")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ssv2_256x256q5_l8")
        if cfg.DATA.DATASET == 'ssv2_cmn':
            cfg.traintestlist = os.path.join("splits/ssv2_CMN")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ssv2_CMN_256x256q5_l8")
        elif cfg.DATA.DATASET == 'hmdb':
            cfg.traintestlist = os.path.join("splits/hmdb_ARN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "hmdb_256x256q5_l8")
        elif cfg.DATA.DATASET == 'ucf':
            cfg.traintestlist = os.path.join("splits/ucf_ARN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "ucf_256x256q5_l8")
        elif cfg.DATA.DATASET == 'kinetics':
            cfg.traintestlist = os.path.join("splits/kinetics_CMN/")
            cfg.path = os.path.join(cfg.DATA.DATA_DIR, "k100_256x256q5_l8")

        return cfg

    def run(self):
        if self.cfg.TEST.ONLY_TEST:
            print('Conduct Testing:')
            accuracy_dict = self.test()
            self.test_accuracies.print(self.logfile, accuracy_dict)
            print('Evaluation Done with', self.test_episodes, ' iteration')
        else:
            print('Conduct Training:')
            best_accuracies = self.best_acc
            train_accuracies = []
            losses = []
            total_iterations = self.train_episodes

            iteration = self.start_iteration
            for task_dict in tqdm(self.video_loader, total=total_iterations, desc="Training", ncols=120):
                if iteration >= total_iterations:
                    break
                iteration += 1
                #print('iteration', iteration)
                torch.set_grad_enabled(True)
                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.cfg.TRAIN.TASKS_PER_BATCH == 0) or (iteration == (total_iterations - 1)):
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                self.scheduler.step()


                self.writer.add_scalar('loss/Train_loss[it]', task_loss, iteration + 1)
                self.writer.add_scalar('acc/Train_acc[it]', task_accuracy, iteration + 1)

                if (iteration + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                    # print training stats
                    print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'.format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(), torch.Tensor(train_accuracies).mean().item()))
                    self.writer.add_scalar('loss/Train_loss[mean]', torch.Tensor(losses).mean().item(), (iteration + 1) // self.cfg.TRAIN.PRINT_FREQ)
                    self.writer.add_scalar('acc/Train_acc[mean]', torch.Tensor(train_accuracies).mean().item(), (iteration + 1) // self.cfg.TRAIN.PRINT_FREQ)
                    #self.writer.add_scalar('para/mo_alpha1', self.model.mo_alpha1, (iteration + 1) // self.cfg.TRAIN.PRINT_FREQ)
                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.cfg.CHECKPOINT.SAVE_FREQ == 0) and (iteration + 1) != total_iterations:
                    # self.save_checkpoint(iteration + 1)
                    # self.save_checkpoint(iteration + 1, 'last', accuracy_dict[self.cfg.DATA.DATASET]["accuracy"])
                    self.save_checkpoint(iteration + 1, f'iter_{iteration + 1}', 0)

                if ((iteration + 1) % self.cfg.TRAIN.VAL_FREQ == 0) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.test()

                    if accuracy_dict[self.cfg.DATA.DATASET]["accuracy"] > best_accuracies:
                        best_accuracies = accuracy_dict[self.cfg.DATA.DATASET]["accuracy"]
                        print('Save best checkpoint in {} iter'.format(iteration + 1))
                        self.save_checkpoint(iteration + 1, 'best', best_accuracies)

                    self.writer.add_scalar('loss/Test_loss', accuracy_dict[self.cfg.DATA.DATASET]["loss"], (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.writer.add_scalar('acc/Test_acc', accuracy_dict[self.cfg.DATA.DATASET]["accuracy"], (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.writer.add_scalar('acc/Best_acc', best_accuracies, (iteration + 1) // self.cfg.TRAIN.VAL_FREQ)
                    self.test_accuracies.print(self.logfile, accuracy_dict)

        self.logfile.close()

    def train_task(self, task_dict):
        input = self.prepare_task(task_dict)

        with autocast(device_type=self.str_device, dtype=torch.bfloat16, enabled=self.use_amp):
            model_dict = self.model(input)
            task_loss, task_acc = self._loss_and_acc(model_dict, input['target_labels'], input['real_target_labels'], input['batch_class_list'], input['real_support_labels'])
        # task_loss.backward(retain_graph=False)
        self.scaler.scale(task_loss).backward(retain_graph=False)

        return task_loss, task_acc

    def test(self):
        self.model.eval()
        with torch.no_grad():
        # with torch.enable_grad():
                self.video_loader.dataset.train = False
                accuracy_dict ={}
                accuracies = []
                losses = []
                iteration = 0
                item = self.cfg.DATA.DATASET
                for task_dict in self.video_loader:
                    if iteration >= self.test_episodes:
                        break
                    iteration += 1

                    input = self.prepare_task(task_dict)
                    # context_images, target_images, context_labels, target_labels, real_target_labels, batch_class_list, real_support_labels = self.prepare_task(task_dict)

                    with autocast(device_type=self.str_device, dtype=torch.bfloat16, enabled=self.use_amp):
                        model_dict = self.model(input)
                        task_loss, task_acc = self._loss_and_acc(model_dict, input['target_labels'], input['real_target_labels'], input['batch_class_list'], input['real_support_labels'], mode='test')

                    losses.append(task_loss.item())
                    accuracies.append(task_acc.item())

                    current_accuracy = np.array(accuracies).mean() * 100.0
                    if self.cfg.TEST.ONLY_TEST:
                        self.writer.add_scalar(f'TEST/{self.cfg.DATA.DATASET}_{self.cfg.TRAIN.SHOT}-shot', current_accuracy, iteration+1)
                    print('current acc:{:0.3f} in iter:{:n}'.format(current_accuracy, iteration+1), end='\r',flush=True)

                accuracy = np.array(accuracies).mean() * 100.0
                loss = np.array(losses).mean()
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
                self.video_loader.dataset.train = True
        self.model.train()
        
        return accuracy_dict


    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_support_labels = task_dict['real_support_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)
        real_target_labels = real_target_labels.to(self.device)
        real_support_labels = real_support_labels.to(self.device)

        return {'context_images':context_images, 
                'target_images':target_images, 
                'context_labels':context_labels, 
                'target_labels':target_labels, 
                'real_target_labels':real_target_labels, 
                'batch_class_list':batch_class_list, 
                'real_support_labels':real_support_labels
                }  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def _loss_and_acc(self, model_dict, target_labels, real_target_labels, batch_class_list, real_support_labels, mode='train'):
        lmd = 0.1
        model_dict = {k: v.to(self.device) for k,v in model_dict.items()}
        target_logits = model_dict.get('logits')

        if self.cfg.MODEL.NAME == 'strm':
            # Target logits after applying query-distance-based similarity metric on patch-level enriched features
            target_logits_post_pat = model_dict['logits_post_pat']

            # Add the logits before computing the accuracy
            target_logits = target_logits + lmd*target_logits_post_pat

            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH
            task_loss_post_pat = self.loss(target_logits_post_pat, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH

            # Joint loss
            task_loss = task_loss + lmd*task_loss_post_pat
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits
                del target_logits_post_pat

        elif self.cfg.MODEL.NAME == 'molo':
            if mode == 'test':
                task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH
                task_accuracy = self.accuracy_fn(target_logits, target_labels)
                del target_logits
            else:
                task_loss = ( self.loss(target_logits, target_labels, self.device)/ self.cfg.TRAIN.TASKS_PER_BATCH \
                      +self.cfg.MODEL.USE_CLASSIFICATION_VALUE * self.loss(model_dict["class_logits"], torch.cat([real_support_labels, real_target_labels], 0).long(), self.device)) /self.cfg.TRAIN.TASKS_PER_BATCH \
                         + self.cfg.MODEL.USE_CONTRASTIVE_COFF * self.loss(model_dict["logits_s2q"], target_labels, self.device) /self.cfg.TRAIN.TASKS_PER_BATCH \
                            + self.cfg.MODEL.USE_CONTRASTIVE_COFF * self.loss(model_dict["logits_q2s"], target_labels, self.device) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                + self.cfg.MODEL.USE_CONTRASTIVE_COFF * self.loss(model_dict["logits_s2q_motion"], target_labels, self.device) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                    + self.cfg.MODEL.USE_CONTRASTIVE_COFF * self.loss(model_dict["logits_q2s_motion"], target_labels, self.device) /self.cfg.TRAIN.TASKS_PER_BATCH \
                                        + self.cfg.MODEL.RECONS_COFF*model_dict["loss_recons"]
                task_accuracy = self.accuracy_fn(target_logits, target_labels)

        elif self.cfg.MODEL.NAME == 'clipfsar':
            task_loss =  (self.loss(target_logits, target_labels, self.device) + self.cfg.MODEL.USE_CLASSIFICATION_VALUE * self.loss(model_dict["class_logits"], torch.cat([real_support_labels, real_target_labels], 0).long(), self.device)) /self.cfg.TRAIN.TASKS_PER_BATCH
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits

        elif self.cfg.MODEL.NAME == 'cpm2c':
            # lambdas = self.cfg.MODEL.LMD
            lambdas = [self.cfg.params['lambdas0'], self.cfg.params['lambdas1'], self.cfg.params['lambdas2'], self.cfg.params['lambdas3']]
            target_logits_total = lambdas[1] * model_dict['logits_local'] + lambdas[2] * model_dict['logits_global']
            task_loss = lambdas[0] * self.loss(model_dict['class_logits'], torch.cat([real_support_labels, real_target_labels], 0).long(), "cuda") / self.cfg.TRAIN.TASKS_PER_BATCH \
                        + lambdas[1] * self.loss(model_dict['logits_local'], target_labels.long(), "cuda") / self.cfg.TRAIN.TASKS_PER_BATCH \
                        + lambdas[2] * self.loss(model_dict['logits_global'], target_labels.long(), "cuda") / self.cfg.TRAIN.TASKS_PER_BATCH              
                        # + lambdas[3] * self.loss(model_dict['class_logits_video'], torch.cat([real_support_labels, real_target_labels], 0).long(), "cuda") / self.cfg.TRAIN.TASKS_PER_BATCH \
            task_accuracy = self.accuracy_fn(target_logits_total, target_labels)
            if mode == 'test':
                del target_logits
            else:
                task_loss += 0.001 * model_dict['target_consist_distance'] #+ 0.001 * model_dict['text_distance']   

        elif self.cfg.MODEL.NAME == 'soap':
            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH + model_dict['t_loss']
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits

        elif self.cfg.MODEL.NAME == 'clipspm':
            task_loss = self.loss(model_dict['logits'], target_labels.long(), "cuda") / self.cfg.TRAIN.TASKS_PER_BATCH + 0.001 * model_dict['dists']
            task_accuracy = self.accuracy_fn(model_dict['logits'], target_labels)
            if mode == 'test':
                del target_logits

        else:
            task_loss = self.loss(target_logits, target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH \
                            #+ self.loss(model_dict['mo_logits'], target_labels, self.device) / self.cfg.TRAIN.TASKS_PER_BATCH 
            task_accuracy = self.accuracy_fn(target_logits, target_labels)
            if mode == 'test':
                del target_logits
        
        return task_loss, task_accuracy

    def save_checkpoint(self, iteration, stat, acc):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'acc': acc}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint_{}.pt'.format(stat)))
        #torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        if self.cfg.TEST.ONLY_TEST:
            checkpoint = torch.load(self.test_checkpoint_path, map_location=self.device)
            print(f'Load checkpoint from {self.test_checkpoint_path} ==> iter: [{checkpoint["iteration"]}]\n')
        else:
            checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
            print(f'Load checkpoint from {self.resume_checkpoint_path} ==> iter: [{checkpoint["iteration"]}]\n')
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'acc' in checkpoint:
            self.best_acc = checkpoint['acc']

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

