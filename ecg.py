import torch, os, json, random, argparse, pprint, datetime, wandb, transformers, collections
import numpy as np
from ecg_data import ECGDataset, get_dataloader
from trainer_base import TrainerBase
from ecg_model import T5forECG
from pathlib import Path
from tqdm import tqdm
from packaging import version
from nlgeval.pycocoevalcap.bleu.bleu import Bleu
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.meteor.meteor import Meteor
from sentence_transformers import SentenceTransformer, util
from bert_score import score

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

_use_apex = False


    
def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Data
    parser.add_argument("--data_dir", type=str, default='data/')
    parser.add_argument('--prefix', type=str, default=None)
    parser.add_argument("--video_id_mapping_file", type=str, default='ECF/feature/video_id_mapping.npy')
    parser.add_argument("--audio_emb_file", type=str, default='ECF/feature/audio_embedding_6373.npy')
    parser.add_argument("--video_emb_file", type=str, default='ECF/feature/video_embedding_3dcnn_4096.npy')

    # Checkpoint
    parser.add_argument('--output_dir', type=str, default='save/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)

    # Model Config
    parser.add_argument('--backbone', type=str, default='google-t5/t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--input_max_length', type=int, default=512)
    parser.add_argument('--audio_feat_dim', type=int, default=6373)
    parser.add_argument('--video_feat_dim', type=int, default=4096)
    parser.add_argument('--task_type', type=str, default='mecg') # caption

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Inference
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--gen_max_length', type=int, default=40)
    parser.add_argument("--main_metric", type=str, default='BLEU4')


    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def metrics(gold, generated, scorers, simcse):
    """
    Compute metrics.
    """
    refs, hyps, task_scores = {}, {}, []
    for j in range(len(gold)):
        refs[j] = [gold[j]]
        hyps[j] = [generated[j]]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if type(score) == list:
            for m, s in zip(method, score):
                task_scores.append(round(s, 4))
        else:
            task_scores.append(round(score, 4))

    embeddings1 = simcse.encode(gold, convert_to_tensor=True)
    embeddings2 = simcse.encode(generated, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    similarity = round(np.mean(cosine_scores.diag().cpu().numpy()), 4)
    task_scores.append(similarity)
    return task_scores

def generation_evaluate(predicts, answers):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    st_model_path = 'princeton-nlp/sup-simcse-roberta-large'
    # st_model_path = 'sentence-transformers/all-mpnet-base-v2'
    simcse = SentenceTransformer(st_model_path).cuda()
    
    scores = metrics(answers, predicts, scorers, simcse)
    
    P, R, F1 = score(predicts, answers, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
    
    scores.append(round(torch.mean(F1).tolist(),4))
    
    metric_name = ["BLEU1", "BLEU2", "BLEU3", "BLEU4", "METEOR", "ROUGE_L", "CIDEr", "Sem-Sim", "F_BERT"]
    from collections import OrderedDict
    return OrderedDict(zip(metric_name, scores))


class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, dataset=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader, 
            train=train)

        if self.args.task_type != 'mecg':
            self.train_loader_origin, _ = get_dataloader(args, 'train', mode='test', batch_size=args.batch_size, workers=args.num_workers)

        self.wandb_initialized = False

        model_kwargs = {}
        config = self.create_config()
        
        # self.tokenizer = self.create_tokenizer()
        self.tokenizer = dataset.tokenizer
        
        self.model = self.create_model(T5forECG, config, **model_kwargs)
        
        self.model.resize_token_embeddings(self.tokenizer.vocab_size+len(dataset.add_tokens))
        
        for token in dataset.add_tokens:
            if token[0] == '<':
                index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=self.tokenizer.vocab_size, (index, self.tokenizer.vocab_size, token)
                indexes = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token[1:-1]))

                embed = self.model.encoder.embed_tokens.weight.data[indexes[0]]

                for i in indexes[1:]:
                    embed += self.model.encoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                self.model.encoder.embed_tokens.weight.data[index] = embed

        self.model.tokenizer = self.tokenizer

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if self.args.from_scratch:
            self.init_weights()
        
        # GPU Options
        print(f'Model Launching at GPU {args.device}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.device)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.momaxtdel, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            best_valid = 0.
            best_epoch = 0

            if not self.wandb_initialized:
                
                project_name = "T5_ECG"
                if self.args.task_type != 'mecg':
                    project_name = "T5_{}".format(self.args.task_type)

                wandb.init(project=project_name)
                wandb.run.name = self.args.run_name
                wandb.config.update(self.args)
                wandb.watch(self.model)

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

                self.wandb_initialized = True

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        epochs = self.args.epochs

        print('\n * Begin training ...\n')
        for epoch in range(epochs):

            if self.start_epoch is not None:
                epoch += self.start_epoch
            self.model.train()
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=120)

            epoch_results = {
                'loss': 0.,

            }

            for step_i, batch in enumerate(self.train_loader):

                if self.args.fp16 and _use_native_amp:
                    with autocast():
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)
                else:
                    if self.args.distributed:
                        results = self.model.module.train_step(batch)
                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']


                if self.args.fp16 and _use_native_amp:
                    self.scaler.scale(loss).backward()
                elif self.args.fp16 and _use_apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()


                loss = loss.detach()

                # Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(
                            self.optim), self.args.clip_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.clip_grad_norm)

                update = True
                if self.args.gradient_accumulation_steps > 1:
                    update = ((step_i+1) % self.args.gradient_accumulation_steps == 0) or (step_i == len(self.train_loader) - 1)

                
                if update:
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                    for param in self.model.parameters():
                        param.grad = None
                    global_step += 1

                for k, v in results.items():
                    if k in epoch_results:
                        epoch_results[k] += v.item()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    try:
                        lr = self.optim.get_lr()[0]
                    except AttributeError:
                        lr = self.args.lr

                if self.verbose:
                    loss_meter.update(loss.item())
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                    desc_str += f' | Loss {loss_meter.val:4f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)

                    wandb.log({"learning_rate": lr, "step": len(self.train_loader)*epoch+step_i})

            if self.args.distributed:
                dist.barrier()

            if self.verbose:
                pbar.close()


            # Validation
            print('\n * Val ...\n')

            valid_results, valid_gen_results = self.evaluate(self.val_loader)

            if self.verbose:
                test_results, test_gen_results = self.evaluate(self.test_loader)

                valid_score = valid_results[self.args.main_metric]

                if valid_score > best_valid or epoch == 0:
                    best_valid = valid_score
                    best_epoch = epoch
                    self.save("BEST")

                    if self.args.task_type != 'mecg':
                        train_results, train_gen_results = self.evaluate(self.train_loader_origin)
                        with open(os.path.join(self.args.output_dir, 'train_generation.json'), 'w') as fo:
                            json.dump(train_gen_results, fo)
                            
                        with open(os.path.join(self.args.output_dir, 'dev_generation.json'), 'w') as fo:
                            json.dump(valid_gen_results, fo)

                    with open(os.path.join(self.args.output_dir, 'test_generation.json'), 'w') as fo:
                        json.dump(test_gen_results, fo)

                with open(os.path.join(self.args.output_dir, 'test_generation_epoch{}.json'.format(epoch)), 'w') as fo:
                    json.dump(test_gen_results, fo)

                log_str = ''
                log_str += pprint.pformat(valid_results)
                log_str += "\nEpoch %d: Valid %s %0.4f" % (epoch, self.args.main_metric, valid_score)
                log_str += "\nEpoch %d: Best %s %0.4f\n" % (best_epoch, self.args.main_metric, best_valid)
                log_str += pprint.pformat(test_results)
                print(log_str)

                wandb_log_dict = {}
                wandb_log_dict['Train/epoch'] = epoch
                wandb_log_dict['Train/Loss'] = epoch_results['loss'] / len(self.train_loader)
                for score_name, score in valid_results.items():
                    wandb_log_dict[f'Valid/{score_name}'] = score
                wandb_log_dict[f'Valid/best_epoch'] = best_epoch


                for score_name, score in test_results.items():
                    wandb_log_dict[f'Test/{score_name}'] = score
                wandb.log(wandb_log_dict)

            if self.args.distributed:
                dist.barrier()

        if self.verbose:
            self.save("LAST")

        # Test Set
        print('\n * Test ...\n')
        best_path = os.path.join(self.args.output_dir, 'BEST')
        self.load(best_path)

        if self.verbose:
            wandb.save(best_path, base_path=self.args.output_dir)
            print(f'\nUploaded checkpoint {best_epoch}', best_path)

        test_results, test_gen_results = self.evaluate(self.test_loader)

        if self.verbose:
            log_str = 'Best Epoch: {}\n'.format(best_epoch+1)
            log_str += 'Test set results\n'
            log_str += pprint.pformat(test_results)

            print(log_str)

        if self.args.distributed:
            dist.barrier()

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams
            gen_kwargs['max_length'] = self.args.gen_max_length

            # losses = 0
            for i, batch in enumerate(tqdm(loader, ncols=120, desc="Prediction", disable=not self.verbose)):

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['pred'])
                # losses += results['loss']

                if 'target_text' in batch:
                    targets.extend(batch['target_text'])

            results = {
                'predictions': predictions,
                'targets': targets
            }
            # results['losses'] = losses

            if self.args.distributed:
                dist.barrier()

                dist_results = dist_utils.all_gather(results)
                predictions = []
                targets = []
                for result in dist_results:
                    predictions.extend(result['predictions'])
                    targets.extend(result['targets'])
                results = {
                    'predictions': predictions,
                    'targets': targets
                }

            return results

    def evaluate(self, loader, dump_path=None):
        results = self.predict(loader, dump_path)

        if self.verbose:
            predictions = results['predictions']
            print('# predictions:', len(predictions))
            if dump_path is None:
                targets = results['targets']
                eval_results = generation_evaluate(predictions, targets)
                return eval_results, results


if __name__ == "__main__":
    
    print(datetime.datetime.now().strftime('\n%Y-%m-%d-%H-%M\n'))
    
    print('torch: ', torch.__version__)
    print(torch.cuda.is_available())
    print(transformers.__version__)
    
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    args.run_name = 'T5'
    if 'flan' in args.backbone:
        args.run_name = 'Flan_T5'
    if args.task_type != 'mecg': 
        args.run_name += '_{}'.format(args.task_type)
    else:
        args.run_name += '_ECG'

    if 'mm' in args.data_dir:
        args.run_name += '_A'
        args.run_name += '_V'

    if 'caption' in args.data_dir.split('/')[-1]:
        args.run_name += '_C'
    if 'cause_aware_caption' in args.data_dir:
        args.run_name += '_CaC'
    
    args.run_name += '_{}'.format(args.batch_size)
    args.run_name += '_{}_{}'.format(args.epochs, args.lr)
    args.run_name += '_seed{}'.format(args.seed)
    args.run_name += '_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    
    print(args)
    if args.seed:
        print('\nSet seed: {}\n'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_loader, dataset = get_dataloader(args, 'train', mode='train', batch_size=args.batch_size, workers=args.num_workers)
    
    if args.valid_batch_size is not None:
        valid_batch_size = args.valid_batch_size
    else:
        valid_batch_size = args.batch_size
    dev_loader, _ = get_dataloader(args, 'dev', mode='dev', batch_size=valid_batch_size, workers=args.num_workers)
    test_loader, _ = get_dataloader(args, 'test', mode='test', batch_size=valid_batch_size, workers=args.num_workers)
    print('\n# batch: train {} dev {} test {}\n'.format(len(train_loader), len(dev_loader), len(test_loader)))
    
    trainer = Trainer(args, train_loader, dev_loader, test_loader, dataset, train=True)
    trainer.train()