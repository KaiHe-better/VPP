import torch
from collections import OrderedDict
import time

from torch.nn.parallel import DataParallel
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply







def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('wrong:')
                print('obj', obj)
                print('obj, size', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)

        # print('len(inputs): ', str(len(inputs)))
        # print('self.device_ids[:len(inputs)]', str(self.device_ids[:len(inputs)]))

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if self.gpu0_bsz == 0:
            replicas = self.replicate(self.module, self.device_ids)
        else:
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

        # replicas = self.replicate(self.module, device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]

        # print('replicas:', str(len(replicas)))

        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0]["word"].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        # print('bsz: ', bsz)
        # print('num_dev: ', num_dev)
        # print('gpu0_bsz: ', gpu0_bsz)
        # print('bsz_unit: ', bsz_unit)
        # print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
    
  







def print_execute_time(func):
    if type(func) is tuple:
        opera_time = (func[1] - func[0])
        if opera_time > 60:
            if opera_time / 60 > 60:
                opera_time = round(opera_time / 3600, 3)
                print(f'execute time: {opera_time} hour')
            else:
                opera_time = round(opera_time / 60, 3)
                print(f'execute time: {opera_time} minute')
        else:
            print(f'execute time: {round(opera_time, 3)} s')
    else:
        def wrapper(*args, **kwargs):
            start = time.time()
            func_return = func(*args, **kwargs)
            end = time.time()
            opera_time = (end - start)
            if opera_time > 60:
                if opera_time / 60 > 60:
                    opera_time = round(opera_time / 3600, 3)
                    print(f'execute time: {opera_time} hour')
                else:
                    opera_time = round(opera_time / 60, 3)
                    print(f'execute time: {opera_time} minute')
            else:
                print(f'execute time: {round(opera_time, 3)} s')

            return func_return

        return wrapper

def get_dic_added(dic, key_list, item_list):
    for key, item in zip(key_list, item_list):
        # print(type(item))
        # print(item)
        if type(item) == torch.Tensor and torch.isnan(item):
            continue
        else:
            dic[key].append(item)
    return dic

def conmine_sent(tokenized_sent):
    new_sent = ""
    for i in tokenized_sent:
        new_sent +=i.replace("Ġ", " ").replace("</s>", "</s>   ")

    return str(new_sent)

def data_to_device(temp_list):
    if len(temp_list)==4:
        if torch.cuda.is_available():
            for k in temp_list[0]:
                temp_list[0][k] = temp_list[0][k].cuda()
            for k in temp_list[1]:
                temp_list[1][k] = temp_list[1][k].cuda()
            label=temp_list[2].cuda()
            loss_label=temp_list[3].cuda()

        return temp_list[0], temp_list[1], label, loss_label

    elif len(temp_list)==2:
        if torch.cuda.is_available():
            for k in temp_list[0]:
                temp_list[0][k] = temp_list[0][k].cuda()
            label=temp_list[1].cuda()
        return temp_list[0], label


def log_gradient_and_parameter_updates(model, writer, epoch) -> None:

    for name, param in model.named_parameters():
        if param.data.numel() > 0:
            writer.add_scalars("query_parameter/"+name, {"parameter/" + name: param.data.mean().item()}, epoch)

        if param.grad is not None:
            if param.grad.is_sparse:
                grad_data = param.grad.data._values()
            else:
                grad_data = param.grad.data

            if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                writer.add_scalars("query_gradient/"+name, {"gradient/" + name: grad_data.mean()}, epoch)

def log_support_gradient_and_parameter_updates(named_parameters, grads, writer, epoch) -> None:

    for (name, param), grad in zip(named_parameters.items(), grads):
        if param.data.numel() > 0:
            writer.add_scalars("support_parameter/"+name, {"parameter/" + name: param.data.mean().item()}, epoch)

        if grad is not None:
            if grad.is_sparse:
                grad_data = grad.data._values()
            else:
                grad_data = grad.data

            if torch.prod(torch.tensor(grad_data.shape)).item() > 0:
                writer.add_scalars("support_gradient/"+name, {"gradient/" + name: grad_data.mean()}, epoch)


def get_print_loss(loss_list, loss):
    if len(loss):
        if type(loss) ==list:
            added_loss = sum(loss) / len(loss)
        else:
            added_loss = loss
        loss_list.append(added_loss)
        print_loss = sum(loss_list) / len(loss_list)
    else:
        print_loss = torch.tensor(0)
    return print_loss

