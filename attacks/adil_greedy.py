import torchattacks

from attacks.utils import *
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from cmaes import SepCMA, CMA
# --------------------------------------------------------------------------------------------- #
# ------------------------------ Adversarial dictionary learning ------------------------------ #
# --------------------------------------------------------------------------------------------- #

class ADIL(Attack):

    """
    The adversarial dictionary learning model introduced in : The data-driven transferable adversarial space
    Args:
        model: The deep learning model to attack
        model_name: The name of the model.
        steps: Determines the length of the checkpoint.
        targeted: Indicates whether a targeted attack is used to train or craft the adversarial example.
        step_size: Initial learning rate for training the adversarial dictionary.
        eps: The initial constraint on the perturbation magnitude.
        loss: The loss function used for the attack, with the default being the difference of logits (logit).
        bbo_methods: Specifies the method for a black-box attack, e.g., CMA, ZOO.
        kappa: Parameter in the logit loss function.
        eps_coding: The perturbation magnitude used during coding, which may differ from the value used during training.
    """

    def __init__(self, model, steps=1e2, targeted=False, norm='linf', batch_size=128, kappa=5,
                 step_size=0.01, eps=10/255, loss='ce', model_name='', bbo_methods='zoo', eps_coding=10/255):
        super().__init__("ADIL", model)
        self.steps = steps
        self.batch_size = batch_size
        self.targeted = targeted
        self.step_size = step_size
        self.eps = eps
        self.loss = loss
        self.norm = norm
        self.kappa = kappa
        self.model_name = model_name
        self.bbo_methods = bbo_methods
        self.eps_coding = eps_coding

    def projection_d(self, var, type_matrix='l2'):
        """
        function to make the var in defined space
        var: input matrix
        type_matrix: specify the projected space
        return: the projected matrix
        """
        if type_matrix == 'l2':
            """ In order to respect l2 bound, D has to lie inside a l2 unit ball """
            return torch.mul(var, 10/torch.linalg.norm(var.view(-1, var.shape[-1]), dim=0).view(1,1,1,-1))

        elif type_matrix == 'linf':
            """ In order to respect l2 bound, D has to lie inside a linf unit ball """
            return torch.clamp(var, min=-1, max=1)

    def f_loss(self, outputs, labels, kappa):
        """
        Function to compute the logit loss.

        Args:
            outputs: The output from the deep learning model.
            labels: The true labels.
            kappa: A bounding parameter.
        """
        one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels]

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        if self.targeted:
            return torch.clamp((i-j), min=-kappa)
        else:
            return torch.clamp((j-i), min=-kappa)

    def f_loss_2(self, outputs, output_ori, labels, kappa):
        # implicite targeted attack, with target the second largest logit
        one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels]

        _, inds = torch.max((1-one_hot_labels*1e6)*output_ori, dim=1)  # get the second largest logit index
        one_hot_second_inds = torch.eye(len(outputs[0]), device=self.device)[inds]
        i = torch.masked_select(outputs, one_hot_second_inds.bool())
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        if self.targeted:
            return torch.clamp((i-j), min=-kappa)
        else:
            return torch.clamp((j-i), min=-kappa)


    def dlr_loss(self, output, labels):
        # scaled difference of logits
        x_sorted, ind_sorted = output.sort(dim=1)
        ind = (ind_sorted[:, -1] == labels).float()
        u = torch.arange(output.shape[0])

        return (output[u, labels] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


    def learn_dictionary_greedy(self, source_models, dataset, val, step_size_v=1.,n_add=50, restart_limit=10,
                                ratio_atoms=3., all_data=False, time_consuming=False):
        """
        Learn the adversarial dictionary in greedy fashion
        """

        model_file = (f'adversarial_directions_greedy_{self.model_name}_full_'
                      f'{n_add}_atom_added_on_dataset_{len(dataset)}.bin')
        if os.path.exists(model_file):
            return

        assert not isinstance(source_models, str), 'source models should be a list'
        dataset.indexed = False
        # Shape parameters
        n_img = len(dataset)
        x, _ = next(iter(dataset))
        nc, nx, ny = x.shape

        # Other parameters
        batch_size = n_img if self.batch_size is None else self.batch_size

        # Data loader
        dataset.indexed = True
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=True)  # , pin_memory=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, pin_memory=False,
                                                 num_workers=0)

        # Function
        criterion = nn.CrossEntropyLoss(reduction='sum')

        # Initialization of the dictionary D and coding vectors v
        if self.norm.lower() == 'l2':
            d = self.projection_d(torch.randn(nc, nx, ny, n_add, device=self.device), type_matrix='l2')
        elif self.norm.lower() == 'linf':
            d = (-1 + 2 * torch.randn(nc, nx, ny, n_add, device=self.device))
        D = None

        v = torch.rand(n_img, n_add, device=self.device) # * self.eps
        non_attacked = torch.ones(n_img, dtype=torch.bool)

        # initialized model
        d = nn.Parameter(d)
        v = nn.Parameter(v)

        # update d and v with different learning rate
        optimiser = torch.optim.AdamW([
            {'params': v, 'lr': step_size_v},
            {'params': d},
        ], lr=self.step_size)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.5)

        # Initialization of intermediate variables
        loss_all = []
        fooling_rate_all = []
        fooling_rate_single = 0
        loss_checkpoint = []
        fooling_rate_all_on_evl_set = []
        fooling_rate_d = []
        d_best_i = torch.clone(d)
        v_best_i = torch.clone(v)
        best_loss_i = 1e6
        cur_checkpoint_length = int(0.22*self.steps)
        reduced_length = int(0.03*self.steps)
        last_lr = None
        cur_lr = self.step_size
        lam_aug = 1
        add_atoms = n_add
        atk_vs = []
        saved_ori_labels = []
        saved_atk_labels = []
        best_fooling_rate = 0
        eps = self.eps
        to_end = False

        iteration = 0
        fr = 0
        restart = 0

        # main loop
        while fr < 1.:

            loss_full = 0
            fooling_sample = 0

            d.requires_grad = True
            v.requires_grad = True
            for index, (x, label) in tqdm(data_loader, position=1, leave=False):
                # Load data
                x, label = x.to(device=self.device), label.to(device=self.device)
                inds = non_attacked[index]
                batch_attack = torch.zeros(inds.int().sum(), dtype=torch.int)

                for source_model in source_models:
                    label = label[inds] # only bengin data is used for training
                    if all_data:
                        # all data is used for training
                        label = source_model(x[inds]).argmax(dim=-1)

                    # compute loss with model
                    optimiser.zero_grad()
                    if D is None:
                        Dv = torch.tanh(torch.tensordot(v[index[inds]], d, dims=([1], [3])))
                    else:
                        Dv = torch.tanh(torch.tensordot(v[index[inds], D.shape[-1]:], d, dims=([1], [3]))
                              + torch.tensordot(v[index[inds], :D.shape[-1]], D, dims=([1], [3])))

                    output = source_model(torch.clamp(x[inds] + eps*Dv, min=0, max=1))
                    fooling_sample += torch.sum(output.argmax(dim=-1) != label)
                    if self.loss == 'ce':
                        loss = -1 * criterion(output, label)
                        loss = torch.clamp_min(loss, min=-9)
                    elif self.loss == 'logits':
                        loss = self.f_loss(output, label, 100).sum()
                    elif self.loss == 'dlr':
                        loss = self.dlr_loss(output, label).sum()
                    elif self.loss == 'logits_1':
                        loss = self.f_loss_2(output, source_model(x[inds]), label, 1).sum()

                    # update d and v with projected gradient descent
                    loss.backward()
                    optimiser.step()

                    with (torch.no_grad()):
                        loss_full += loss
                        d.data.copy_(self.projection_d(d.data, type_matrix='linf'))
                        # print(cur_eps[-1], loss.item())
                fooling_sample += (batch_attack == len(source_models)).sum()

            # save the intermediate variable
            loss_all.append(loss_full.item() / len(non_attacked) / len(source_models))
            loss_checkpoint.append(loss_all[-1])
            fooling_rate_all.append(fooling_sample.item() / n_img)
            # print('eps, fooling rate on training set,loss', [iteration, fooling_rate_all[-1],
            #                                                 round(loss_all[-1], 4)])# ,

            if loss_checkpoint[-1] < best_loss_i and fooling_rate_single < fooling_rate_all[-1]:
                # update the current best solutions when a smaller loss and larger fooling rate is found
                d_best_i = torch.clone(d)
                v_best_i = torch.clone(v)
                best_loss_i = loss_checkpoint[-1]
                fooling_rate_single = fooling_rate_all[-1]

            if len(loss_checkpoint) == cur_checkpoint_length:
                # check the searching state and restart a new searching with updated learning rate
                temp_cur_lr = cur_lr
                if len(loss_checkpoint) < len(loss_all):
                    loss_checkpoint = np.array(loss_checkpoint)
                    if (loss_checkpoint[1:]-loss_checkpoint[:-1]<0).astype(int).sum() < 0.75*cur_checkpoint_length:
                        scheduler.step()
                        cur_lr = cur_lr*0.5
                        lam_aug *= 1.2
                    elif loss_checkpoint[0] == best_loss_i and cur_lr == last_lr:
                        scheduler.step()
                        cur_lr = cur_lr*0.5
                        lam_aug *= 1.2

                # reset the new start point the current best solution
                d.data.copy_(d_best_i.data)
                v.data.copy_(v_best_i.data)
                last_lr = temp_cur_lr
                cur_checkpoint_length = max(cur_checkpoint_length-reduced_length, 2*reduced_length)
                loss_checkpoint = [best_loss_i]

            if iteration % 100 == 0:
                print('eps, fooling rate on training set,loss', [iteration, fooling_rate_all[-1],
                                                                 round(loss_all[-1], 4), cur_lr])

            if cur_lr <= 1e-4:
                # stop current sub-dictionary learning
                if D is not None and ((fooling_rate_single)*n_img < d.shape[-1]):
                    # When it corresponds to the final stage of sub-dictionary learning.
                    if restart == restart_limit and best_fooling_rate == 0:
                        # none of examples can be attacked and stop the training, saved d without this sub-dictionary
                        print('stopped')
                        return
                    elif restart == restart_limit and to_end:
                        # Successfully finish the final stage of sub-dictionary learning
                        # and set the 'd' and 'v' to the best solution found during trials.
                        d.data = d_best_test.data
                        v.data = v_best_test.data
                        restart += 1
                    else:

                        if eps < self.eps_coding:
                            # increment eps to attack more examples
                            eps += 1 / 255
                        else:
                            # Adjust the size of the final sub-dictionary according
                            # to the number of remaining examples to attack
                            if int((fooling_rate_single) * n_img) > best_fooling_rate:
                                # update the best solution when a new trial finishes
                                best_fooling_rate = int((fooling_rate_single) * n_img)
                                d_best_test = torch.clone(d)
                                v_best_test = torch.clone(v)

                            if len(non_attacked) <= d.shape[-1] or len(non_attacked) < 30:
                                # Only a small number of examples remain, so enlarge 'd' to match the number of examples.
                                add_atoms = len(non_attacked)
                                print('directs for last data')
                                to_end = True
                            elif len(non_attacked) >= 30:
                                # for some deep model hard to attack, e.g., robust model
                                loc_start = 0
                                if restart == 0:
                                    # First, conduct trials.
                                    print('1st reduce atom number')
                                    add_atoms = d.shape[-1]  # min(d.shape[-1], 10)
                                if restart == restart_limit:
                                    # No trial is successful, so reset the size of 'd'.
                                    to_end = True
                                    print('reduce again atom number')
                                    extra_atoms = int(len(non_attacked)//10//1.5) if len(non_attacked) > 1000 else len(non_attacked)
                                    add_atoms = min(max(int(best_fooling_rate * 1.5), int(extra_atoms // ratio_atoms)), extra_atoms)
                                    best_fooling_rate = 0
                                    restart = 0

                            fooling_rate_single = 0

                            del d
                            # restart training with new initialization
                            d = (-1 + 2 * torch.randn(nc, nx, ny, add_atoms, device=self.device))

                            data_loader = torch.utils.data.DataLoader(dataset, batch_size=add_atoms, shuffle=True)

                            for index, x, label in data_loader:
                                x, labels = x.to(device=self.device), label.to(device=self.device)
                                if all_data:
                                    labels = source_models[0](x).argmax(dim=-1)
                                # print(labels.detach().cpu().numpy())

                                if x.shape[0] == len(non_attacked):
                                    # When a limited number of examples remain to be attacked,
                                    # apply AutoAttack to search for the final directions
                                    del v
                                    d.data.copy_((torchattacks.AutoAttack(source_models[0], eps=eps)(x, labels) - x).view(add_atoms, -1).transpose(1, 0).reshape(nc, nx, ny, -1))
                                    d.data.copy_(torch.arctanh(torch.clamp(d.data/eps, min=-1+1e-6, max=1-1e-6)))
                                    non_zero_ds = d.view(-1, d.shape[-1]).norm(dim=0) > 0

                                    if not all(non_zero_ds):
                                        d = d[:, :, :, non_zero_ds]

                                    non_zero_index = index[non_zero_ds.detach().cpu().numpy()]
                                    v = torch.zeros(len(non_attacked), d.shape[-1] + D.shape[-1], device=self.device)
                                    v[non_zero_index, D.shape[-1]:] = torch.diag(d.view(-1, d.shape[-1]).abs().max(dim=0).values)

                                else:
                                    # AutoAttack is used to search for the final directions
                                    # only if it is particularly difficult to find them,
                                    # otherwise, we use PGD.
                                    if restart < 5 or to_end or fr<0.5:
                                        d_temp = (torchattacks.PGD(source_models[0], eps=eps)(x[:add_atoms], labels[:add_atoms]) - x[:add_atoms]).view(add_atoms, -1).transpose(1, 0).reshape(nc, nx, ny, -1)
                                    else:
                                        d_temp = (torchattacks.AutoAttack(source_models[0], eps=eps)(x[:add_atoms], labels[:add_atoms]) - x[:add_atoms]).view(add_atoms, -1).transpose(1, 0).reshape(nc, nx, ny, -1)

                                    d_temp = torch.arctanh(torch.clamp(d_temp.data / eps, min=-1 + 1e-6, max=1 - 1e-6))
                                    non_zero_ds = d_temp.view(-1, add_atoms).norm(dim=0) > 0
                                    non_zero_index = index[non_zero_ds.detach().cpu().numpy()]

                                    if all(non_zero_ds) and loc_start == 0:
                                        d.data.copy_(d_temp)
                                        del v
                                        v = torch.zeros(len(non_attacked), d.shape[-1] + D.shape[-1], device=self.device)
                                        v[non_zero_index, D.shape[-1]:] = torch.diag(d.view(-1, d.shape[-1]).abs().max(dim=0).values)

                                    elif non_zero_ds.sum() < add_atoms and any(non_zero_ds):
                                        if loc_start == 0:
                                            del v
                                            v = torch.zeros(len(non_attacked), d.shape[-1] + D.shape[-1], device=self.device)
                                        add_atoms_i = non_zero_ds.sum()
                                        loc_end = min(loc_start + add_atoms_i, add_atoms)
                                        non_zero_index_i = np.arange(add_atoms)[non_zero_ds.detach().cpu().numpy()]
                                        d.data[:,:,:, loc_start:loc_end].copy_(d_temp[:, :, :, non_zero_index_i[:loc_end-loc_start]])
                                        v.data[non_zero_index[:loc_end-loc_start], D.shape[-1]+loc_start:D.shape[-1]+loc_end] = torch.diag(d.data[:,:,:,loc_start:loc_end].view(-1, loc_end-loc_start).abs().max(dim=0).values)
                                        loc_start = loc_end
                                        if loc_start < add_atoms:
                                            Dv = torch.tanh(torch.tensordot(v[index, D.shape[-1]:], d, dims=([1], [3])) +
                                                            torch.tensordot(v[index, :D.shape[-1]], D, dims=([1], [3])))
                                            fooling_rate_single += (source_models[0](torch.clamp(x + eps * Dv, min=0, max=1)).argmax(dim=-1) != labels).int().detach().cpu().sum() / n_img
                                            continue

                                    elif non_zero_ds.sum() == 0:
                                        continue

                                    del d_temp
                                # initilize d with the computed directions
                                d.data.copy_(d / d.view(-1, d.shape[-1]).abs().max(dim=0).values.view(1, 1, 1, -1))
                                Dv = torch.tanh(torch.tensordot(v[index, D.shape[-1]:], d, dims=([1], [3])) +
                                                torch.tensordot(v[index, :D.shape[-1]], D, dims=([1], [3])))
                                fooling_rate_single += (source_models[0](torch.clamp(x + eps * Dv, min=0, max=1)).argmax(dim=-1) != labels).int().detach().cpu().sum() / n_img
                                break

                            restart += 1

                        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

                        d = nn.Parameter(d)
                        v = nn.Parameter(v)
                        # print(restart, d.shape[-1], int((fooling_rate_single) * n_img), len(non_attacked))
                        optimiser = torch.optim.AdamW([
                            {'params': v, 'lr': step_size_v},
                            {'params': d},
                        ], lr=self.step_size)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.5)
                        best_loss_i = 1e6
                        cur_checkpoint_length = int(0.22 * self.steps)
                        last_lr = None
                        cur_lr = self.step_size
                        loss_checkpoint = []
                        d_best_i = torch.clone(d)
                        v_best_i = torch.clone(v)

                if fooling_rate_single*n_img >= d.shape[-1] or restart > restart_limit:
                    # if not the final stage of training then concatenate the 'd' to 'D'
                    D = torch.clone(d.data) if D == None else torch.cat([D, torch.clone(d.data)], dim=-1)
                    loss_all = []
                    ori_labels = torch.zeros(len(dataset), dtype=torch.long)
                    atk_labels = torch.zeros(len(dataset), dtype=torch.long)

                    # Find the attacked examples and exclude them from the training data
                    for index, x, label in tqdm(data_loader, position=1, leave=False):
                        x, labels = x.to(device=self.device), label.to(device=self.device)
                        inds = non_attacked[index]
                        batch_attack = torch.zeros(inds.int().sum(), dtype=torch.int)
                        for source_model in source_models:
                            label = labels[inds]
                            if all_data:
                                label = source_model(x[inds]).argmax(dim=-1)
                            ori_labels[index] = label.detach().cpu()
                            Dv = torch.tanh(torch.tensordot(v[index[inds]], D, dims=([1], [3])))
                            atk_imgs = (x[inds]+eps*Dv).clamp(min=0, max=1)
                            output_atk = source_model(atk_imgs).argmax(dim=-1)
                            atk_labels[index] = output_atk.detach().cpu()
                            not_attacked_batch = (output_atk == label).int().detach().cpu()

                            # If time permits, search exhaustively for the solution of 'v' for examples that have not yet been attacked
                            if time_consuming and  D.shape[-1] > 1 and not_attacked_batch.sum() > 0:
                                index_images = torch.arange(x.shape[0])[inds]
                                v.data[index[index_images[not_attacked_batch>0]]], atk_imgs, _, _ \
                                    = self.coding(x[index_images[not_attacked_batch>0]],
                                                  label[index_images[not_attacked_batch>0]], D, step_size=1, loss_coding='logits',
                                                  target_model=source_model, early_stop=True, initial=True, steps=2000)
                                                  # v_init=v.data[index[index_images[not_attacked_batch>0]]])
                                output_atk = source_model(atk_imgs).argmax(dim=-1)
                                atk_labels[index[not_attacked_batch > 0]] = output_atk.detach().cpu()
                                not_attacked_batch[not_attacked_batch>0] = (output_atk == label[index_images[not_attacked_batch>0]]).int().detach().cpu()
                            batch_attack += not_attacked_batch  # (output_atk == labels).int().detach().cpu()
                        non_attacked[index[inds]] = batch_attack > 0

                    print('eps, fooling rate on training set,loss',
                          [iteration, eps, 1-non_attacked.int().sum()/n_img, D.shape[-1]])

                    fooling_rate_d.append((eps, D.shape[-1], 1-non_attacked.int().sum()/n_img))
                    fr = fooling_rate_d[-1][-1]
                    num_rest_data = len(non_attacked)

                    added_atoms = d.shape[-1]
                    print('print added atoms', added_atoms)
                    atk_vs = v[~non_attacked] if len(atk_vs) == 0 else torch.cat([torch.cat([atk_vs, torch.zeros(atk_vs.shape[0], added_atoms, device=self.device)], dim=1), v[~non_attacked]], dim=0)
                    saved_ori_labels = ori_labels[~non_attacked] if len(saved_ori_labels) == 0 else torch.cat([saved_ori_labels, ori_labels[~non_attacked]], dim=0)
                    saved_atk_labels = atk_labels[~non_attacked] if len(saved_atk_labels) == 0 else torch.cat([saved_atk_labels, atk_labels[~non_attacked]], dim=0)

                    if any(non_attacked):
                        # There are still examples to be attacked; extract these examples to further train 'd'
                        dataset.indices = dataset.indices[non_attacked]

                        v_old = torch.clone(v[non_attacked])
                        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        non_attacked = non_attacked[non_attacked]
                        # add_atoms = 1 if len(non_attacked) < 100 else add_atoms
                        del d, optimiser
                        d = (-1 + 2 * torch.randn(nc, nx, ny, n_add, device=self.device))
                        # d = self.projection_d(torch.randn(nc, nx, ny, n_add, device=self.device), type_matrix='l2')

                        v = torch.rand(len(non_attacked), n_add + D.shape[-1], device=self.device)
                        v[:, :-n_add] = v_old
                        d = nn.Parameter(d)
                        v = nn.Parameter(v)
                        optimiser = torch.optim.AdamW([
                            {'params': v, 'lr': step_size_v},
                            {'params': d},
                        ], lr=self.step_size)
                        # optimiser = torch.optim.AdamW([d, v], lr=self.step_size)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.5)
                        best_loss_i = 1e6
                        best_fooling_rate = 0
                        cur_checkpoint_length = int(0.22 * self.steps)
                        # reduced_length = int(0.03 * self.steps)
                        last_lr = None
                        cur_lr = self.step_size
                        loss_checkpoint = []
                        fooling_rate_single = 0
                        best_fooling_rate = 0

                        lam_aug = 1.0
                        d_best_i = torch.clone(d)
                        v_best_i = torch.clone(v)

                    if time_consuming:
                        # Evaluate the performance on the validation data.
                        d.requires_grad = False
                        v.requires_grad = False
                        fool_on_valset = 0
                        coding_steps = 2000  # min(1000*(D.shape[-1]//n_add), 10000)
                        num_val = 0
                        for x, label in tqdm(val_loader, position=2, leave=False):
                            ind = self.model(x.to(self.device)).argmax(dim=-1)==label.to(self.device)
                            num_val += ind.sum().item()
                            _, fool_sample, _ = self.coding(x[ind], label[ind], D, step_size=1, early_stop=True, steps=coding_steps, loss_coding='logits')
                            with torch.no_grad():
                                fool_on_valset += fool_sample
                        print('D size, fooling rate on eval_sets', [D.shape[-1], fool_on_valset / num_val])
                        fooling_rate_all_on_evl_set.append(fool_on_valset / num_val)
                    torch.save([D, atk_vs, saved_ori_labels, saved_atk_labels, fooling_rate_d, fooling_rate_all_on_evl_set], model_file)
                    if to_end: # or (D.shape[-1]>=50 and n_add==10): # or D.shape[-1]>=300:
                        print(restart, added_atoms)
                        return
                    restart = 0

            iteration += 1

        print('eps, fooling rate on training set, fooling rate on validation set:', fooling_rate_all_on_evl_set)

    def coding(self, images, labels, d, step_size=0.05, target_model=None, initial=False, early_stop=False,
               v_init=None, loss_coding='dlr', steps=2000):
        """
        Function to search for the solution of 'v'
        Args:
            images: the input to attack
            labels: true label of images
            d: the learned adversarial dictionary
            target_model: deep model to attack
            initial: if returned solution is used for an initial value of other attacks
            early_stop: true if the computing stopped when an adversarial example found
            v_init: an initial value of 'v'
            loss_coding: loss function used for computing adversarial perturbation
            steps: number of iterations
        """
        n_img = len(labels)
        images, labels = images.to(device=self.device), labels.to(device=self.device)
        _, nc, nx, ny = images.shape

        if v_init == None:
            v = torch.randn(n_img, d.shape[-1], device=self.device)
        else:
            v = v_init.data

        v = nn.Parameter(v)
        mask = torch.arange(n_img)
        adv_images = torch.zeros_like(images)

        # Here, we consider attacking an example to fool its true label
        label = self.model(images[mask, :, :, :]).argmax(dim=-1)

        optimise = torch.optim.AdamW([v], lr=step_size)  # leaning rate set to 0.05 for linf and 0.1 for l2
        criterion = nn.CrossEntropyLoss(reduction='sum')
        scheduler = torch.optim.lr_scheduler.StepLR(optimise, step_size=1, gamma=0.5)
        loss_track = []
        self.beta = 0.1
        num_iter = 1000
        fr_wrt_iteration = np.zeros(num_iter+1)

        if early_stop:
            # a prior check to find the example attacked with initial 'v'
            with torch.no_grad():
                Dv = torch.tensordot(v[mask, :], d, dims=([1], [3]))
                if target_model == None:
                    label_changed = self.model(
                        (images[mask, :, :, :] + self.eps_coding * torch.tanh(Dv)).clamp(min=0, max=1)).argmax(dim=-1) != label
                else:
                    label_changed = target_model(
                        (images[mask, :, :, :] + self.eps_coding * torch.tanh(Dv)).clamp(min=0, max=1)).argmax(dim=-1) != label
                early_stopped = mask[label_changed.detach().cpu()]
                adv_images[early_stopped] = clamp_image(
                    images[early_stopped] + self.eps_coding * torch.tanh(Dv[label_changed]))
                mask = mask[~label_changed.detach().cpu()]
                # print(iteration, early_stopped.detach().cpu().numpy(), round(1-len(mask)/n_img, 2))

        cur_checkpoint_length = int(0.22*steps)
        best_loss = 1e6
        best_v = torch.clone(v)
        cur_step_size = step_size
        fr_wrt_iteration[0] = (n_img-len(mask))

        v.requires_grad = True
        for iteration in range(num_iter):
            # Load data
            # compute loss with model
            if mask.shape[0] == 0:
                fr_wrt_iteration[iteration+1:] = n_img
                break
            label = self.model(images[mask, :, :, :]).argmax(dim=-1)
            optimise.zero_grad()
            Dv = torch.tensordot(v[mask, :], d, dims=([1], [3]))

            if target_model == None:
                output = self.model((images[mask, :, :, :] + self.eps_coding * torch.tanh(Dv)).clamp(min=0, max=1))
            else:
                output = target_model((images[mask, :, :, :] + self.eps_coding * torch.tanh(Dv)).clamp(min=0, max=1))

            if loss_coding == 'ce':
                loss_f = -1*criterion(output, label)
            elif loss_coding == 'logits':
                loss_f = self.f_loss(output, label, 1).sum()
            elif loss_coding == 'logits_1':
                loss_f = self.f_loss_2(output, self.model(images[mask, :, :, :]), labels,1).sum()
            elif loss_coding == 'dlr':
                loss_f = self.dlr_loss(output, label).sum()

            loss = loss_f

            loss.backward()
            optimise.step()
            # print(iteration, loss.item())

            with torch.no_grad():
                loss_track.append(loss.item()/len(mask))

            if loss_track[-1] < best_loss:
                best_loss = loss_track[-1]
                best_v = torch.clone(v.data)

            if len(loss_track) == cur_checkpoint_length:
                loss_track = np.array(loss_track)
                if len(loss_track) < iteration:
                    if (loss_track[1:]-loss_track[:-1] < 0).astype(int).sum() < 0.75*cur_checkpoint_length:
                        scheduler.step()
                        cur_step_size = cur_step_size/2
                    elif best_loss == loss_track[0]:
                        scheduler.step()
                        cur_step_size = cur_step_size/2
                loss_track = [best_loss]
                cur_checkpoint_length = max(int(cur_checkpoint_length-0.01*steps), int(0.1*steps))
                v.data[mask] = best_v.data[mask]
                #
                print(1-len(mask)/n_img+(self.dlr_loss(output, label)<0).sum().item()/n_img, loss.item(), cur_step_size)

            if early_stop:
                # stop and save the found adversarial examples
                with torch.no_grad():
                    label_changed = self.dlr_loss(output, label) < 0
                    if any(label_changed):
                        early_stopped = mask[label_changed.detach().cpu()]
                        adv_images[early_stopped] = clamp_image(
                            images[early_stopped] + self.eps_coding * torch.tanh(Dv[label_changed]))
                        mask = mask[~label_changed.detach().cpu()]
                        best_loss = (loss-self.f_loss(output[label_changed.detach().cpu()],
                                                      label[label_changed.detach().cpu()], 1).sum()).item()
                        loss_track = [best_loss]
                        v.data[mask] = best_v.data[mask]
                        optimise = torch.optim.AdamW([v], lr=step_size)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimise, step_size=1, gamma=0.5)
                        v.requires_grad = True
                        cur_step_size = step_size
                        cur_checkpoint_length = int(0.22 * steps)

                        print(iteration, early_stopped.detach().cpu().numpy(),
                              label[label_changed.detach().cpu()].detach().cpu().numpy(),
                              round(1-len(mask)/n_img, 2), d.shape[-1])
                    fr_wrt_iteration[iteration+1] = (n_img-len(mask))

            if cur_step_size < 0.2:
                break

        if len(mask) > 0:
            print(np.array(np.unique(self.model(images[mask, :, :, :]).argmax(dim=-1).detach().cpu().numpy(), return_counts=True)))
        Dv = torch.tensordot(v[mask, :], d, dims=([1], [3]))
        adv_images[mask] = (images[mask, :, :, :] + self.eps * torch.tanh(Dv)).clamp(min=0, max=1)

        if initial:
            return v, adv_images.detach(), (n_img-len(mask)), fr_wrt_iteration
        return adv_images.detach(), (n_img-len(mask)), fr_wrt_iteration

    def forward(self, images, labels, data_name='', step_size=0.05, source_model_name=None, source_models=None,
                black_box=False, eps_train=0, n_add=10):
        model_file = 'adversarial_directions_l2_vgg_full_10_atom_added_on_dataset_1000.bin'
        assert os.path.exists(model_file)
        d = torch.load(model_file)[0].data
        ''' attack where the coding vectors are optimized '''
        if black_box:
            if self.bbo_methods == 'cma':
                adv_img, fooling_sample, queries = self.coding_blackbox_sepcma(images, labels, d)
        else:
            # step_size = 50.
            adv_img, fooling_sample, fr_vs_iter = self.coding(images, labels, d, step_size=step_size,
                                                              early_stop=True, loss_coding='logits')
        return adv_img, fooling_sample, fr_vs_iter