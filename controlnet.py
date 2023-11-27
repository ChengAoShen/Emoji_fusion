"""
从U-net构建的controlnet
"""
import copy
from typing import Union

import torch
import torch.nn as nn


class MyControlNet(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.config = unet.config
        self.time_proj = unet.time_proj
        self.time_embedding = unet.time_embedding

        self.conv_in = unet.conv_in

        self.down_blocks = unet.down_blocks
        self.mid_block = unet.mid_block
        self.up_blocks = unet.up_blocks

        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out
        self.dtype = unet.dtype

        # define controlnet
        self.control_conv_in = nn.Conv2d(
            8, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.control_conv_in.weight.data.fill_(0)
        self.control_conv_in.bias.data.fill_(0)

        self.control_down_blocks = copy.deepcopy(unet.down_blocks)
        self.control_mid_block = copy.deepcopy(unet.mid_block)

        control_conv = []
        last_num = 128
        for num_c in self.config.block_out_channels:
            for _ in range(3):
                if num_c != last_num:
                    temp = nn.Conv2d(
                        last_num,
                        num_c,
                        kernel_size=(1, 1),
                    )
                else:
                    temp = nn.Conv2d(
                        num_c,
                        num_c,
                        kernel_size=(1, 1),
                    )
                temp.weight.data.fill_(0)
                temp.bias.data.fill_(0)
                control_conv.append(temp)
                last_num = num_c

        self.control_conv = nn.ModuleList(control_conv)

        control_conv_mid = nn.Conv2d(
            512,
            512,
            kernel_size=(1, 1),
        )
        control_conv_mid.weight.data.fill_(0)
        control_conv_mid.bias.data.fill_(0)
        self.control_conv_mid = control_conv_mid

        for p in self.time_embedding.parameters():
            p.requires_grad_(False)
        for p in self.time_proj.parameters():
            p.requires_grad_(False)
        for p in self.conv_in.parameters():
            p.requires_grad_(False)
        for p in self.down_blocks.parameters():
            p.requires_grad_(False)
        for p in self.mid_block.parameters():
            p.requires_grad_(False)
        for p in self.up_blocks.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        control_condition: torch.FloatTensor,
    ):
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        control_condition = self.control_conv_in(control_condition) + sample

        # 3. down
        # 3.1 normal down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                print("skip_conv")
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 3.2 control down
        control_down_block_res_condition = [
            control_condition,
        ]
        for control_downsample_block in self.control_down_blocks:
            # if hasattr(control_downsample_block, "skip_conv"):
            #     control_condition, control_res_samples, skip_sample = control_downsample_block(
            #         hidden_states=control_condition, temb=emb, skip_sample=skip_sample
            #     )
            # else:
            #     control_condition, control_res_samples = control_downsample_block(hidden_states=control_condition, temb=emb)
            control_condition, control_res_samples = control_downsample_block(
                hidden_states=control_condition, temb=emb
            )

            control_down_block_res_condition += control_res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)
        control_condition = self.mid_block(control_condition, emb)

        # 4.1 zero conv out
        for index, conv in enumerate(self.control_conv):
            # print(control_down_block_res_condition[index].shape)
            # print(conv)
            control_down_block_res_condition[index] = conv(
                control_down_block_res_condition[index]
            )

        control_condition = self.control_conv_mid(control_condition)

        # 4.2 add
        sample += control_condition

        # # --test
        # print("------------------------------------")
        # for i in control_down_block_res_condition:
        #     print(i.shape)

        # print("------------------------------------")

        # for i in down_block_res_samples:
        #     print(i.shape)
        # print("------------------------------------")

        # # ---

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if hasattr(upsample_block, "skip_conv"):
            #     print("skip conv")
            #     sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            # else:
            #     sample = upsample_block(sample, res_samples, emb)

            # sample = upsample_block(sample, res_samples, emb)
            sample = upsample_block(
                sample, res_samples, emb, control_down_block_res_condition
            )
            control_down_block_res_condition = control_down_block_res_condition[:-3]

            # print(f"sample:{sample.shape}")

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        return sample
