"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import numpy as np
import pathlib
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.models import model as _model  # 关键：导入模型类型枚举


#ygx
# 导入Piper机械臂自定义的输入输出转换类
# from openpi.policies.dual_piper_policy import DualPiperInputs, DualPiperOutputs
#ygx

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )


        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )






# #####################################################################
# ############################# pi05_piper ############################

# @dataclasses.dataclass(frozen=True)
# class _CombinedPiperTransform(_transforms.DataTransformFn):
#     """
#     Agibot 的统一转换类：包含重打包、状态/动作处理、delta 动作转换、图像处理、prompt 注入等。
#     该类为顶层可 picklable 的 Transform，便于多进程/多机环境使用。
#     """
#     action_dim: int = 32  # 动作维度（根据模型配置设置）
#     default_prompt: str | None = ""  # 默认指令（当数据中无任务描述时使用）

#     gripper_position: str | None = ""

#     def _parse_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         将输入图像统一为 HWC（uint8）格式：
#         - 若是浮点类型，归一化到 [0, 255] 并转 uint8
#         - 若是 CHW，将其转为 HWC
#         """
#         image = np.asarray(image)
#         if np.issubdtype(image.dtype, np.floating):
#             image = (255 * image).astype(np.uint8)
#         if image.shape[0] == 3:  # 认为是 CHW
#             image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
#         return image

#     def __call__(self, data: dict) -> dict:
#         """
#         处理单条样本字典，执行字段重命名、状态/动作对齐、delta 动作变换、图像解析与 prompt 选择。
#         返回包含图像、图像掩码、delta 动作、状态、语言提示的标准化字典。
#         对raw_item进行处理
#         """

#         # ---------------------------------------------------------------------
#         # Step 1: 字段重命名（上游多源数据对齐为统一键名）
#         # 说明：
#         # - 将 observation 与 action 相关字段映射到内部统一命名
#         # - 若原键不存在则跳过，避免 KeyError
#         # ---------------------------------------------------------------------
#         processed_data = data.copy()
#         rename_map = {
#             "observation.state": "state",
#             "action": "actions",
#             "observation.images.hand_left": "left_image",
#             "observation.images.hand_right": "right_image",
#             "observation.images.top_head": "head_image",
#         }
#         for old_key, new_key in rename_map.items():
#             if old_key in processed_data:
#                 processed_data[new_key] = processed_data.pop(old_key)

#         # ---------------------------------------------------------------------
#         # Step 2: 状态与动作对齐（维度填充）
#         # 说明：
#         # - 使用 pad_to_dim 将 state 与 actions 补齐到 self.action_dim
#         # - 要求 processed_data 含有 "state" 与 "actions" 键
#         # ---------------------------------------------------------------------
#         state = processed_data["state"]
#         state = _transforms.pad_to_dim(state, self.action_dim)  # 按末维补齐/截断
#         actions = processed_data["actions"]
#         actions = _transforms.pad_to_dim(actions, self.action_dim)  # 按末维补齐/截断

#         # ---------------------------------------------------------------------
#         # Step 3: Delta 动作变换（按掩码选择差分维度）
#         # 说明：
#         # - 维度策略：
#         #   * 前 14 维使用 delta（True）
#         #   * 紧随 2 维非 delta（False）
#         #   * 当 action_dim==32 时，后 16 维保持非 delta（False）
#         # - 其它维度配置暂不支持，将回退为全 False 并告警
#         # ---------------------------------------------------------------------
#         if self.action_dim == 32:

#             if self.gripper_position == '15-16':
#                 delta_action_mask = np.array([True] * 14 + [False] * 2 + [False] * 16)
#             elif self.gripper_position == '8-16':
#                 delta_action_mask = np.array([True] * 7 + [False] + [True] * 7 + [False] + [False] * 16)

#         elif self.action_dim == 16:

#             if self.gripper_position == '15-16':
#                 delta_action_mask = np.arraR7SSuizP6ZHay([True] * 14 + [False] * 2)
#             elif self.gripper_position == '8-16':
#                 delta_action_mask = np.array([True] * 7 + [False] + [True] * 7 + [False])

#         else:
#             print(f"Not implemented {self.action_dim=}")
#             delta_action_mask = np.zeros(self.action_dim, dtype=bool)  # 回退策略：全非 delta

#         delta_transform = _transforms.DeltaActions(delta_action_mask)
#         transformed_data = delta_transform({"state": state, "actions": actions})
#         actions_delta = transformed_data["actions"]

#         # ---------------------------------------------------------------------
#         # Step 4: 图像解析与默认补全
#         # 说明：
#         # - 解析 head/left/right 三路图像，转为模型期望格式
#         # - 缺失时填充 224x224x3 的全零图，保证键齐全
#         # - 输出键命名为 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
#         # ---------------------------------------------------------------------
#         images = {}
#         for key, img_key in [
#             ("base_0_rgb", "head_image"),
#             ("left_wrist_0_rgb", "left_image"),
#             ("right_wrist_0_rgb", "right_image"),
#         ]:
#             if img_key in processed_data:
#                 images[key] = self._parse_image(processed_data[img_key])  # 解析/标准化图像
#             else:
#                 images[key] = np.zeros((224, 224, 3), dtype=np.uint8)     # 默认占位图

#         # ---------------------------------------------------------------------
#         # Step 6: 汇总输出（键齐全、类型明确）
#         # 说明：
#         # - image_mask: 对存在的三路图像全部置 True
#         # - actions: 返回 delta 后的动作
#         # - state: 返回填充后的状态
#         # - prompt: 返回语言化的任务提示
#         # ---------------------------------------------------------------------
#         return {
#             "image": images,
#             "image_mask": {k: np.True_ for k in images},
#             "actions": actions_delta,
#             "state": state,
#             "prompt": processed_data['lang'],
#             "task_index": processed_data["task_index"],  # Add:按任务归一化新增 采样时data需要返回提供当前任务信息 hf_dataset中保存的结果
#             "episode_index": processed_data["episode_index"],  # Add:区分不同条轨迹
#             "frame_index": processed_data["frame_index"],  # Add: 区分在轨迹的帧数
#             "info": processed_data["info"]
#         }

# ############################# pi05_piper ############################
# #####################################################################





# #####################################################################
# ############################# pi05_piper ############################
# # ===================================================================
# #                      LeRobotPiperDataConfig 类
# # ===================================================================

# @dataclasses.dataclass(frozen=True)
# class LeRobotPiperDataConfig(DataConfigFactory):
#     """
#     Agibot 数据集的专用配置工厂：
#     - 直接使用统一的 _CombinedPiperTransform 作为输入变换组
#     - 仍可叠加标准的模型级变换（例如分词）
#     返回data_loader中的配置
#     """
#     repo_id: str = tyro.MISSING
#     assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
#     base_config: tyro.conf.Suppress[DataConfig | None] = None
#     # 默认 prompt，可通过配置注入
#     default_prompt: str | None = None

#     # 新增：
#     gripper_position: str = "8-16"
#     vlm_prediction_type: str | None = None  # 进度表示方法：cont-进度百分比; disc-子任务阶段
#     prompt_version_for_subtask: str | None = None  # 预测子任务时使用的prompt版本

#     # predict stage specific
#     stage_class_num : int | None = None # 阶段分类个数
#     stage_mask_id: int | None = None # 预测阶段使用当前或者未来chunk
#     end_frame_num: int | None = None  # 结束关键帧数量

#     prob_pos: float | None = None  # 正样本的比例
#     prob_neg_name: float | None = None  # 负样本中更换任务名的比例

#     @override
#     def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
#         # 实例化顶层可 picklable 的统一变换
#         combined_transform = _CombinedPiperTransform(
#             action_dim=model_config.action_dim,
#             default_prompt=self.default_prompt,
#             gripper_position=self.gripper_position,
#         )
#         # 仅在 inputs 侧使用该统一变换（输出侧按需可添加）
#         final_transforms = _transforms.Group(inputs=[combined_transform])

#         # 模型级变换（如分词/FAST token）
#         model_transforms = ModelTransformFactory()(model_config)

#         # 返回完整数据配置（repack 为空，因为在 combined_transform 内部完成了重打包）
#         return dataclasses.replace(
#             self.create_base_config(assets_dirs, model_config),
#             repack_transforms=_transforms.Group(),
#             data_transforms=final_transforms,
#             model_transforms=model_transforms,
#             vlm_prediction_type=self.vlm_prediction_type,
#             prompt_version_for_subtask=self.prompt_version_for_subtask,
#             stage_mask_id=self.stage_mask_id,
#             stage_class_num=self.stage_class_num,
#             end_frame_num=self.end_frame_num,
#             prob_pos=self.prob_pos,
#             prob_neg_name=self.prob_neg_name,
#         )
# ############################# pi05_piper ############################
# #####################################################################




###ygx
@dataclasses.dataclass(frozen=True)
class LeRobotDualPiperDataConfig(DataConfigFactory):
    """Piper双臂机械臂数据配置（PI0.5微调专用）"""
    use_delta_actions: bool = True  # 14维绝对动作转delta
    dataset_path: str = "/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/test_data_15"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 键映射：对齐Piper数据集和PI0.5模型
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state": "observation.state",
                        "action": "action",
                        "task_index": "task_index",
                        "frame_index": "frame_index",
                        "episode_index": "episode_index",
                    }
                )
            ]
        )

        # 数据转换：Piper输入输出 + PI0.5适配
        data_transforms = _transforms.Group(
            inputs=[DualPiperInputs(model_type=model_config.model_type)],
            outputs=[DualPiperOutputs()],
        )

        # Piper双臂14维delta掩码（6+1+6+1）
        if self.use_delta_actions:
            delta_action_mask = np.array([
                True, True, True, True, True, True, False,
                True, True, True, True, True, True, False
            ])
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型转换：PI0.5默认逻辑
        model_transforms = ModelTransformFactory()(model_config)

        # 最终配置
        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base_config,
            repo_id=self.dataset_path,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            prompt_from_task=False,  # 使用Piper自定义任务描述
            action_sequence_keys=("action",),
        )

#ygx

@dataclasses.dataclass(frozen=True)
class LeRobotDualPiperDataConfig(DataConfigFactory):
    """适配Piper双臂机械臂的配置（14维绝对动作+自动读取任务描述）"""
    # 是否启用delta动作转换（绝对坐标→相对变化量）
    use_delta_actions: bool = True
    # Piper机械臂数据集根路径
    dataset_path: str = "/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/test_data_15"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. 重映射数据集键（对齐Piper机械臂模型预期）
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state": "observation.state",  # Piper 14维状态键
                        "action": "action",                        # Piper 14维动作键
                        "task_index": "task_index",                # 任务索引键
                        "frame_index": "frame_index",              # 帧索引（加载视频用）
                        "episode_index": "episode_index",          # 片段索引（加载视频用）
                    }
                )
            ]
        )

        # 2. 定义数据转换（使用Piper机械臂自定义的输入输出类）
        data_transforms = _transforms.Group(
            inputs=[DualPiperInputs(model_type=model_config.model_type)],
            outputs=[DualPiperOutputs()],
        )

        # 3. 适配Piper双臂14维绝对动作的delta转换
        # 掩码规则：0-5(左关节)/7-12(右关节)转delta，6(左夹爪)/13(右夹爪)保持绝对
        if self.use_delta_actions:
            delta_action_mask = np.array([
                True, True, True, True, True, True, False,
                True, True, True, True, True, True, False
            ])
            # 训练时：绝对→delta；推理时：delta→绝对
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 4. 模型转换（复用默认逻辑）
        model_transforms = ModelTransformFactory()(model_config)

        # 5. 构造最终配置
        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base_config,
            # Piper数据集ID（本地路径）
            repo_id=self.dataset_path,
            # 键映射转换
            repack_transforms=repack_transform,
            # 数据转换
            data_transforms=data_transforms,
            # 模型转换
            model_transforms=model_transforms,
            # 禁用自动从task生成提示（使用tasks.parquet的自定义描述）
            prompt_from_task=False,
            # 动作序列键
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            # repo_id="physical-intelligence/libero",
            repo_id="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/libero_2_lerobot",
            # repo_id="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/test_data_15",

            # 预训练模型的资产配置（可选，充分利用预训练数据的norm资产，更稳定）
            # assets=AssetsConfig(
            #     assets_dir="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home",
            #     asset_id="libero_2_lerobot",
            # ),

            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),

        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),

        # 用pytorch训练时不需要，可注释掉
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"), # 原始值
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/autodl-tmp/RoboParty_pi/openpi/openpi_data_home/openpi-assets/checkpoints/pi05_base/params"), # 原始值

        # pytorch_weight_path="/path/to/your/pytorch_weight_path", # 原始值 
        # pytorch_weight_path=None,  # 用jax训练初始base模型，不需要pytorch预训练权重
        # pytorch_weight_path="/root/autodl-tmp/RoboParty_pi/openpi/openpi_data_home/openpi-assets/checkpoints/pi05_libero_pytorch", # 用pytorch微调，要加载pytorch版的pi05_libero预训练权重     

        num_train_steps=30_000,  # 原始值 
        # num_train_steps=20,  # 仅运行 20 步用于测试

        # batch_size=256, # 原始值                         
        batch_size=2,   

        # num_workers=1, # 新加入参数

        # ema_decay=0.999, # 原始值
        ema_decay= None, # 禁用EMA，可减少模型参数的内存占用
    ),




# ############################################################
# ####################### pi05_piper #########################
#     #
#     # Fine-tuning piper configs 全参微调的训练配置
#     #

#     TrainConfig(
#         name="pi05_piper",
#         model=pi0_config.Pi0Config(
#             action_dim=32,
#             action_horizon=50,
#             pi05=True,
#             vlm_prediction_type='stage',
#             prob_pos=0.8,
#             coef_loss_subtask=0.1,
#             coef_loss_prompt=0.1,
#             coef_loss_progress=1.0,
#             coef_loss_stage=0.1,
#             stage_class_num=4,
#             stage_class_weights=(1.0, 1.0, 2.0, 2.0)
#         ),
#         lr_schedule=_optimizer.CosineDecaySchedule(
#             warmup_steps=2_000,
#             peak_lr=1e-4,
#             decay_steps=100_000,
#             decay_lr=1e-5,
#         ),
#         data=LeRobotPiperDataConfig(
#             # 你的数据集根路径（LeRobot 规范或你自定义转换后的路径）
#             repo_id="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/test_data_15",
#             base_config=DataConfig(
#                 prompt_from_task=False,
#             ),
#             default_prompt="",
#             prompt_version_for_subtask='v3',
#             gripper_position="15-16",
#             vlm_prediction_type='stage',
#             stage_mask_id=1,
#             stage_class_num=4,
#             end_frame_num=6,
#             prob_pos=0.8,
#             prob_neg_name=0.5,
#         ),

#         # 用jax训练
#         weight_loader=weight_loaders.CheckpointWeightLoader("/root/autodl-tmp/RoboParty_pi/openpi/openpi_data_home/openpi-assets/checkpoints/pi05_base/params"), # 原始值

#         # 用pytorch训练      
#         # pytorch_weight_path="/data/vjuicefs_ai_botdata/public_data/challenge/AgiBot_World_Challenge/model/openpi/checkpoints/pi05_base_pytorch",
#         num_train_steps=100_000,
#         batch_size=2,
#         num_workers=1,
#         # checkpoint_base_dir="/data/vjuicefs_ai_botdata/public_data/challenge/AgiBot_World_Challenge/model/openpi/checkpoints",
#         ema_decay=None,
#     ),
# ####################### pi05_piper #########################
# ############################################################


    # #ygx
    # TrainConfig(
    #     name="pi0_5_dual_piper_finetune",  # 唯一配置名（微调专用）
    #     # PI0.5模型核心配置（官方基准参数）
    #     model=pi0_config.Pi0Config(
    #         action_dim=14,               # Piper双臂14维动作
    #         action_horizon=10,           # 预测未来10步动作
    #         model_type=_model.ModelType.PI0_5,  # 指定PI0.5模型
    #         # PI0.5官方基准参数（不可随意修改）
    #         num_encoders=12,             # 编码器层数（PI0.5=12）
    #         num_decoders=12,             # 解码器层数（PI0.5=12）
    #         embed_dim=768,               # 嵌入维度（PI0.5=768）
    #         num_heads=12,                # 注意力头数（PI0.5=12）
    #         dropout=0.1,                 # 正则化（防止过拟合）
    #         mlp_ratio=4.0,               # 隐藏层维度比例
    #     ),
    #     # 数据配置：Piper专用
    #     data=LeRobotDualPiperDataConfig(
    #         use_delta_actions=True,
    #         dataset_path="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/test_data_15",
    #         base_config=DataConfig(
    #             prompt_from_task=False,
    #             batch_size=16,           # PI0.5显存优化：批量大小16（单卡A100可设32）
    #             train_split=0.9,         # 90%训练/10%验证
    #             num_workers=4,           # 数据加载线程
    #             shuffle=True,            # 训练集洗牌
    #             pin_memory=True,         # 显存锁定（加速）
    #         ),
    #     ),
    #     # PI0.5预训练权重（官方）
    #     weight_loader=weight_loaders.CheckpointWeightLoader(
    #         "/root/autodl-tmp/RoboParty_pi/openpi/openpi_data_home/openpi-assets/checkpoints/pi05_base/params"
    #     ),
    #     # 微调训练参数（轻量化，避免过拟合）
    #     num_train_steps=50000,          # PI0.5微调步数（比PI0多）
    #     log_interval=100,               # 每100步打印日志
    #     eval_interval=1000,             # 每1000步验证
    #     save_interval=5000,             # 每5000步保存模型
    #     learning_rate=1e-4,             # 微调学习率（预训练后降低）
    #     weight_decay=1e-5,              # 权重衰减（正则化）
    #     warmup_steps=1000,              # 学习率预热（稳定训练）
    #     seed=42,                        # 固定种子（可复现）
    #     mixed_precision=True,           # 混合精度训练（PI0.5显存优化）
    # ),
    # #ygx

    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
