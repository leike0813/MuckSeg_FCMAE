import yaml
import warnings
import copy
from collections.abc import Sequence
from yacs.config import CfgNode, _assert_with_logging, _valid_type, _VALID_TYPES


class CustomCfgNode(CfgNode):
    VISIBLE = "__visible__"
    INVISIBLE_KEYS = "__invisible_keys__"
    KEYS_WITHOUT_TYPECHECK = "__keys_without_typecheck__"

    def __init__(self, init_dict=None, key_list=None, new_allowed=False, visible=True):
        super(CustomCfgNode, self).__init__(init_dict, key_list, new_allowed)
        self.__dict__[CustomCfgNode.VISIBLE] = visible
        self.__dict__[CustomCfgNode.INVISIBLE_KEYS] = set()
        self.__dict__[CustomCfgNode.KEYS_WITHOUT_TYPECHECK] = set()

    def is_visible(self):
        return self.__dict__[CustomCfgNode.VISIBLE]

    def _visible(self, is_visible):
        self.__dict__[CustomCfgNode.VISIBLE] = is_visible

    def set_invisible_keys(self, keys):
        if not isinstance(keys, Sequence):
            keys = [keys]
        for key in keys:
            if key in self.__dict__:
                warnings.warn(
                    "Invalid attempt to change visibility internal CfgNode state: {}, the attempt will be ignored".format(key),
                    UserWarning
                )
                continue
            if key not in self.keys():
                warnings.warn(
                    "{} is not a valid CfgNode key and will be ignored".format(key),
                    UserWarning
                )
                continue
            self.__dict__[CustomCfgNode.INVISIBLE_KEYS].add(key)

    def set_visible_keys(self, keys):
        if not isinstance(keys, Sequence):
            keys = [keys]
        for key in keys:
            if key in self.__dict__:
                warnings.warn(
                    "Cannot change visibility of internal CfgNode state: {}".format(
                        key),
                    UserWarning
                )
                continue
            if key not in self.keys():
                warnings.warn(
                    "{} is not a valid CfgNode key and will be ignored".format(key),
                    UserWarning
                )
                continue
            if key not in self.__dict__[CustomCfgNode.INVISIBLE_KEYS]:
                warnings.warn(
                    "{} is not invisible and will be ignored".format(key),
                    UserWarning
                )
                continue
            self.__dict__[CustomCfgNode.INVISIBLE_KEYS].remove(key)

    def set_typecheck_exclude_keys(self, keys):
        if not isinstance(keys, Sequence):
            keys = [keys]
        for key in keys:
            if key in self.__dict__:
                warnings.warn(
                    "Invalid attempt to change property internal CfgNode state: {}, the attempt will be ignored".format(key),
                    UserWarning
                )
                continue
            if key not in self.keys():
                warnings.warn(
                    "{} is not a valid CfgNode key and will be ignored".format(key),
                    UserWarning
                )
                continue
            self.__dict__[CustomCfgNode.KEYS_WITHOUT_TYPECHECK].add(key)

    def set_typecheck_include_keys(self, keys):
        if not isinstance(keys, Sequence):
            keys = [keys]
        for key in keys:
            if key in self.__dict__:
                warnings.warn(
                    "Cannot change property of internal CfgNode state: {}".format(
                        key),
                    UserWarning
                )
                continue
            if key not in self.keys():
                warnings.warn(
                    "{} is not a valid CfgNode key and will be ignored".format(key),
                    UserWarning
                )
                continue
            if key not in self.__dict__[CustomCfgNode.KEYS_WITHOUT_TYPECHECK]:
                warnings.warn(
                    "{} is not excluded from type checking and will be ignored".format(key),
                    UserWarning
                )
                continue
            self.__dict__[CustomCfgNode.KEYS_WITHOUT_TYPECHECK].remove(key)

    def invisible_keys(self):
        return tuple(self.__dict__[CustomCfgNode.INVISIBLE_KEYS])

    def typecheck_excluded_keys(self):
        return tuple(self.__dict__[CustomCfgNode.KEYS_WITHOUT_TYPECHECK])

    def dump_visible(self, **kwargs):
        """Dump to a string."""

        def convert_to_dict(cfg_node, key_list):
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
                return cfg_node
            else:
                if cfg_node.is_visible():
                    cfg_dict = dict(cfg_node)
                    k_to_pop = set()
                    for k, v in cfg_dict.items():
                        ret = convert_to_dict(v, key_list + [k])
                        if ret != {}:
                            cfg_dict[k] = ret
                        else:
                            k_to_pop.add(k)
                    for k in k_to_pop:
                        cfg_dict.pop(k)
                    for k in cfg_node.invisible_keys():
                        cfg_dict.pop(k)
                    return cfg_dict
                return {}

        self_as_dict = convert_to_dict(self, [])
        return yaml.safe_dump(self_as_dict, **kwargs)

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CfgNode."""
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in d, "Non-existent key: {}".format(full_key)
                )
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
            value = self._decode_cfg_value(v)
            if not subkey in d.typecheck_excluded_keys():  # added by Joshua Reed
                value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
            d[subkey] = value


def _merge_a_into_b(a, b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)

        if k in b:
            if not k in b.typecheck_excluded_keys(): # added by Joshua Reed
                v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
            # Recursively merge dicts
            if isinstance(v, CfgNode):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])
                except BaseException:
                    raise
            else:
                b[k] = v
        elif b.is_new_allowed():
            b[k] = v
        else:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                raise KeyError("Non-existent config key: {}".format(full_key))


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]
    casts.extend([(int, float), (int, str), (float, str)]) # Added by Joshua Reed
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )