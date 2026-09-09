import argparse
import warnings

import torch


def fallback_messages(caught):
    return [str(item.message) for item in caught if "wildcard fallback invoked" in str(item.message)]


def run_tests(device):
    import pytorch_vk

    previous = pytorch_vk.get_fallback_strict()
    pytorch_vk.set_fallback_strict(False)
    try:
        before = pytorch_vk.get_fallback_count()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = torch.triu_indices(3, 3, device=device)

        messages = fallback_messages(caught)
        assert result.device == torch.device(device)
        assert len(messages) == 1
        assert "aten::triu_indices" in messages[0]
        assert device in messages[0]
        assert "no PrivateUse1 kernel registered" in messages[0]
        assert (
            "The operator 'aten::triu_indices' is not currently supported on the vk backend. "
            "Please open an issue at for requesting support "
            "https://github.com/softcookiepp/pytorch_dlprim/issues"
        ) in messages[0]
        assert pytorch_vk.get_fallback_count() == before + 1

        pytorch_vk.set_fallback_strict(True)
        try:
            torch.triu_indices(3, 3, device=device)
        except RuntimeError as error:
            message = str(error)
            assert "aten::triu_indices" in message
            assert device in message
            assert "no PrivateUse1 kernel registered" in message
        else:
            raise AssertionError("strict fallback mode allowed CPU fallback")
    finally:
        pytorch_vk.set_fallback_strict(previous)

    print("fallback policy regression: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
    run_tests(parser.parse_args().device)
