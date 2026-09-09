import argparse

import torch


def assert_same(cpu, vk):
    assert cpu.shape == vk.shape
    assert cpu.stride() == vk.stride()
    torch.testing.assert_close(cpu, vk.cpu())


def run_tests(device):
    import pytorch_vk

    previous = pytorch_vk.get_fallback_strict()
    pytorch_vk.set_fallback_strict(True)
    try:
        cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        before = pytorch_vk.get_fallback_count()
        vk = cpu.to(device)
        assert_same(cpu, vk)

        cpu_view = cpu.reshape(2, 12)
        vk_view = vk.reshape(2, 12)
        assert_same(cpu_view, vk_view)

        cpu_transposed = cpu.t()
        vk_transposed = vk.t()
        assert_same(cpu_transposed, vk_transposed)
        assert_same(cpu_transposed.contiguous(), vk_transposed.contiguous())

        cpu_destination = torch.zeros_like(cpu)
        vk_destination = torch.zeros_like(vk)
        cpu_destination.copy_(cpu)
        vk_destination.copy_(vk)
        assert_same(cpu_destination, vk_destination)

        cpu_strided = torch.zeros(8, 6, dtype=torch.float32)[::2]
        vk_strided = torch.zeros(8, 6, dtype=torch.float32, device=device)[::2]
        cpu_strided.copy_(cpu)
        vk_strided.copy_(vk)
        assert_same(cpu_strided, vk_strided)

        cpu_input = torch.arange(24, dtype=torch.float32).reshape(4, 6).requires_grad_()
        vk_input = cpu_input.detach().to(device).requires_grad_()
        cpu_loss = cpu_input.t().reshape(-1).pow(2).sum()
        vk_loss = vk_input.t().reshape(-1).pow(2).sum()
        cpu_loss.backward()
        vk_loss.backward()
        torch.testing.assert_close(cpu_input.grad, vk_input.grad.cpu())

        assert pytorch_vk.get_fallback_count() == before
    finally:
        pytorch_vk.set_fallback_strict(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
    run_tests(parser.parse_args().device)
    print("layout differential regression: OK")
