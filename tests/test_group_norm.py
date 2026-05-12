import argparse
import torch


def run_case(device: str, shape, groups: int, eps: float = 1e-5):
    n, c, h, w = shape
    x_cpu = torch.randn(n, c, h, w, requires_grad=True)
    grad_out_cpu = torch.randn(n, c, h, w)

    gn_cpu = torch.nn.GroupNorm(groups, c, eps=eps)
    y_cpu = gn_cpu(x_cpu)
    y_cpu.backward(grad_out_cpu)

    x_dev = x_cpu.detach().to(device).requires_grad_(True)
    grad_out_dev = grad_out_cpu.to(device)
    gn_dev = torch.nn.GroupNorm(groups, c, eps=eps).to(device)
    with torch.no_grad():
        gn_dev.weight.copy_(gn_cpu.weight)
        gn_dev.bias.copy_(gn_cpu.bias)

    y_dev = gn_dev(x_dev)
    y_dev.backward(grad_out_dev)

    y_err = (y_cpu.detach() - y_dev.detach().cpu()).abs().max().item()
    dx_err = (x_cpu.grad - x_dev.grad.cpu()).abs().max().item()
    dw_err = (gn_cpu.weight.grad - gn_dev.weight.grad.cpu()).abs().max().item()
    db_err = (gn_cpu.bias.grad - gn_dev.bias.grad.cpu()).abs().max().item()

    print(f"shape={shape} groups={groups} y={y_err:.8f} dx={dx_err:.8f} dw={dw_err:.8f} db={db_err:.8f}")

    assert y_err < 1e-5
    assert dx_err < 1e-5
    assert dw_err < 2e-5
    assert db_err < 2e-5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="ocl:0")
    args = p.parse_args()

    if args.device.startswith("ocl") or args.device.startswith("privateuseone"):
        import pytorch_ocl  # noqa: F401

    torch.manual_seed(0)
    run_case(args.device, (2, 8, 4, 4), groups=2)
    run_case(args.device, (4, 16, 8, 8), groups=4)

    print("group norm regression: OK")


if __name__ == "__main__":
    main()
