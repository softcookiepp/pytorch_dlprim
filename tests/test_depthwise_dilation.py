import argparse
import torch


def run_case(device: str, shape, kernel: int, dilation: int):
    n, c, h, w = shape
    groups = c
    out_channels = c
    padding = (kernel - 1) * dilation // 2

    x_cpu = torch.randn(n, c, h, w, requires_grad=True)
    w_cpu = torch.randn(out_channels, c // groups, kernel, kernel, requires_grad=True)
    b_cpu = torch.randn(out_channels, requires_grad=True)

    y_cpu = torch.nn.functional.conv2d(
        x_cpu,
        w_cpu,
        b_cpu,
        stride=1,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    grad_out_cpu = torch.randn_like(y_cpu)
    y_cpu.backward(grad_out_cpu)

    x_dev = x_cpu.detach().to(device).requires_grad_(True)
    w_dev = w_cpu.detach().to(device).requires_grad_(True)
    b_dev = b_cpu.detach().to(device).requires_grad_(True)

    y_dev = torch.nn.functional.conv2d(
        x_dev,
        w_dev,
        b_dev,
        stride=1,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    y_dev.backward(grad_out_cpu.to(device))

    y_err = (y_cpu.detach() - y_dev.detach().cpu()).abs().max().item()
    dx_err = (x_cpu.grad - x_dev.grad.cpu()).abs().max().item()
    dw_err = (w_cpu.grad - w_dev.grad.cpu()).abs().max().item()
    db_err = (b_cpu.grad - b_dev.grad.cpu()).abs().max().item()

    print(
        f"shape={shape} k={kernel} d={dilation} "
        f"y={y_err:.8f} dx={dx_err:.8f} dw={dw_err:.8f} db={db_err:.8f}"
    )

    assert y_err < 1e-5
    assert dx_err < 2e-5
    assert dw_err < 5e-5
    assert db_err < 2e-5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="ocl:0")
    args = p.parse_args()

    if args.device.startswith("ocl") or args.device.startswith("privateuseone"):
        import pytorch_ocl  # noqa: F401

    torch.manual_seed(0)
    run_case(args.device, (2, 8, 12, 12), kernel=3, dilation=1)
    run_case(args.device, (2, 8, 12, 12), kernel=3, dilation=2)
    run_case(args.device, (1, 16, 20, 20), kernel=5, dilation=2)

    print("depthwise dilation regression: OK")


if __name__ == "__main__":
    main()
