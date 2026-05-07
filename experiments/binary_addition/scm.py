from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _int_to_bits_lsb(value: int, width: int) -> tuple[int, ...]:
    if value < 0 or value >= 2**width:
        raise ValueError(f"value must lie in [0, {2**width - 1}] for width={width}")
    return tuple((value >> idx) & 1 for idx in range(width))


def _bits_lsb_to_int(bits: tuple[int, ...]) -> int:
    return sum((int(bit) & 1) << idx for idx, bit in enumerate(bits))


def longest_active_carry_run(carries: tuple[int, ...]) -> int:
    best = 0
    cur = 0
    for bit in carries:
        if int(bit):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


@dataclass(frozen=True)
class BinaryAdditionExample:
    width: int
    a: int
    b: int
    a_bits_lsb: tuple[int, ...]
    b_bits_lsb: tuple[int, ...]
    sum_bits_lsb: tuple[int, ...]
    carries: tuple[int, ...]
    output_bits_lsb: tuple[int, ...]
    output_value: int
    propagation_length: int

    def carry(self, index: int) -> int:
        if index < 1 or index > self.width:
            raise ValueError(f"carry index must be in [1, {self.width}]")
        return int(self.carries[index - 1])

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "a": self.a,
            "b": self.b,
            "a_bits_lsb": list(self.a_bits_lsb),
            "b_bits_lsb": list(self.b_bits_lsb),
            "sum_bits_lsb": list(self.sum_bits_lsb),
            "carries": list(self.carries),
            "output_bits_lsb": list(self.output_bits_lsb),
            "output_value": self.output_value,
            "propagation_length": self.propagation_length,
        }


def compute_example(a: int, b: int, width: int = 4) -> BinaryAdditionExample:
    a_bits = _int_to_bits_lsb(a, width)
    b_bits = _int_to_bits_lsb(b, width)

    carry_in = 0
    sum_bits: list[int] = []
    carries: list[int] = []
    for idx in range(width):
        total = int(a_bits[idx]) + int(b_bits[idx]) + carry_in
        sum_bits.append(total & 1)
        carry_out = 1 if total >= 2 else 0
        carries.append(carry_out)
        carry_in = carry_out

    output_bits = tuple(sum_bits + [carries[-1]])
    return BinaryAdditionExample(
        width=width,
        a=a,
        b=b,
        a_bits_lsb=a_bits,
        b_bits_lsb=b_bits,
        sum_bits_lsb=tuple(sum_bits),
        carries=tuple(carries),
        output_bits_lsb=output_bits,
        output_value=_bits_lsb_to_int(output_bits),
        propagation_length=longest_active_carry_run(tuple(carries)),
    )


def intervene_carries(
    base: BinaryAdditionExample,
    forced_carries: Mapping[int, int],
) -> BinaryAdditionExample:
    width = base.width
    for index, value in forced_carries.items():
        if index < 1 or index > width:
            raise ValueError(f"carry index must be in [1, {width}]")
        if int(value) not in (0, 1):
            raise ValueError("forced carry values must be binary")

    carry_in = 0
    sum_bits: list[int] = []
    carries: list[int] = []
    for idx in range(width):
        total = int(base.a_bits_lsb[idx]) + int(base.b_bits_lsb[idx]) + carry_in
        sum_bits.append(total & 1)
        carry_out = 1 if total >= 2 else 0
        carry_out = int(forced_carries.get(idx + 1, carry_out))
        carries.append(carry_out)
        carry_in = carry_out

    output_bits = tuple(sum_bits + [carries[-1]])
    return BinaryAdditionExample(
        width=width,
        a=base.a,
        b=base.b,
        a_bits_lsb=base.a_bits_lsb,
        b_bits_lsb=base.b_bits_lsb,
        sum_bits_lsb=tuple(sum_bits),
        carries=tuple(carries),
        output_bits_lsb=output_bits,
        output_value=_bits_lsb_to_int(output_bits),
        propagation_length=longest_active_carry_run(tuple(carries)),
    )
