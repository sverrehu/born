//go:build !wasm

package operators

import (
	"math"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

func TestMod_Float32_DefaultFmod(t *testing.T) {
	// Default fmod is 0 (floor_mod)
	a := f32Tensor(t, tensor.Shape{4}, []float32{-4.5, -3.0, 1.5, 3.0})
	b := f32Tensor(t, tensor.Shape{4}, []float32{2.0, 2.0, 2.0, 2.0})

	out := execOp(t, "Mod", nil, a, b)
	assertShape(t, out, tensor.Shape{4})
	assertClose(t, out.AsFloat32(), []float32{1.5, 1.0, 1.5, 1.0})
}

func TestMod_Float32_Fmod1(t *testing.T) {
	// fmod=1 (truncated / C fmod)
	a := f32Tensor(t, tensor.Shape{4}, []float32{-4.5, -3.0, 1.5, 3.0})
	b := f32Tensor(t, tensor.Shape{4}, []float32{2.0, 2.0, 2.0, 2.0})

	out := execOp(t, "Mod", []Attribute{{Name: "fmod", I: 1}}, a, b)
	assertShape(t, out, tensor.Shape{4})
	assertClose(t, out.AsFloat32(), []float32{-0.5, -1.0, 1.5, 1.0})
}

func TestMod_Float64(t *testing.T) {
	a := f64Tensor(t, tensor.Shape{2}, []float64{-3.5, 3.5})
	b := f64Tensor(t, tensor.Shape{2}, []float64{2.0, -2.0})

	out := execOp(t, "Mod", []Attribute{{Name: "fmod", I: 0}}, a, b)
	assertShape(t, out, tensor.Shape{2})
	got := out.AsFloat64()
	if math.Abs(got[0]-0.5) > 1e-6 || math.Abs(got[1]-(-0.5)) > 1e-6 {
		t.Fatalf("got %v, want [0.5, -0.5]", got)
	}
}

func TestMod_Int64(t *testing.T) {
	a := i64Tensor(t, []int64{-10, 10})
	b := i64Tensor(t, []int64{3, -3})

	out0 := execOp(t, "Mod", []Attribute{{Name: "fmod", I: 0}}, a, b)
	got0 := out0.AsInt64()
	if got0[0] != 2 || got0[1] != -2 {
		t.Fatalf("fmod=0 got %v, want [2, -2]", got0)
	}

	out1 := execOp(t, "Mod", []Attribute{{Name: "fmod", I: 1}}, a, b)
	got1 := out1.AsInt64()
	if got1[0] != -1 || got1[1] != 1 {
		t.Fatalf("fmod=1 got %v, want [-1, 1]", got1)
	}
}

func TestMod_Int32(t *testing.T) {
	rawA, err := tensor.NewRaw(tensor.Shape{2}, tensor.Int32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw int32 a: %v", err)
	}
	copy(rawA.AsInt32(), []int32{-7, 7})

	rawB, err := tensor.NewRaw(tensor.Shape{2}, tensor.Int32, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw int32 b: %v", err)
	}
	copy(rawB.AsInt32(), []int32{4, -4})

	out := execOp(t, "Mod", []Attribute{{Name: "fmod", I: 0}}, rawA, rawB)
	got := out.AsInt32()
	if got[0] != 1 || got[1] != -1 {
		t.Fatalf("got %v, want [1, -1]", got)
	}
}

func TestMod_Uint8(t *testing.T) {
	rawA, err := tensor.NewRaw(tensor.Shape{2}, tensor.Uint8, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw uint8 a: %v", err)
	}
	copy(rawA.AsUint8(), []uint8{25, 255})

	rawB, err := tensor.NewRaw(tensor.Shape{2}, tensor.Uint8, tensor.CPU)
	if err != nil {
		t.Fatalf("NewRaw uint8 b: %v", err)
	}
	copy(rawB.AsUint8(), []uint8{7, 10})

	out := execOp(t, "Mod", nil, rawA, rawB)
	got := out.AsUint8()
	if got[0] != 4 || got[1] != 5 {
		t.Fatalf("got %v, want [4, 5]", got)
	}
}

func TestMod_Broadcast(t *testing.T) {
	a := f32Tensor(t, tensor.Shape{2, 3}, []float32{
		-4, 5, -6,
		7, -8, 9,
	})
	b := f32Tensor(t, tensor.Shape{1, 3}, []float32{3, 3, 3})

	out := execOp(t, "Mod", nil, a, b)
	assertShape(t, out, tensor.Shape{2, 3})
	assertClose(t, out.AsFloat32(), []float32{
		2, 2, 0,
		1, 1, 0,
	})
}

func TestMod_ValidationErrors(t *testing.T) {
	a := f32Tensor(t, tensor.Shape{2}, []float32{1, 2})
	b := f32Tensor(t, tensor.Shape{2}, []float32{3, 4})

	if err := execOpErr("Mod", nil); err == nil {
		t.Fatal("expected error for 0 inputs, got nil")
	}
	if err := execOpErr("Mod", nil, a); err == nil {
		t.Fatal("expected error for 1 input, got nil")
	}
	if err := execOpErr("Mod", nil, a, b, a); err == nil {
		t.Fatal("expected error for 3 inputs, got nil")
	}
	if err := execOpErr("Mod", nil, a, nil); err == nil {
		t.Fatal("expected error for nil input, got nil")
	}
	if err := execOpErr("Mod", nil, nil, b); err == nil {
		t.Fatal("expected error for nil input, got nil")
	}

	bI64 := i64Tensor(t, []int64{1, 2})
	if err := execOpErr("Mod", nil, a, bI64); err == nil {
		t.Fatal("expected error for mismatched dtypes, got nil")
	}

	divZero := i64Tensor(t, []int64{0, 2})
	if err := execOpErr("Mod", nil, bI64, divZero); err == nil {
		t.Fatal("expected error for integer division by zero, got nil")
	}
}
