package tensor

import (
	"math"
	"testing"
)

func rawF32(t *testing.T, shape Shape, data []float32) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Float32, CPU)
	if err != nil {
		t.Fatalf("NewRaw float32: %v", err)
	}
	copy(raw.AsFloat32(), data)
	return raw
}

func rawF64(t *testing.T, shape Shape, data []float64) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Float64, CPU)
	if err != nil {
		t.Fatalf("NewRaw float64: %v", err)
	}
	copy(raw.AsFloat64(), data)
	return raw
}

func rawI64(t *testing.T, shape Shape, data []int64) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Int64, CPU)
	if err != nil {
		t.Fatalf("NewRaw int64: %v", err)
	}
	copy(raw.AsInt64(), data)
	return raw
}

func rawI32(t *testing.T, shape Shape, data []int32) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Int32, CPU)
	if err != nil {
		t.Fatalf("NewRaw int32: %v", err)
	}
	copy(raw.AsInt32(), data)
	return raw
}

func rawU8(t *testing.T, shape Shape, data []uint8) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Uint8, CPU)
	if err != nil {
		t.Fatalf("NewRaw uint8: %v", err)
	}
	copy(raw.AsUint8(), data)
	return raw
}

func rawBool(t *testing.T, shape Shape, data []bool) *RawTensor {
	t.Helper()
	raw, err := NewRaw(shape, Bool, CPU)
	if err != nil {
		t.Fatalf("NewRaw bool: %v", err)
	}
	copy(raw.AsBool(), data)
	return raw
}

func TestMod_Float32_Fmod0(t *testing.T) {
	// fmod=0: floor_mod (Python % style: sign follows divisor)
	// a: [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5]
	// b: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	// want: [1.5, 1.0, 0.5, 0.0, 1.5, 1.0, 0.5]
	a := rawF32(t, Shape{7}, []float32{-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5})
	b := rawF32(t, Shape{7}, []float32{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0})

	out, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod failed: %v", err)
	}

	got := out.AsFloat32()
	want := []float32{1.5, 1.0, 0.5, 0.0, 1.5, 1.0, 0.5}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMod_Float32_Fmod1(t *testing.T) {
	// fmod=1: C fmod style (sign follows dividend)
	// a: [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5]
	// b: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
	// want: [-0.5, -1.0, -1.5, 0.0, 1.5, 1.0, 0.5]
	a := rawF32(t, Shape{7}, []float32{-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5})
	b := rawF32(t, Shape{7}, []float32{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0})

	out, err := Mod(a, b, 1)
	if err != nil {
		t.Fatalf("Mod failed: %v", err)
	}

	got := out.AsFloat32()
	want := []float32{-0.5, -1.0, -1.5, 0.0, 1.5, 1.0, 0.5}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMod_Float64(t *testing.T) {
	a := rawF64(t, Shape{2}, []float64{-3.5, 3.5})
	b := rawF64(t, Shape{2}, []float64{2.0, -2.0})

	// fmod=0:
	// -3.5 mod 2.0 = 0.5 (sign of 2.0)
	// 3.5 mod -2.0 = -0.5 (sign of -2.0)
	out0, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod fmod=0 failed: %v", err)
	}
	got0 := out0.AsFloat64()
	if math.Abs(got0[0]-0.5) > 1e-6 || math.Abs(got0[1]-(-0.5)) > 1e-6 {
		t.Fatalf("fmod=0 got %v, want [0.5, -0.5]", got0)
	}

	// fmod=1:
	// -3.5 fmod 2.0 = -1.5 (sign of -3.5)
	// 3.5 fmod -2.0 = 1.5 (sign of 3.5)
	out1, err := Mod(a, b, 1)
	if err != nil {
		t.Fatalf("Mod fmod=1 failed: %v", err)
	}
	got1 := out1.AsFloat64()
	if math.Abs(got1[0]-(-1.5)) > 1e-6 || math.Abs(got1[1]-1.5) > 1e-6 {
		t.Fatalf("fmod=1 got %v, want [-1.5, 1.5]", got1)
	}
}

func TestMod_Int64(t *testing.T) {
	a := rawI64(t, Shape{5}, []int64{-10, -10, 10, 10, math.MinInt64})
	b := rawI64(t, Shape{5}, []int64{3, -3, 3, -3, -1})

	// fmod=0 (floor_mod):
	// -10 % 3 = 2
	// -10 % -3 = -1
	// 10 % 3 = 1
	// 10 % -3 = -2
	// MinInt64 % -1 = 0
	out0, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod fmod=0 failed: %v", err)
	}
	got0 := out0.AsInt64()
	want0 := []int64{2, -1, 1, -2, 0}
	for i := range want0 {
		if got0[i] != want0[i] {
			t.Fatalf("fmod=0 [%d]: got %d, want %d", i, got0[i], want0[i])
		}
	}

	// fmod=1 (fmod / truncated):
	// -10 % 3 = -1
	// -10 % -3 = -1
	// 10 % 3 = 1
	// 10 % -3 = 1
	// MinInt64 % -1 = 0
	out1, err := Mod(a, b, 1)
	if err != nil {
		t.Fatalf("Mod fmod=1 failed: %v", err)
	}
	got1 := out1.AsInt64()
	want1 := []int64{-1, -1, 1, 1, 0}
	for i := range want1 {
		if got1[i] != want1[i] {
			t.Fatalf("fmod=1 [%d]: got %d, want %d", i, got1[i], want1[i])
		}
	}
}

func TestMod_Int32(t *testing.T) {
	a := rawI32(t, Shape{3}, []int32{-7, 7, math.MinInt32})
	b := rawI32(t, Shape{3}, []int32{4, -4, -1})

	out0, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod fmod=0 failed: %v", err)
	}
	got0 := out0.AsInt32()
	want0 := []int32{1, -1, 0}
	for i := range want0 {
		if got0[i] != want0[i] {
			t.Fatalf("fmod=0 [%d]: got %d, want %d", i, got0[i], want0[i])
		}
	}

	out1, err := Mod(a, b, 1)
	if err != nil {
		t.Fatalf("Mod fmod=1 failed: %v", err)
	}
	got1 := out1.AsInt32()
	want1 := []int32{-3, 3, 0}
	for i := range want1 {
		if got1[i] != want1[i] {
			t.Fatalf("fmod=1 [%d]: got %d, want %d", i, got1[i], want1[i])
		}
	}
}

func TestMod_Uint8(t *testing.T) {
	a := rawU8(t, Shape{3}, []uint8{10, 25, 255})
	b := rawU8(t, Shape{3}, []uint8{3, 7, 10})

	out, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod uint8 failed: %v", err)
	}
	got := out.AsUint8()
	want := []uint8{1, 4, 5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestMod_Broadcasting(t *testing.T) {
	// a: shape [2, 1, 3]
	// b: shape [1, 2, 3]
	// out: shape [2, 2, 3]
	aData := []float32{
		-4, 5, -6,
		7, -8, 9,
	}
	bData := []float32{
		3, 3, 3,
		4, 4, 4,
	}
	a := rawF32(t, Shape{2, 1, 3}, aData)
	b := rawF32(t, Shape{1, 2, 3}, bData)

	out, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod broadcast failed: %v", err)
	}
	if !out.Shape().Equal(Shape{2, 2, 3}) {
		t.Fatalf("expected shape [2, 2, 3], got %v", out.Shape())
	}

	got := out.AsFloat32()
	// [2, 2, 3]
	// batch 0, row 0: [-4, 5, -6] mod [3, 3, 3] = [2, 2, 0]
	// batch 0, row 1: [-4, 5, -6] mod [4, 4, 4] = [0, 1, 2]
	// batch 1, row 0: [7, -8, 9] mod [3, 3, 3] = [1, 1, 0]
	// batch 1, row 1: [7, -8, 9] mod [4, 4, 4] = [3, 0, 1]
	want := []float32{
		2, 2, 0,
		0, 1, 2,
		1, 1, 0,
		3, 0, 1,
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("[%d]: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMod_ScalarBroadcasting(t *testing.T) {
	a := rawF32(t, Shape{4}, []float32{10, 11, 12, 13})
	b := rawF32(t, Shape{1}, []float32{3})

	out, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod scalar failed: %v", err)
	}
	got := out.AsFloat32()
	want := []float32{1, 2, 0, 1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d]: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMod_ScalarShape(t *testing.T) {
	a := rawF32(t, Shape{}, []float32{10})
	b := rawF32(t, Shape{}, []float32{3})

	out, err := Mod(a, b, 0)
	if err != nil {
		t.Fatalf("Mod scalar shape failed: %v", err)
	}
	if !out.Shape().Equal(Shape{}) {
		t.Fatalf("expected shape {}, got %v", out.Shape())
	}
	got := out.AsFloat32()
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("got %v, want [1]", got)
	}
}

func TestMod_Errors(t *testing.T) {
	t.Run("nil tensor", func(t *testing.T) {
		a := rawF32(t, Shape{1}, []float32{1})
		if _, err := Mod(nil, a, 0); err == nil {
			t.Fatal("expected error for nil a, got nil")
		}
		if _, err := Mod(a, nil, 0); err == nil {
			t.Fatal("expected error for nil b, got nil")
		}
	})

	t.Run("mismatched dtypes", func(t *testing.T) {
		a := rawF32(t, Shape{1}, []float32{1})
		b := rawI64(t, Shape{1}, []int64{1})
		if _, err := Mod(a, b, 0); err == nil {
			t.Fatal("expected error for mismatched dtypes, got nil")
		}
	})

	t.Run("incompatible shapes", func(t *testing.T) {
		a := rawF32(t, Shape{3}, []float32{1, 2, 3})
		b := rawF32(t, Shape{2}, []float32{1, 2})
		if _, err := Mod(a, b, 0); err == nil {
			t.Fatal("expected error for incompatible shapes, got nil")
		}
	})

	t.Run("division by zero integer", func(t *testing.T) {
		a := rawI64(t, Shape{1}, []int64{10})
		b := rawI64(t, Shape{1}, []int64{0})
		if _, err := Mod(a, b, 0); err == nil {
			t.Fatal("expected error for integer division by zero, got nil")
		}

		u1 := rawU8(t, Shape{1}, []uint8{10})
		u2 := rawU8(t, Shape{1}, []uint8{0})
		if _, err := Mod(u1, u2, 0); err == nil {
			t.Fatal("expected error for uint8 division by zero, got nil")
		}
	})

	t.Run("unsupported dtype", func(t *testing.T) {
		a := rawBool(t, Shape{1}, []bool{true})
		b := rawBool(t, Shape{1}, []bool{false})
		if _, err := Mod(a, b, 0); err == nil {
			t.Fatal("expected error for unsupported dtype bool, got nil")
		}
	})
}
