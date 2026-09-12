package tensor

import (
	"reflect"
	"testing"
)

func TestTile_1D(t *testing.T) {
	// input: [1, 2, 3]
	x, err := NewRaw(Shape{3}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3})

	out, err := Tile(x, []int64{3})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{9}) {
		t.Errorf("got shape %v, want [9]", out.Shape())
	}

	expected := []float32{1, 2, 3, 1, 2, 3, 1, 2, 3}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestTile_2D_Axis0(t *testing.T) {
	// input: [[1, 2], [3, 4]]
	x, err := NewRaw(Shape{2, 2}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	out, err := Tile(x, []int64{2, 1})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{4, 2}) {
		t.Errorf("got shape %v, want [4 2]", out.Shape())
	}

	expected := []float32{1, 2, 3, 4, 1, 2, 3, 4}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestTile_2D_Axis1(t *testing.T) {
	// input: [[1, 2], [3, 4]]
	x, err := NewRaw(Shape{2, 2}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	out, err := Tile(x, []int64{1, 2})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{2, 4}) {
		t.Errorf("got shape %v, want [2 4]", out.Shape())
	}

	expected := []float32{1, 2, 1, 2, 3, 4, 3, 4}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestTile_2D_BothAxes(t *testing.T) {
	// input: [[1, 2, 3], [4, 5, 6]]
	x, err := NewRaw(Shape{2, 3}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4, 5, 6})

	out, err := Tile(x, []int64{2, 2})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{4, 6}) {
		t.Errorf("got shape %v, want [4 6]", out.Shape())
	}

	expected := []float32{
		1, 2, 3, 1, 2, 3,
		4, 5, 6, 4, 5, 6,
		1, 2, 3, 1, 2, 3,
		4, 5, 6, 4, 5, 6,
	}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestTile_3D(t *testing.T) {
	// input: shape [2, 1, 2]
	x, err := NewRaw(Shape{2, 1, 2}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	out, err := Tile(x, []int64{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{2, 2, 6}) {
		t.Errorf("got shape %v, want [2 2 6]", out.Shape())
	}

	expected := []float32{
		// batch 0, row 0
		1, 2, 1, 2, 1, 2,
		// batch 0, row 1
		1, 2, 1, 2, 1, 2,
		// batch 1, row 0
		3, 4, 3, 4, 3, 4,
		// batch 1, row 1
		3, 4, 3, 4, 3, 4,
	}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestTile_Identity(t *testing.T) {
	x, err := NewRaw(Shape{2, 3}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4, 5, 6})

	out, err := Tile(x, []int64{1, 1})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{2, 3}) {
		t.Errorf("got shape %v, want [2 3]", out.Shape())
	}
	if !reflect.DeepEqual(out.AsFloat32(), x.AsFloat32()) {
		t.Errorf("got %v, want %v", out.AsFloat32(), x.AsFloat32())
	}
}

func TestTile_Scalar(t *testing.T) {
	x, err := NewRaw(Shape{}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	x.AsFloat32()[0] = 42

	out, err := Tile(x, []int64{})
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(out.Shape(), Shape{}) {
		t.Errorf("got shape %v, want []", out.Shape())
	}
	if out.AsFloat32()[0] != 42 {
		t.Errorf("got %f, want 42", out.AsFloat32()[0])
	}
}

func TestTile_AllDTypes(t *testing.T) {
	repeats := []int64{2, 1}

	// Float64
	x64, _ := NewRaw(Shape{1, 2}, Float64, CPU)
	copy(x64.AsFloat64(), []float64{1.5, 2.5})
	out64, err := Tile(x64, repeats)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out64.AsFloat64(), []float64{1.5, 2.5, 1.5, 2.5}) {
		t.Errorf("Float64 got %v", out64.AsFloat64())
	}

	// Int64
	xI64, _ := NewRaw(Shape{1, 2}, Int64, CPU)
	copy(xI64.AsInt64(), []int64{10, 20})
	outI64, err := Tile(xI64, repeats)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outI64.AsInt64(), []int64{10, 20, 10, 20}) {
		t.Errorf("Int64 got %v", outI64.AsInt64())
	}

	// Int32
	xI32, _ := NewRaw(Shape{1, 2}, Int32, CPU)
	copy(xI32.AsInt32(), []int32{30, 40})
	outI32, err := Tile(xI32, repeats)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outI32.AsInt32(), []int32{30, 40, 30, 40}) {
		t.Errorf("Int32 got %v", outI32.AsInt32())
	}

	// Uint8
	xU8, _ := NewRaw(Shape{1, 2}, Uint8, CPU)
	copy(xU8.AsUint8(), []uint8{50, 60})
	outU8, err := Tile(xU8, repeats)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outU8.AsUint8(), []uint8{50, 60, 50, 60}) {
		t.Errorf("Uint8 got %v", outU8.AsUint8())
	}

	// Bool
	xBool, _ := NewRaw(Shape{1, 2}, Bool, CPU)
	copy(xBool.AsBool(), []bool{true, false})
	outBool, err := Tile(xBool, repeats)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outBool.AsBool(), []bool{true, false, true, false}) {
		t.Errorf("Bool got %v", outBool.AsBool())
	}
}

func TestTile_Errors(t *testing.T) {
	// Nil input
	if _, err := Tile(nil, []int64{1}); err == nil {
		t.Error("expected error for nil input")
	}

	// Rank mismatch
	x, _ := NewRaw(Shape{2, 3}, Float32, CPU)
	if _, err := Tile(x, []int64{2}); err == nil {
		t.Error("expected error for repeats length mismatch")
	}
	if _, err := Tile(x, []int64{2, 2, 2}); err == nil {
		t.Error("expected error for repeats length mismatch")
	}

	// Zero or negative repeat values
	if _, err := Tile(x, []int64{0, 1}); err == nil {
		t.Error("expected error for zero repeat value")
	}
	if _, err := Tile(x, []int64{1, -1}); err == nil {
		t.Error("expected error for negative repeat value")
	}
}
