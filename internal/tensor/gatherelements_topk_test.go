package tensor

import (
	"reflect"
	"testing"
)

// GatherElements Tests

func TestGatherElements_2D_Axis1(t *testing.T) {
	// data: [[1, 2], [3, 4]]
	data, err := NewRaw(Shape{2, 2}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(data.AsFloat32(), []float32{1, 2, 3, 4})

	// indices: [[0, 0], [1, 0]]
	idx, err := NewRaw(Shape{2, 2}, Int64, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(idx.AsInt64(), []int64{0, 0, 1, 0})

	out, err := GatherElements(data, idx, 1)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{1, 1, 4, 3}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestGatherElements_2D_Axis0(t *testing.T) {
	// data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	data, err := NewRaw(Shape{3, 3}, Float32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(data.AsFloat32(), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})

	// indices: [[1, 2, 0], [2, 0, 1]]
	idx, err := NewRaw(Shape{2, 3}, Int32, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(idx.AsInt32(), []int32{1, 2, 0, 2, 0, 1})

	out, err := GatherElements(data, idx, 0)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{4, 8, 3, 7, 2, 6}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestGatherElements_NegativeIndicesAndAxis(t *testing.T) {
	// data: [[1, 2, 3], [4, 5, 6]]
	data, err := NewRaw(Shape{2, 3}, Float64, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(data.AsFloat64(), []float64{1, 2, 3, 4, 5, 6})

	// indices: [[-1, -2], [0, -1]] -> [2, 1], [0, 2]
	idx, err := NewRaw(Shape{2, 2}, Int64, CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(idx.AsInt64(), []int64{-1, -2, 0, -1})

	// axis = -1 is axis 1
	out, err := GatherElements(data, idx, -1)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{3, 2, 4, 6}
	if !reflect.DeepEqual(out.AsFloat64(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat64(), expected)
	}
}

func TestGatherElements_AllDTypes(t *testing.T) {
	// 1D test
	idx, _ := NewRaw(Shape{3}, Int64, CPU)
	copy(idx.AsInt64(), []int64{2, 0, 1})

	// Int64 data
	dataI64, _ := NewRaw(Shape{3}, Int64, CPU)
	copy(dataI64.AsInt64(), []int64{10, 20, 30})
	outI64, err := GatherElements(dataI64, idx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outI64.AsInt64(), []int64{30, 10, 20}) {
		t.Errorf("Int64 got %v, want [30 10 20]", outI64.AsInt64())
	}

	// Int32 data
	dataI32, _ := NewRaw(Shape{3}, Int32, CPU)
	copy(dataI32.AsInt32(), []int32{10, 20, 30})
	outI32, err := GatherElements(dataI32, idx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outI32.AsInt32(), []int32{30, 10, 20}) {
		t.Errorf("Int32 got %v, want [30 10 20]", outI32.AsInt32())
	}

	// Uint8 data
	dataU8, _ := NewRaw(Shape{3}, Uint8, CPU)
	copy(dataU8.AsUint8(), []uint8{10, 20, 30})
	outU8, err := GatherElements(dataU8, idx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outU8.AsUint8(), []uint8{30, 10, 20}) {
		t.Errorf("Uint8 got %v, want [30 10 20]", outU8.AsUint8())
	}

	// Bool data
	dataBool, _ := NewRaw(Shape{3}, Bool, CPU)
	copy(dataBool.AsBool(), []bool{true, false, true})
	outBool, err := GatherElements(dataBool, idx, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(outBool.AsBool(), []bool{true, true, false}) {
		t.Errorf("Bool got %v, want [true true false]", outBool.AsBool())
	}
}

func TestGatherElements_3D(t *testing.T) {
	// Shape [2, 3, 2]
	data, _ := NewRaw(Shape{2, 3, 2}, Float32, CPU)
	d := data.AsFloat32()
	for i := range d {
		d[i] = float32(i)
	}

	// Gather along axis = 1 with index shape [2, 2, 2]
	idx, _ := NewRaw(Shape{2, 2, 2}, Int64, CPU)
	copy(idx.AsInt64(), []int64{
		2, 1,
		0, 2,

		1, 0,
		2, 1,
	})

	out, err := GatherElements(data, idx, 1)
	if err != nil {
		t.Fatal(err)
	}

	// For p=0:
	// a=0, s=0: in[0][2][0] = 4
	// a=0, s=1: in[0][1][1] = 3
	// a=1, s=0: in[0][0][0] = 0
	// a=1, s=1: in[0][2][1] = 5
	// For p=1:
	// a=0, s=0: in[1][1][0] = 8
	// a=0, s=1: in[1][0][1] = 7
	// a=1, s=0: in[1][2][0] = 10
	// a=1, s=1: in[1][1][1] = 9
	expected := []float32{4, 3, 0, 5, 8, 7, 10, 9}
	if !reflect.DeepEqual(out.AsFloat32(), expected) {
		t.Errorf("got %v, want %v", out.AsFloat32(), expected)
	}
}

func TestGatherElements_Errors(t *testing.T) {
	data, _ := NewRaw(Shape{2, 2}, Float32, CPU)
	idx, _ := NewRaw(Shape{2, 2}, Int64, CPU)

	if _, err := GatherElements(nil, idx, 0); err == nil {
		t.Error("expected error on nil data")
	}
	if _, err := GatherElements(data, nil, 0); err == nil {
		t.Error("expected error on nil idx")
	}

	// Rank mismatch
	idx1D, _ := NewRaw(Shape{2}, Int64, CPU)
	if _, err := GatherElements(data, idx1D, 0); err == nil {
		t.Error("expected error on rank mismatch")
	}

	// Axis out of range
	if _, err := GatherElements(data, idx, 2); err == nil {
		t.Error("expected error on axis >= rank")
	}
	if _, err := GatherElements(data, idx, -3); err == nil {
		t.Error("expected error on axis < -rank")
	}

	// Index out of bounds
	copy(idx.AsInt64(), []int64{0, 2, 0, 0})
	if _, err := GatherElements(data, idx, 0); err == nil {
		t.Error("expected error on out of bounds index")
	}

	// Non-axis dim of indices exceeding data
	idxBig, _ := NewRaw(Shape{3, 2}, Int64, CPU)
	if _, err := GatherElements(data, idxBig, 1); err == nil {
		t.Error("expected error when non-axis dimension exceeds data dimension")
	}
}

// TopK Tests

func TestTopK_1D_Largest(t *testing.T) {
	x, _ := NewRaw(Shape{5}, Float32, CPU)
	copy(x.AsFloat32(), []float32{1.0, 5.0, 3.0, 2.0, 4.0})

	vals, idxs, err := TopK(x, 3, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}

	expectedVals := []float32{5.0, 4.0, 3.0}
	expectedIdxs := []int64{1, 4, 2}

	if !reflect.DeepEqual(vals.AsFloat32(), expectedVals) {
		t.Errorf("vals got %v, want %v", vals.AsFloat32(), expectedVals)
	}
	if !reflect.DeepEqual(idxs.AsInt64(), expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs.AsInt64(), expectedIdxs)
	}
}

func TestTopK_1D_Smallest(t *testing.T) {
	x, _ := NewRaw(Shape{5}, Float32, CPU)
	copy(x.AsFloat32(), []float32{1.0, 5.0, 3.0, 2.0, 4.0})

	vals, idxs, err := TopK(x, 3, 0, false, true)
	if err != nil {
		t.Fatal(err)
	}

	expectedVals := []float32{1.0, 2.0, 3.0}
	expectedIdxs := []int64{0, 3, 2}

	if !reflect.DeepEqual(vals.AsFloat32(), expectedVals) {
		t.Errorf("vals got %v, want %v", vals.AsFloat32(), expectedVals)
	}
	if !reflect.DeepEqual(idxs.AsInt64(), expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs.AsInt64(), expectedIdxs)
	}
}

func TestTopK_2D_Axis1(t *testing.T) {
	// [[0, 10, 20, 30], [30, 20, 10, 0]]
	x, _ := NewRaw(Shape{2, 4}, Float32, CPU)
	copy(x.AsFloat32(), []float32{0, 10, 20, 30, 30, 20, 10, 0})

	vals, idxs, err := TopK(x, 2, 1, true, true)
	if err != nil {
		t.Fatal(err)
	}

	expectedVals := []float32{30, 20, 30, 20}
	expectedIdxs := []int64{3, 2, 0, 1}

	if !reflect.DeepEqual(vals.AsFloat32(), expectedVals) {
		t.Errorf("vals got %v, want %v", vals.AsFloat32(), expectedVals)
	}
	if !reflect.DeepEqual(idxs.AsInt64(), expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs.AsInt64(), expectedIdxs)
	}
}

func TestTopK_2D_Axis0(t *testing.T) {
	// [[0, 10],
	//  [30, 20],
	//  [15, 25]]
	x, _ := NewRaw(Shape{3, 2}, Float32, CPU)
	copy(x.AsFloat32(), []float32{0, 10, 30, 20, 15, 25})

	vals, idxs, err := TopK(x, 2, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}

	expectedVals := []float32{30, 25, 15, 20}
	expectedIdxs := []int64{1, 2, 2, 1}

	if !reflect.DeepEqual(vals.AsFloat32(), expectedVals) {
		t.Errorf("vals got %v, want %v", vals.AsFloat32(), expectedVals)
	}
	if !reflect.DeepEqual(idxs.AsInt64(), expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs.AsInt64(), expectedIdxs)
	}
}

func TestTopK_TiebreakerStability(t *testing.T) {
	// Same values: [3, 5, 5, 2]
	x, _ := NewRaw(Shape{4}, Float32, CPU)
	copy(x.AsFloat32(), []float32{3, 5, 5, 2})

	vals, idxs, err := TopK(x, 3, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}

	expectedVals := []float32{5, 5, 3}
	expectedIdxs := []int64{1, 2, 0}

	if !reflect.DeepEqual(vals.AsFloat32(), expectedVals) {
		t.Errorf("vals got %v, want %v", vals.AsFloat32(), expectedVals)
	}
	if !reflect.DeepEqual(idxs.AsInt64(), expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs.AsInt64(), expectedIdxs)
	}

	// Same values with smallest
	xSmall, _ := NewRaw(Shape{4}, Float32, CPU)
	copy(xSmall.AsFloat32(), []float32{5, 2, 2, 4})
	valsSmall, idxsSmall, err := TopK(xSmall, 2, 0, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(valsSmall.AsFloat32(), []float32{2, 2}) {
		t.Errorf("vals got %v, want [2 2]", valsSmall.AsFloat32())
	}
	if !reflect.DeepEqual(idxsSmall.AsInt64(), []int64{1, 2}) {
		t.Errorf("idxs got %v, want [1 2]", idxsSmall.AsInt64())
	}
}

func TestTopK_AllDTypes(t *testing.T) {
	// Float64
	xf64, _ := NewRaw(Shape{4}, Float64, CPU)
	copy(xf64.AsFloat64(), []float64{10.5, 40.2, 20.1, 30.7})
	vf64, if64, err := TopK(xf64, 2, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(vf64.AsFloat64(), []float64{40.2, 30.7}) || !reflect.DeepEqual(if64.AsInt64(), []int64{1, 3}) {
		t.Errorf("Float64 failed")
	}

	// Int64
	xi64, _ := NewRaw(Shape{4}, Int64, CPU)
	copy(xi64.AsInt64(), []int64{10, 40, 20, 30})
	vi64, ii64, err := TopK(xi64, 2, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(vi64.AsInt64(), []int64{40, 30}) || !reflect.DeepEqual(ii64.AsInt64(), []int64{1, 3}) {
		t.Errorf("Int64 failed")
	}

	// Int32
	xi32, _ := NewRaw(Shape{4}, Int32, CPU)
	copy(xi32.AsInt32(), []int32{10, 40, 20, 30})
	vi32, ii32, err := TopK(xi32, 2, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(vi32.AsInt32(), []int32{40, 30}) || !reflect.DeepEqual(ii32.AsInt64(), []int64{1, 3}) {
		t.Errorf("Int32 failed")
	}

	// Uint8
	xu8, _ := NewRaw(Shape{4}, Uint8, CPU)
	copy(xu8.AsUint8(), []uint8{10, 40, 20, 30})
	vu8, iu8, err := TopK(xu8, 2, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(vu8.AsUint8(), []uint8{40, 30}) || !reflect.DeepEqual(iu8.AsInt64(), []int64{1, 3}) {
		t.Errorf("Uint8 failed")
	}
}

func TestTopK_Errors(t *testing.T) {
	x, _ := NewRaw(Shape{3, 4}, Float32, CPU)

	if _, _, err := TopK(nil, 2, 0, true, true); err == nil {
		t.Error("expected error on nil x")
	}
	if _, _, err := TopK(x, 0, 0, true, true); err == nil {
		t.Error("expected error on k <= 0")
	}
	if _, _, err := TopK(x, 5, 0, true, true); err == nil {
		t.Error("expected error on k > axisDim")
	}
	if _, _, err := TopK(x, 2, 2, true, true); err == nil {
		t.Error("expected error on axis out of range")
	}
}
