//go:build !wasm

package operators

import (
	"reflect"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

func TestRegistry_GatherElementsAndTopK(t *testing.T) {
	r := NewRegistry()

	if _, ok := r.Get("GatherElements"); !ok {
		t.Error("Expected GatherElements operator to be registered")
	}
	if _, ok := r.Get("TopK"); !ok {
		t.Error("Expected TopK operator to be registered")
	}
}

func TestHandleGatherElements(t *testing.T) {
	// data: [[1, 2], [3, 4]]
	data, err := tensor.NewRaw(tensor.Shape{2, 2}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(data.AsFloat32(), []float32{1, 2, 3, 4})

	// indices: [[0, 0], [1, 0]]
	idx, err := tensor.NewRaw(tensor.Shape{2, 2}, tensor.Int64, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(idx.AsInt64(), []int64{0, 0, 1, 0})

	node := &Node{
		OpType: "GatherElements",
		Attributes: []Attribute{
			{Name: "axis", I: 1},
		},
	}

	res, err := handleGatherElements(nil, node, []*tensor.RawTensor{data, idx})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 output, got %d", len(res))
	}

	expected := []float32{1, 1, 4, 3}
	if !reflect.DeepEqual(res[0].AsFloat32(), expected) {
		t.Errorf("got %v, want %v", res[0].AsFloat32(), expected)
	}
}

func TestHandleTopK_InputK(t *testing.T) {
	x, _ := tensor.NewRaw(tensor.Shape{2, 4}, tensor.Float32, tensor.CPU)
	copy(x.AsFloat32(), []float32{0, 10, 20, 30, 30, 20, 10, 0})

	kTensor, _ := tensor.NewRaw(tensor.Shape{1}, tensor.Int64, tensor.CPU)
	kTensor.AsInt64()[0] = 2

	node := &Node{
		OpType: "TopK",
		Attributes: []Attribute{
			{Name: "axis", I: 1},
			{Name: "largest", I: 1},
			{Name: "sorted", I: 1},
		},
	}

	res, err := handleTopK(nil, node, []*tensor.RawTensor{x, kTensor})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(res))
	}

	vals := res[0].AsFloat32()
	idxs := res[1].AsInt64()

	expectedVals := []float32{30, 20, 30, 20}
	expectedIdxs := []int64{3, 2, 0, 1}

	if !reflect.DeepEqual(vals, expectedVals) {
		t.Errorf("vals got %v, want %v", vals, expectedVals)
	}
	if !reflect.DeepEqual(idxs, expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs, expectedIdxs)
	}
}

func TestHandleTopK_AttributeK(t *testing.T) {
	x, _ := tensor.NewRaw(tensor.Shape{5}, tensor.Float32, tensor.CPU)
	copy(x.AsFloat32(), []float32{1, 5, 3, 2, 4})

	node := &Node{
		OpType: "TopK",
		Attributes: []Attribute{
			{Name: "k", I: 3},
			{Name: "axis", I: 0},
			{Name: "largest", I: 0}, // smallest
			{Name: "sorted", I: 1},
		},
	}

	res, err := handleTopK(nil, node, []*tensor.RawTensor{x})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(res))
	}

	vals := res[0].AsFloat32()
	idxs := res[1].AsInt64()

	expectedVals := []float32{1, 2, 3}
	expectedIdxs := []int64{0, 3, 2}

	if !reflect.DeepEqual(vals, expectedVals) {
		t.Errorf("vals got %v, want %v", vals, expectedVals)
	}
	if !reflect.DeepEqual(idxs, expectedIdxs) {
		t.Errorf("idxs got %v, want %v", idxs, expectedIdxs)
	}
}
