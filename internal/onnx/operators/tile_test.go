//go:build !wasm

package operators

import (
	"reflect"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

func TestHandleTile_Float32_Int64Repeats(t *testing.T) {
	// input: [[1, 2], [3, 4]]
	x, err := tensor.NewRaw(tensor.Shape{2, 2}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	repeats, err := tensor.NewRaw(tensor.Shape{2}, tensor.Int64, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(repeats.AsInt64(), []int64{1, 2})

	node := &Node{
		OpType: "Tile",
	}

	res, err := handleTile(nil, node, []*tensor.RawTensor{x, repeats})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 output, got %d", len(res))
	}

	if !reflect.DeepEqual(res[0].Shape(), tensor.Shape{2, 4}) {
		t.Errorf("got shape %v, want [2 4]", res[0].Shape())
	}

	expected := []float32{1, 2, 1, 2, 3, 4, 3, 4}
	if !reflect.DeepEqual(res[0].AsFloat32(), expected) {
		t.Errorf("got %v, want %v", res[0].AsFloat32(), expected)
	}
}

func TestHandleTile_Float32_Int32Repeats(t *testing.T) {
	// input: [[1, 2], [3, 4]]
	x, err := tensor.NewRaw(tensor.Shape{2, 2}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	repeats, err := tensor.NewRaw(tensor.Shape{2}, tensor.Int32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(repeats.AsInt32(), []int32{2, 1})

	node := &Node{
		OpType: "Tile",
	}

	res, err := handleTile(nil, node, []*tensor.RawTensor{x, repeats})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 output, got %d", len(res))
	}

	if !reflect.DeepEqual(res[0].Shape(), tensor.Shape{4, 2}) {
		t.Errorf("got shape %v, want [4 2]", res[0].Shape())
	}

	expected := []float32{1, 2, 3, 4, 1, 2, 3, 4}
	if !reflect.DeepEqual(res[0].AsFloat32(), expected) {
		t.Errorf("got %v, want %v", res[0].AsFloat32(), expected)
	}
}

func TestHandleTile_3D(t *testing.T) {
	x, err := tensor.NewRaw(tensor.Shape{2, 1, 2}, tensor.Float32, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(x.AsFloat32(), []float32{1, 2, 3, 4})

	repeats, err := tensor.NewRaw(tensor.Shape{3}, tensor.Int64, tensor.CPU)
	if err != nil {
		t.Fatal(err)
	}
	copy(repeats.AsInt64(), []int64{1, 2, 3})

	node := &Node{
		OpType: "Tile",
	}

	res, err := handleTile(nil, node, []*tensor.RawTensor{x, repeats})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 output, got %d", len(res))
	}

	if !reflect.DeepEqual(res[0].Shape(), tensor.Shape{2, 2, 6}) {
		t.Errorf("got shape %v, want [2 2 6]", res[0].Shape())
	}

	expected := []float32{
		1, 2, 1, 2, 1, 2,
		1, 2, 1, 2, 1, 2,
		3, 4, 3, 4, 3, 4,
		3, 4, 3, 4, 3, 4,
	}
	if !reflect.DeepEqual(res[0].AsFloat32(), expected) {
		t.Errorf("got %v, want %v", res[0].AsFloat32(), expected)
	}
}

func TestHandleTile_Errors(t *testing.T) {
	node := &Node{OpType: "Tile"}

	// Wrong number of inputs
	if _, err := handleTile(nil, node, []*tensor.RawTensor{}); err == nil {
		t.Error("expected error for 0 inputs")
	}

	x, _ := tensor.NewRaw(tensor.Shape{2, 2}, tensor.Float32, tensor.CPU)
	if _, err := handleTile(nil, node, []*tensor.RawTensor{x}); err == nil {
		t.Error("expected error for 1 input")
	}

	// Nil inputs
	repeats, _ := tensor.NewRaw(tensor.Shape{2}, tensor.Int64, tensor.CPU)
	if _, err := handleTile(nil, node, []*tensor.RawTensor{nil, repeats}); err == nil {
		t.Error("expected error for nil input tensor")
	}
	if _, err := handleTile(nil, node, []*tensor.RawTensor{x, nil}); err == nil {
		t.Error("expected error for nil repeats tensor")
	}

	// Invalid repeats dtype (Float32)
	badRepeats, _ := tensor.NewRaw(tensor.Shape{2}, tensor.Float32, tensor.CPU)
	if _, err := handleTile(nil, node, []*tensor.RawTensor{x, badRepeats}); err == nil {
		t.Error("expected error for float32 repeats")
	}
}
