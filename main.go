// Tensorflow Serving Go client for the inception model

// Please update proto package, gRPC package and rebuild the proto files:
// go get -u github.com/golang/protobuf/{proto,protoc-gen-go}
// go get -u google.golang.org/grpc

// First of all compile the proto files:
// git clone https://github.com/tensorflow/serving.git
// git clone https://github.com/tensorflow/tensorflow.git
// protoc -I=serving -I tensorflow --go_out=plugins=grpc:$GOPATH/src serving/tensorflow_serving/apis/*.proto
// protoc -I=tensorflow --go_out=plugins=grpc:$GOPATH/src tensorflow/tensorflow/core/framework/*.proto
// protoc -I=tensorflow --go_out=plugins=grpc:$GOPATH/src tensorflow/tensorflow/core/protobuf/{saver,meta_graph}.proto
// protoc -I=tensorflow --go_out=plugins=grpc:$GOPATH/src tensorflow/tensorflow/core/example/*.proto

package main

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	tf_core_framework "tensorflow/core/framework"
	pb "tensorflow_serving/apis"

	google_protobuf "github.com/golang/protobuf/ptypes/wrappers"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"

	"google.golang.org/grpc"
)

func main() {
	servingAddress := flag.String("serving-address", "localhost:9000", "The tensorflow serving address")
	flag.Parse()

	if flag.NArg() != 1 {
		fmt.Println("Usage: " + os.Args[0] + " --serving-address localhost:9000 path/to/img.png")
		os.Exit(1)
	}

	imgPath, err := filepath.Abs(flag.Arg(0))
	if err != nil {
		log.Fatalln(err)
	}

	imageBytes, err := ioutil.ReadFile(imgPath)
	if err != nil {
		log.Fatalln(err)
	}

	tensor, err := tf.NewTensor(string(imageBytes))
	if err != nil {
		log.Fatalln("Cannot read image file")
	}

	tensorString, ok := tensor.Value().(string)
	if !ok {
		log.Fatalln("Cannot type assert tensor value to string")
	}

	request := &pb.PredictRequest{
		ModelSpec: &pb.ModelSpec{
			Name:          "inception",
			SignatureName: "predict_images",
			Version: &google_protobuf.Int64Value{
				Value: int64(1),
			},
		},
		Inputs: map[string]*tf_core_framework.TensorProto{
			"images": &tf_core_framework.TensorProto{
				Dtype: tf_core_framework.DataType_DT_STRING,
				TensorShape: &tf_core_framework.TensorShapeProto{
					Dim: []*tf_core_framework.TensorShapeProto_Dim{
						&tf_core_framework.TensorShapeProto_Dim{
							Size: int64(1),
						},
					},
				},
				StringVal: [][]byte{[]byte(tensorString)},
			},
		},
	}

	conn, err := grpc.Dial(*servingAddress, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Cannot connect to the grpc server: %v\n", err)
	}
	defer conn.Close()

	client := pb.NewPredictionServiceClient(conn)

	resp, err := client.Predict(context.Background(), request)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println(resp)
}
