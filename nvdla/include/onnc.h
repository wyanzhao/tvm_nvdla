#ifndef ONNC_H_
#define ONNC_H_

#include <onnc/ADT/StringList.h>
#include <onnc/IR/IRBuilder.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/CodeGen/BuildMemOperand.h>
#include <onnc/CodeGen/SetMemOperand.h>

void setMemOperand(onnc::Module& pModule);
void createMemOperandsOfGraph(onnc::ComputeGraph& pCG);


#endif