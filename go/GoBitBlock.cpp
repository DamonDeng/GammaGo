#include "GoBitBlock.h"

GoBitBlock::GoBitBlock(GoColor inputColor){
    color = inputColor;
}

GoBitBlock::~GoBitBlock(){
    
}

void GoBitBlock::AddStone(GoPoint point){
    stones.set(point);
    stoneNumber = stones.size();
}

void GoBitBlock::MergeStones(bitset<TOTAL_SIZE> inputStones){
    stones |= inputStones;
    stoneNumber = stones.size();
}

void GoBitBlock::RemoveStone(GoPoint point){
    stones.reset(point);
    stoneNumber = stones.size();
}

void GoBitBlock::AddLiberty(GoPoint point){
    liberties.set(point);
    libertyNumber = liberties.size();
}

void GoBitBlock::MergeLiberties(bitset<TOTAL_SIZE> inputLiberties){
    liberties |= inputLiberties;
    libertyNumber = liberties.size();
}

void GoBitBlock::RemoveLiberty(GoPoint point){
    liberties.reset(point);
    libertyNumber = liberties.size();
}

void GoBitBlock::MergeBlockFrom(GoBitBlock& sourceBlock){
    stones |= sourceBlock.stones;
    liberties |= sourceBlock.liberties;

    stoneNumber = stones.size();
    libertyNumber = liberties.size();
}

int GoBitBlock::GetStoneNumber(){
    return stoneNumber;
}

int GoBitBlock::GetLibertyNumber(){
    return libertyNumber;
}


