#include "CUDABoard.h"


namespace{
  __global__
  void initBoard(BoardPoint *boardDevice, curandState *state, long randSeed){
   
    int index = threadIdx.y * boardSize + threadIdx.x;

    curand_init(randSeed, index, 0, &state[index]);
  
    if (threadIdx.x == 0 || threadIdx.x == boardSize-1 || threadIdx.y == 0 || threadIdx.y == boardSize-1){
      boardDevice[index].color = 3;
    } else {
      boardDevice[index].color = 0;
    }
  
    //all the initial group ID will be zero..
  
  }
  __device__
  inline int invertColor(int color){
    if (color == GO_BLACK){
      return GO_WHITE;
    }else if(color == GO_WHITE){
      return GO_BLACK;
    }
    return GO_EMPTY;
  }
 

  __device__
  inline int generateRandomValue(int index, curandState *state){
    return curand(&state[index])>>3; // move left by 3 bit to make sure that it will not be negative after assigned to int.
  }
  
  __device__
  inline void updateStatus(BoardPoint *boardDevice, 
                          int index, 
                          GoColor color, 
                          int *globalLiberty, 
                          int *globalMoveValue, 
                          curandState *state){
     if (boardDevice[index].color == GO_EMPTY){
      // updating liberty for each point 
      atomicAdd(&globalLiberty[boardDevice[index-1].groupID], 1);
   
      if (boardDevice[index+boardSize].groupID != boardDevice[index-1].groupID){
        atomicAdd(&globalLiberty[boardDevice[index+boardSize].groupID], 1);

      } 
  
      if (boardDevice[index+1].groupID != boardDevice[index-1].groupID &&
          boardDevice[index+1].groupID != boardDevice[index+boardSize].groupID){
        atomicAdd(&globalLiberty[boardDevice[index+1].groupID], 1);

       } 
  
      if (boardDevice[index-boardSize].groupID != boardDevice[index-1].groupID &&
          boardDevice[index-boardSize].groupID != boardDevice[index+1].groupID &&
          boardDevice[index-boardSize].groupID != boardDevice[index+boardSize].groupID){
        atomicAdd(&globalLiberty[boardDevice[index-boardSize].groupID], 1);

       } 

     }
    
    __syncthreads();
    __threadfence_block();
  
    int libertyNumber = globalLiberty[boardDevice[index].groupID];
    boardDevice[index].libertyNumber = libertyNumber;
     
    __syncthreads();
    __threadfence_block();

    // computing move value for each point

    if (boardDevice[index].color == GO_EMPTY){
      if (color == GO_WHITE){
        //assuming that next move will be black, as current move is white.
        if( (boardDevice[index - 1].color == GO_WHITE && boardDevice[index-1].libertyNumber == 1) ||
            (boardDevice[index + 1].color == GO_WHITE && boardDevice[index+1].libertyNumber == 1) ||
            (boardDevice[index - boardSize].color == GO_WHITE && boardDevice[index-boardSize].libertyNumber == 1) ||
            (boardDevice[index + boardSize].color == GO_WHITE && boardDevice[index+boardSize].libertyNumber == 1)){
          globalMoveValue[index] = generateRandomValue(index, state);
        }else {
          if (boardDevice[index - 1].color == GO_EMPTY ||
              (boardDevice[index - 1].color == GO_BLACK && boardDevice[index-1].libertyNumber > 1)||
              boardDevice[index + 1].color == GO_EMPTY ||
              (boardDevice[index + 1].color == GO_BLACK && boardDevice[index+1].libertyNumber > 1)||
              boardDevice[index - boardSize].color == GO_EMPTY ||
              (boardDevice[index - boardSize].color == GO_BLACK && boardDevice[index-boardSize].libertyNumber > 1)||
              boardDevice[index + boardSize].color == GO_EMPTY ||
              (boardDevice[index + boardSize].color == GO_BLACK && boardDevice[index+boardSize].libertyNumber > 1)){
            globalMoveValue[index] = generateRandomValue(index, state);
          }else{
            globalMoveValue[index] = -1;
          }
        }
      }else if (color == GO_BLACK){
        //assuming that next move will be white, as current move is black.
        if( (boardDevice[index - 1].color == GO_BLACK && boardDevice[index-1].libertyNumber == 1) ||
            (boardDevice[index + 1].color == GO_BLACK && boardDevice[index+1].libertyNumber == 1) ||
            (boardDevice[index - boardSize].color == GO_BLACK && boardDevice[index-boardSize].libertyNumber == 1) ||
            (boardDevice[index + boardSize].color == GO_BLACK && boardDevice[index+boardSize].libertyNumber == 1)){
          globalMoveValue[index] = generateRandomValue(index, state);
        }else {
         if (boardDevice[index - 1].color == GO_EMPTY ||
            (boardDevice[index - 1].color == GO_WHITE && boardDevice[index-1].libertyNumber > 1)||
            boardDevice[index + 1].color == GO_EMPTY ||
            (boardDevice[index + 1].color == GO_WHITE && boardDevice[index+1].libertyNumber > 1)||
            boardDevice[index - boardSize].color == GO_EMPTY ||
            (boardDevice[index - boardSize].color == GO_WHITE && boardDevice[index-boardSize].libertyNumber > 1)||
            boardDevice[index + boardSize].color == GO_EMPTY ||
            (boardDevice[index + boardSize].color == GO_WHITE && boardDevice[index+boardSize].libertyNumber > 1)){
            globalMoveValue[index] = generateRandomValue(index, state);
          }else{
            globalMoveValue[index] = -1;
          }
        }
      }
    }else{
       // current point is not empty, it is ilegal move, set the value to zero.
       globalMoveValue[index] = -1;
     }

    __syncthreads();
    __threadfence_block();
 
   boardDevice[index].moveValue = globalMoveValue[index];
 
  }

  __device__
  void playStone(BoardPoint *boardDevice, 
                DebugFlag *debugFlagDevice, 
                int *selectedMove, 
                GoColor color, 
                int *globalLiberty, 
                int *globalMoveValue, 
                curandState *state){
    int index = threadIdx.y*boardSize + threadIdx.x;
    int playPoint = *selectedMove;
    GoColor enemyColor = invertColor(color);

    __shared__ int targetGroupID[4];
    __shared__ int removedGroupID[4];
    //__shared__ bool hasStoneRemoved;
  
  
    if (threadIdx.y == 0 || threadIdx.y == boardSize || threadIdx.x == 0 || threadIdx.x == boardSize){
      // out of the real board, reset the liberty of Group 0 to 0, then return.
      globalLiberty[0] = 0;
      return;
    }
  
  
    if (index == playPoint){
        boardDevice[index].color = color;
        boardDevice[index].groupID = index;
  
        if (boardDevice[index+1].color == color){
          targetGroupID[0] = boardDevice[index+1].groupID;
        }else if(boardDevice[index + 1].color == enemyColor){
          if (boardDevice[index + 1].libertyNumber == 1){
            removedGroupID[0] = boardDevice[index + 1].groupID;
          }else{
            removedGroupID[0] = -1;
          }
        }
        else{
          targetGroupID[0] = -1;
          removedGroupID[0] = -1;
        }
  
        if (boardDevice[index-1].color == color){
          targetGroupID[1] = boardDevice[index-1].groupID;
        }else if(boardDevice[index - 1].color == enemyColor){
          if (boardDevice[index - 1].libertyNumber == 1){
            removedGroupID[1] = boardDevice[index - 1].groupID;
          }else{
            removedGroupID[1] = -1;
          }
        }
        else{
          targetGroupID[1] = -1;
          removedGroupID[1] = -1;
        }
        
        if (boardDevice[index+boardSize].color == color){
          targetGroupID[2] = boardDevice[index+boardSize].groupID;
        }else if(boardDevice[index + boardSize].color == enemyColor){
          if (boardDevice[index + boardSize].libertyNumber == 1){
            removedGroupID[0] = boardDevice[index + boardSize].groupID;
          }else{
            removedGroupID[2] = -1;
          }
        }
        else{
          targetGroupID[2] = -1;
          removedGroupID[2] = -1;
        }
  
        if (boardDevice[index-boardSize].color == color){
          targetGroupID[3] = boardDevice[index-boardSize].groupID;
        }else if(boardDevice[index - boardSize].color == enemyColor){
          if (boardDevice[index - boardSize].libertyNumber == 1){
            removedGroupID[0] = boardDevice[index - boardSize].groupID;
          }else{
            removedGroupID[3] = -1;
          }
        }
        else{
          targetGroupID[3] = -1;
          removedGroupID[3] = -1;
        }
  
    }
  
    globalLiberty[index] = 0;
    //hasStoneRemoved = false;
 
    __syncthreads();
  
    //@todo , check whether this fence is necessory.
    __threadfence_block();
  
  
    if (boardDevice[index].groupID == targetGroupID[0] ||
        boardDevice[index].groupID == targetGroupID[1] ||
        boardDevice[index].groupID == targetGroupID[2] ||
        boardDevice[index].groupID == targetGroupID[3] ){
      boardDevice[index].groupID = playPoint;
    }

   if (boardDevice[index].groupID == removedGroupID[0] ||
        boardDevice[index].groupID == removedGroupID[1] ||
        boardDevice[index].groupID == removedGroupID[2] ||
        boardDevice[index].groupID == removedGroupID[3] ){
      boardDevice[index].groupID = 0;
      boardDevice[index].color = GO_EMPTY;
      //hasStoneRemoved = true;
    }
   
 
    __syncthreads();
    __threadfence_block();
  
    updateStatus(boardDevice, index, color, globalLiberty, globalMoveValue, state);
  //
  //
  //
  //  if (boardDevice[index].pointGroup != NULL){
  //    debugFlagDevice[index].changeFlag = boardDevice[index].pointGroup.numberOfLiberty; 
  //    
  //  }
  //
  //
  //    debugFlagDevice[index].counter++;
  //  }
  //  
  
  }

__device__ void selectMove(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, GoColor color, int *globalMoveValue, int *selectedMove){
  if (threadIdx.x == 0 && threadIdx.y == 0){
    int maxValue = -1;
    int maxIndex = 0;

    for (int i=0; i<totalSize; i++){
        if (globalMoveValue[i] > maxValue){
          maxValue = globalMoveValue[i];
          maxIndex = i;
        }
    }

    *selectedMove = maxIndex;
  }

}

  __global__
    void randomPlay(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, GoColor color, curandState *state){
      int index = threadIdx.y*boardSize + threadIdx.x;

      __shared__ int globalLiberty[totalSize];
      __shared__ int globalMoveValue[totalSize];
      __shared__ int selectedMove;

      GoColor currentColor = invertColor(color);

      updateStatus(boardDevice, index, currentColor, globalLiberty, globalMoveValue, state);

      __syncthreads();
      __threadfence_block();
  
      selectMove(boardDevice, debugFlagDevice, currentColor, globalMoveValue, &selectedMove);

//#pragma unroll
      for (int i=0; i<500; i++){
 
        __syncthreads();
        __threadfence_block();
  
        if (selectedMove < 0){
          break;
        }

        playStone(boardDevice, debugFlagDevice, &selectedMove, currentColor, globalLiberty, globalMoveValue, state);
 
        currentColor = invertColor(currentColor);

        __syncthreads();
        __threadfence_block();
  
        selectMove(boardDevice, debugFlagDevice, currentColor, globalMoveValue, &selectedMove);
 

      }

    } 
 
  __global__
  void playBoard(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int row, int col, GoColor color, curandState *state){

    __shared__ int selectedMove;
    __shared__ int globalLiberty[totalSize]; // shared array to count the liberty of each group.
    __shared__ int globalMoveValue[totalSize]; 
 
    if (threadIdx.x == 0 && threadIdx.y ==0){
      // the corner point is special point for global operation.
        int playPoint = row*boardSize + col;
        selectedMove = playPoint;
    }

    __syncthreads();
    __threadfence_block();
    
    playStone(boardDevice, debugFlagDevice, &selectedMove, color, globalLiberty, globalMoveValue, state);
 
  
  }

//  __global__
//  void playBoard(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int row, int col, int color){
//    dim3 threadShape( boardSize, boardSize );
//    int numberOfBlock = 1;
//    playBoardInside<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, row, col, color);
//   
//  }
//   
 
//  __global__
//  void updateLegleMove(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int color){
//    int index = threadIdx.y*boardSize + threadIdx.x;
//  
//    if (boardDevice[index].color != GO_EMPTY){
//      boardDevice[index].isBlackLegal = false;
//      boardDevice[index].isWhiteLegal = false;
//    }else{
//      if (boardDevice[index - 1].color == GO_EMPTY ||
//          boardDevice[index + 1].color == GO_EMPTY ||
//          boardDevice[index - boardSize].color == GO_EMPTY ||
//          boardDevice[index + boardSize].color == GO_EMPTY){
//        boardDevice[index].isBlackLegal = true;
//        boardDevice[index].isWhiteLegal = true;
//
//      }else{
//        int totalLiberty = 0;
//        
//        if (boardDevice[index - 1].color == color){
//          totalLiberty = totalLiberty + boardDevice[index - 1].libertyNumber - 1;
//        }else if(boardDevice[index - 1].color == GO_EMPTY){
//          totalLiberty++;
//        }
//    
//        if (boardDevice[index + 1].color == color){
//          totalLiberty = totalLiberty + boardDevice[index + 1].libertyNumber - 1;
//        }else if(boardDevice[index + 1].color == GO_EMPTY){
//          totalLiberty++;
//        }
//    
//        if (boardDevice[index - boardSize].color == color){
//          totalLiberty = totalLiberty + boardDevice[index - boardSize].libertyNumber - 1;
//        }else if(boardDevice[index - boardSize].color == GO_EMPTY){
//          totalLiberty++;
//        }
//    
//        if (boardDevice[index + boardSize].color == color){
//          totalLiberty = totalLiberty + boardDevice[index + boardSize].libertyNumber - 1;
//        }else if(boardDevice[index + boardSize].color == GO_EMPTY){
//          totalLiberty++;
//        }
//    
//        debugFlagDevice[index].libertyCount = totalLiberty;
//    
//        if (totalLiberty > 0){
//          if (color == GO_BLACK){
//            boardDevice[index].isBlackLegal = true;
//          }else if (color == GO_WHITE){
//            boardDevice[index].isWhiteLegal = true;
//          }
//        }else{
//          if (color == GO_BLACK){
//            boardDevice[index].isBlackLegal = false;
//          }else if (color == GO_WHITE){
//            boardDevice[index].isWhiteLegal = false;
//          }
//     
//        }
//        
//      }       
// 
//   }// any of the points around boardDevice[index] is GO_EMPTY?
// }// boardDevice[index].color == GO_EMPTY?
//   

}//namespace


CUDABoard::CUDABoard(){
  this->currentPlayer = GO_BLACK;
  this->detailDebug = false;


  cudaMalloc( (void**)&(this->boardDevice), this->valueSizeDevice );
  cudaMalloc( (void**)&(this->debugFlagDevice), this->debugFlagSize );

  cudaMalloc( (void**)&(this->stateDevice), valueSizeDevice  );

  dim3 threadShape( boardSize, boardSize );
  int numberOfBlock = 1;

  srand((unsigned int)time(NULL));
  
  initBoard<<<numberOfBlock, threadShape>>>(boardDevice, stateDevice, rand());
 
}

CUDABoard::~CUDABoard(){
  cudaFree( boardDevice );
  cudaFree( debugFlagDevice );
  cudaFree( stateDevice );
   
}

void CUDABoard::Play(int row, int col, GoColor color){
//    GoPoint targetPoint = GoPointUtil::Pt(col, row);
//    Play(targetPoint, color);
  //dim3 threadShape( boardSize, boardSize  );
  dim3 threadShape( boardSize, boardSize );
  int numberOfBlock = 1;
  playBoard<<<numberOfBlock, threadShape>>>(this->boardDevice, this->debugFlagDevice, row, col, color, this->stateDevice);
}

void CUDABoard::Play(GoPoint p, GoColor color){

}

void CUDABoard::Play(GoPoint p){
  
}

void CUDABoard::RandomPlay(){
  dim3 threadShape( boardSize, boardSize );
  int numberOfBlock = 1;
  randomPlay<<<numberOfBlock, threadShape>>>(this->boardDevice, this->debugFlagDevice, this->currentPlayer, this->stateDevice);

  cudaDeviceSynchronize();

}

void CUDABoard::RestoreData(){
  cudaMemcpy( this->boardHost, this->boardDevice, this->valueSizeDevice, cudaMemcpyDeviceToHost );
  cudaMemcpy( this->debugFlagHost, this->debugFlagDevice, this->debugFlagSize, cudaMemcpyDeviceToHost );

  cudaDeviceSynchronize();


}

ostream& operator<<(ostream& out, const CUDABoard& cudaBoard){

  out << "Whole board:" << endl;


  for (int i=boardSize-1; i>=0; i--){
    for (int j=0; j<boardSize; j++){
      int index = i*boardSize + j;
      if (cudaBoard.boardHost[index].color == 0){
        out << ".";
      }else if (cudaBoard.boardHost[index].color == GO_BLACK){
        out << "o";
      }else if (cudaBoard.boardHost[index].color == GO_WHITE){
        out << "x";
      }else if (cudaBoard.boardHost[index].color == GO_BORDER){
        out << "H";
      }

       

    }
    if (cudaBoard.detailDebug){
      out << "     ";
      for (int j=0; j<boardSize; j++){
       int index = i*boardSize + j;
     
       if (cudaBoard.boardHost[index].color == GO_BORDER){
          out << "HHHH";
        }else {
          int value = cudaBoard.boardHost[index].moveValue%1000;
          std::stringstream ss;
          std::string outputString;
          //ss<<"      ";
          ss<< "___" << value;
          ss>>outputString;

          out << outputString.substr(outputString.length()-3);
          out << "|";

        }  
      } 
    }
    out << "\n";
   
  }

  return out;
}





//int main()
//{
//  
//  struct timeval start_tv;
//  gettimeofday(&start_tv,NULL);
//  
//  
// 
////  for (int i=0; i<19; i++){
////    playBoard<<<numberOfBlock, threadShape>>>(boardDevice, globalFlag, i, i, 2);
////  }
//
////  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 15, 12, 1);
//
//  //updateLegleMove<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, GO_BLACK);
//  //updateLegleMove<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, GO_WHITE);
//
//  cudaDeviceSynchronize();
//
//  cudaMemcpy( boardHost, boardDevice, valueSizeDevice, cudaMemcpyDeviceToHost );
//  cudaMemcpy( debugFlagHost, debugFlagDevice, debugFlagSize, cudaMemcpyDeviceToHost );
//
//
// 
//  cudaDeviceSynchronize();
//
//  struct timeval end_tv;
//  gettimeofday(&end_tv,NULL);
// 
//  for (int i=boardSize-1; i>=0; i--){
//    for (int j=0; j<boardSize; j++){
//      int index = i*boardSize + j;
//      if (boardHost[index].color == 0){
//        printf(".");
//      }else if (boardHost[index].color == GO_BLACK){
//        printf("o");
//      }else if (boardHost[index].color == GO_WHITE){
//        printf("x");
//      }else if (boardHost[index].color == GO_BORDER){
//        printf("H");
//      }
//    }
//    printf("\n");
//   
//  }
//
////  for (int i=boardSize-1; i>=0; i--){
////    for (int j=0; j<boardSize; j++){
////      int index = i*boardSize + j;
//////      if (boardHost[index].color == GO_BLACK || boardHost[index].color == GO_WHITE){
////        printf("%d, %d | ", boardHost[index].groupID, boardHost[index].libertyNumber);
//////      } else if (boardHost[index].color == GO_EMPTY) {
//////        printf("   ,   | ");
//////      }
////    }
////    printf("\n");
////   
////  }
//
//  for (int i=boardSize-1; i>=0; i--){
//    for (int j=0; j<boardSize; j++){
//      int index = i*boardSize + j;
//      if (boardHost[index].color == GO_BORDER){
//        printf("H");
//      }else{
//        if (boardHost[index].isBlackLegal){
//          printf("o");
//        }else {
//          printf(".");
//        }
//      }
//    }
//
//    printf("        ");
//
//    for (int j=0; j<boardSize; j++){
//      int index = i*boardSize + j;
//      if (boardHost[index].color == GO_BORDER){
//        printf("H");
//      }else{
//        if (boardHost[index].isWhiteLegal){
//          printf("x");
//        }else {
//          printf(".");
//        }
//      }
//    }
//    
//    printf("\n");
//   
//  }
//
//
//
////  for (int i=boardSize-1; i>=0; i--){
////    for (int j=0; j<boardSize; j++){
////      int index = i*boardSize + j;
////      printf("%d | ", debugFlagHost[index].libertyCount);
////      }
////    printf("\n");
////   
////  }
//
//
//  printf("\n");
//
//  if(end_tv.tv_usec >= start_tv.tv_usec){
//    printf("time %lu:%lu\n",end_tv.tv_sec - start_tv.tv_sec,  end_tv.tv_usec - start_tv.tv_usec);
//  }else{
//    printf("time %lu:%lu\n",end_tv.tv_sec - start_tv.tv_sec - 1,  1000000 - start_tv.tv_usec + end_tv.tv_usec);
//  }
//
//  
//  return EXIT_SUCCESS;
//  
//}
