#include "CUDABoard.h"

namespace{
  __global__
  void initBoard(BoardPoint *boardDevice){
   
    int index = threadIdx.y * boardSize + threadIdx.x;
  
    if (threadIdx.x == 0 || threadIdx.x == boardSize-1 || threadIdx.y == 0 || threadIdx.y == boardSize-1){
      boardDevice[index].color = 3;
    } else {
      boardDevice[index].color = 0;
    }
  
    //all the initial group ID will be zero..
  
  }
  
  __device__
  
  inline void updateLiberty(BoardPoint *boardDevice, int index, int *globalLiberty){
     if (boardDevice[index].color == GO_EMPTY){
  
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
  }

  __global__
  void playBoard(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int row, int col, int color){
    dim3 threadShape( boardSize, boardSize );
    int numberOfBlock = 1;
    playBoardInside<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, row, col, color);
   
  }
  
  __global__
  void playBoardInside(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int row, int col, int color){
    int index = threadIdx.y*boardSize + threadIdx.x;
    int playPoint = row*boardSize + col;
  
    __shared__ int globalLiberty[totalSize]; // shared array to count the liberty of each group.
    __shared__ int targetGroupID[4] ;
    __shared__ bool hasStoneRemoved;
  
  
    if (threadIdx.y == 0 || threadIdx.y == boardSize || threadIdx.x == 0 || threadIdx.x == boardSize){
      globalLiberty[0] = 0;
      return;
    }
  
  
    if (index == playPoint){
        boardDevice[index].color = color;
        boardDevice[index].groupID = index;
  
        if (boardDevice[index+1].color == color){
          targetGroupID[0] = boardDevice[index+1].groupID;
        }else{
          targetGroupID[0] = -1;
        }
  
        if (boardDevice[index-1].color == color){
          targetGroupID[1] = boardDevice[index-1].groupID;
        }else{
          targetGroupID[1] = -1;
        }
        
        if (boardDevice[index+boardSize].color == color){
          targetGroupID[2] = boardDevice[index+boardSize].groupID;
        }else{
          targetGroupID[2] = -1;
        }
  
        if (boardDevice[index-boardSize].color == color){
          targetGroupID[3] = boardDevice[index-boardSize].groupID;
        }else{
          targetGroupID[3] = -1;
        }
  
    }
  
    __syncthreads();
  
    //@todo , check whether this fence is necessory.
    __threadfence_block();
  
  
    if (boardDevice[index].groupID == targetGroupID[0] ||
        boardDevice[index].groupID == targetGroupID[1] ||
        boardDevice[index].groupID == targetGroupID[2] ||
        boardDevice[index].groupID == targetGroupID[3] ){
      boardDevice[index].groupID = playPoint;
    }
  
    globalLiberty[index] = 0;
    hasStoneRemoved = false;
  
    __syncthreads();
    __threadfence_block();
  
    updateLiberty(boardDevice, index, globalLiberty);
  
    __syncthreads();
    __threadfence_block();
  
    int libertyNumber = globalLiberty[boardDevice[index].groupID];
    if ( libertyNumber == 0 && boardDevice[index].groupID != playPoint){
      boardDevice[index].color = GO_EMPTY;
      boardDevice[index].groupID = 0;
      boardDevice[index].libertyNumber = 0;
      hasStoneRemoved = true;
    } else {
      boardDevice[index].libertyNumber = libertyNumber;
    }
  
    __syncthreads();
    __threadfence_block();
  
    if (hasStoneRemoved){
    
      globalLiberty[index] = 0;
    
      __syncthreads();
      __threadfence_block();
    
      updateLiberty(boardDevice, index, globalLiberty);
    
      __syncthreads();
      __threadfence_block();
    
      libertyNumber = globalLiberty[boardDevice[index].groupID];
      boardDevice[index].libertyNumber = libertyNumber;
    }
  
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
  
  __device__
  inline int inverseColor(int color){
    if (color == GO_BLACK){
      return GO_WHITE;
    }else if(color == GO_WHITE){
      return GO_BLACK;
    }
    return GO_EMPTY;
  }
  
  __global__
  void updateLegleMove(BoardPoint *boardDevice, DebugFlag *debugFlagDevice, int color){
    int index = threadIdx.y*boardSize + threadIdx.x;
  
    if (boardDevice[index].color == GO_EMPTY){
      int totalLiberty = 0;
      
      if (boardDevice[index - 1].color == color){
        totalLiberty = totalLiberty + boardDevice[index - 1].libertyNumber - 1;
      }else if(boardDevice[index - 1].color == GO_EMPTY){
        totalLiberty++;
      }
  
      if (boardDevice[index + 1].color == color){
        totalLiberty = totalLiberty + boardDevice[index + 1].libertyNumber - 1;
      }else if(boardDevice[index + 1].color == GO_EMPTY){
        totalLiberty++;
      }
  
      if (boardDevice[index - boardSize].color == color){
        totalLiberty = totalLiberty + boardDevice[index - boardSize].libertyNumber - 1;
      }else if(boardDevice[index - boardSize].color == GO_EMPTY){
        totalLiberty++;
      }
  
      if (boardDevice[index + boardSize].color == color){
        totalLiberty = totalLiberty + boardDevice[index + boardSize].libertyNumber - 1;
      }else if(boardDevice[index + boardSize].color == GO_EMPTY){
        totalLiberty++;
      }
  
      debugFlagDevice[index].libertyCount = totalLiberty;
  
      if (totalLiberty > 0){
        if (color == GO_BLACK){
          boardDevice[index].isBlackLegal = true;
        }else if (color == GO_WHITE){
          boardDevice[index].isWhiteLegal = true;
        }
      }else{
        if (color == GO_BLACK){
          boardDevice[index].isBlackLegal = false;
        }else if (color == GO_WHITE){
          boardDevice[index].isWhiteLegal = false;
        }
   
      }
      
    } else {
        if (color == GO_BLACK){
          boardDevice[index].isBlackLegal = false;
        }else if (color == GO_WHITE){
          boardDevice[index].isWhiteLegal = false;
        }
    }
       
  }
   
}


CUDABoard::CUDABoard(){
  cudaMalloc( (void**)&(this->boardDevice), this->valueSizeDevice );
  cudaMalloc( (void**)&(this->debugFlagDevice), this->debugFlagSize );

  dim3 threadShape( boardSize, boardSize );
  int numberOfBlock = 1;

  initBoard<<<numberOfBlock, threadShape>>>(boardDevice);
 
}

CUDABoard::~CUDABoard(){
  cudaFree( boardDevice );
  cudaFree( debugFlagDevice );
   
}

void CUDABoard::Play(int row, int col, GoColor color){
//    GoPoint targetPoint = GoPointUtil::Pt(col, row);
//    Play(targetPoint, color);
  //dim3 threadShape( boardSize, boardSize  );
  int threadShape = 1;
  int numberOfBlock = 1;
  playBoard<<<numberOfBlock, threadShape>>>(this->boardDevice, this->debugFlagDevice, row, col, color);
 

}

void CUDABoard::Play(GoPoint p, GoColor color){

}

void CUDABoard::Play(GoPoint p){
  
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
