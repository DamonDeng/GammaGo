#include <stdio.h>
#include <sys/time.h>
#include <time.h>


#define GO_EMPTY 0
#define GO_BLACK 1
#define GO_WHITE 2
#define GO_BORDER 3

 
const int boardSize = 21; 
const int totalSize = boardSize * boardSize;


struct BoardPoint{
  int color;
  int groupID;
  int libertyNumber;
  bool isBlackLegal;
  bool isWhiteLegal;
};

struct DebugFlag{
  int counter;
  int changeFlag;
  int targetGroupID[4];
  int libertyCount;
};
 
__global__
void initBoard(BoardPoint *boardDevice){
 
  int index = threadIdx.y * boardSize + threadIdx.x;

  if (threadIdx.x == 0 || threadIdx.x == boardSize-1 || threadIdx.y == 0 || threadIdx.y == boardSize-1){
    boardDevice[index].color = 3;
  } else {
    boardDevice[index].color = 0;
  }

  //boardDevice[index].groupID = totalSize; // all the initial group ID was set to none group id.

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
  if ( libertyNumber == 0 ){
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
 
int main()
{
  BoardPoint boardHost[totalSize];
  BoardPoint *boardDevice;
  DebugFlag debugFlagHost[totalSize];
  DebugFlag *debugFlagDevice;

  const int valueSizeDevice = totalSize*sizeof(BoardPoint);
  const int debugFlagSize = totalSize*sizeof(DebugFlag);

  cudaMalloc( (void**)&boardDevice, valueSizeDevice );
  cudaMalloc( (void**)&debugFlagDevice, debugFlagSize );

  
  struct timeval start_tv;
  gettimeofday(&start_tv,NULL);
  
  
  dim3 threadShape( boardSize, boardSize );
  int numberOfBlock = 1;

  initBoard<<<numberOfBlock, threadShape>>>(boardDevice);
  
//  for (int i=0; i<19; i++){
//    playBoard<<<numberOfBlock, threadShape>>>(boardDevice, globalFlag, i, i, 2);
//  }

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 11, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 12, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 11, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 12, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 10, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 9, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 11, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 14, 10, 2);

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 1, 1, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 2, 1, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 1, 2, 1);

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 5, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 5, 11, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 5, 12, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 6, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 7, 10, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 8, 10, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 8, 9, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 8, 11, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 9, 10, 2);

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 19, 19, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 18, 19, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 19, 18, 1);

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 4, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 5, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 10, 6, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 11, 4, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 12, 4, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 4, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 3, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 13, 5, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 14, 4, 2);

  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 17, 16, 2);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 17, 17, 1);
  playBoard<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, 15, 12, 1);

  //updateLegleMove<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, GO_BLACK);
  //updateLegleMove<<<numberOfBlock, threadShape>>>(boardDevice, debugFlagDevice, GO_WHITE);

  cudaDeviceSynchronize();

  cudaMemcpy( boardHost, boardDevice, valueSizeDevice, cudaMemcpyDeviceToHost );
  cudaMemcpy( debugFlagHost, debugFlagDevice, debugFlagSize, cudaMemcpyDeviceToHost );


  cudaFree( boardDevice );
  cudaFree( debugFlagDevice );
  
  cudaDeviceSynchronize();

  struct timeval end_tv;
  gettimeofday(&end_tv,NULL);
 
  for (int i=boardSize-1; i>=0; i--){
    for (int j=0; j<boardSize; j++){
      int index = i*boardSize + j;
      if (boardHost[index].color == 0){
        printf(".");
      }else if (boardHost[index].color == GO_BLACK){
        printf("o");
      }else if (boardHost[index].color == GO_WHITE){
        printf("x");
      }else if (boardHost[index].color == GO_BORDER){
        printf("H");
      }
    }
    printf("\n");
   
  }

//  for (int i=boardSize-1; i>=0; i--){
//    for (int j=0; j<boardSize; j++){
//      int index = i*boardSize + j;
////      if (boardHost[index].color == GO_BLACK || boardHost[index].color == GO_WHITE){
//        printf("%d, %d | ", boardHost[index].groupID, boardHost[index].libertyNumber);
////      } else if (boardHost[index].color == GO_EMPTY) {
////        printf("   ,   | ");
////      }
//    }
//    printf("\n");
//   
//  }

  for (int i=boardSize-1; i>=0; i--){
    for (int j=0; j<boardSize; j++){
      int index = i*boardSize + j;
      if (boardHost[index].color == GO_BORDER){
        printf("H");
      }else{
        if (boardHost[index].isBlackLegal){
          printf("o");
        }else {
          printf(".");
        }
      }
    }

    printf("        ");

    for (int j=0; j<boardSize; j++){
      int index = i*boardSize + j;
      if (boardHost[index].color == GO_BORDER){
        printf("H");
      }else{
        if (boardHost[index].isWhiteLegal){
          printf("x");
        }else {
          printf(".");
        }
      }
    }
    
    printf("\n");
   
  }



//  for (int i=boardSize-1; i>=0; i--){
//    for (int j=0; j<boardSize; j++){
//      int index = i*boardSize + j;
//      printf("%d | ", debugFlagHost[index].libertyCount);
//      }
//    printf("\n");
//   
//  }


  printf("\n");

  if(end_tv.tv_usec >= start_tv.tv_usec){
    printf("time %lu:%lu\n",end_tv.tv_sec - start_tv.tv_sec,  end_tv.tv_usec - start_tv.tv_usec);
  }else{
    printf("time %lu:%lu\n",end_tv.tv_sec - start_tv.tv_sec - 1,  1000000 - start_tv.tv_usec + end_tv.tv_usec);
  }

  
  return EXIT_SUCCESS;
  
}
