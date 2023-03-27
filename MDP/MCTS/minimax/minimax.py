function minimax(position, depth, maximizingPlayer)
   if depth == 0 or game over in position
       return static evaluation of position

    if maximizingPlayer
       maxEval = -infinity
        for each child of position
           eval = minimax(child, depth - 1, false)
            maxEval = max(maxEval, eval)
        return maxEval

    else
       minEval = +infinity
        for each child of position
           eval = minimax(child, depth - 1, true)
            minEval = min(minEval, eval)
        return minEval


// initial call
minimax(currentPosition, 3, true)