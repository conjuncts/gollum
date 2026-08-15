I want to *redesign* this batch handler.

The batch handler should retain the 3 central tasks of handling batch arrival, batch submit, and reconnecting entries. However, 

The batch handler should track BatchJob as in one of four states:

1. unrecognized
2. pending
3. finalizing
4. complete
The batch handler should centrally store an event handler type thing which tracks state on a per-BatchJob level.

Upon RECONNECT (an entry): get the corresponding batch job. Then read the BatchJob's status

1. pending: register the entry as listener which will be notified when complete.
2. finalizing: That means we are in the critical section (per this batch job). Hence, have the entry wait until the the critical section is done. (listen for completion event) then should be good to check permacache
3. complete: should be good to check permacache
4. unrecognized (absence of status): this means that the entry should be good to go to be sent in a batch, and eventually, will complete (another piece of code is responsible for this)
5. failure: equivalent to complete, but means that 

Upon ARRIVAL:
1. Handles state transition from pending -> finalizing -> complete.
2. The "finalizing" state helps delineate the critical section (should have locks/some guard to prevent race conditions)

Upon SUBMIT:
1. Handles state transition from unrecognized -> pending

Feel free to criticize this design and suggest improvements. I would much rather know about a design flaw now rather than later down the line.

# 3.

Excellent, that analysis is exactly what I'm looking for. I think it's time to more clearly lay out the BatchJob lifecycle.

BatchJob lifecycle
1. (no state)
    - reconnect(), if arrives at this time, will lead to either permacache check or creation of a new batch
2. pending
    - reconnect(), if arrives at this time, will lead to attachment of callback
3. finalizing
    - reconnect(), if arrives at this time, will lead to attachment of callback
    - This is a critical section with atomicity guarantees
    1. Here, all values are moved to permacache.
    2. (Perhaps) BatchStorage.free_batch is called.
4. completed / failure
    - reconnect(), if arrives at this time, will depend on permacache, which we can assume will have completed
    1. Here callbacks are notified.

submit handles the state transition from (1) -> (2); arrival handles (2) -> (3) -> (4).

Now we can analyze: in which states is reconnect(entry) valid. In (2-3), entry gets attached as callback and is notified, so entry completes. In (4), we subsequently check permacache, which (if the batch succeeded) is guaranteed to have filled because we completed state 3 -> entry completes. If the batch failed, then the entry will pass through (retry), which is a reasonable outcome (right now, failures are not stored anyway - that is a problem I will address down the road.) In (1), the batch job doesn't exist so passthrough is the correct behavior.


Secondly, we need to decide when BatchStorage.retrieve_batch is valid. Based on the analysis, it seems to me that there are two options that are both valid. 
Option 1: BatchStorage.retrieve_batch is only valid in (1-3), and becomes freed just after moving everything to permacache. 
    Pro: free_batch() behavior remains the same.
    Con: risky if timing logic has error.
Option 2: BatchStorage.retrieve_batch is valid in (1-4) and up until program exit. Redesign free_batch() into two methods: complete_batch(), which marks a BatchJob as complete, and free_completed(), which frees all batches marked complete, which is slated to run at the *next* program start (ie. always at program start)
    Pro: more resistant to logic errors and convenience of retrieve_batch always returning the corresponding BatchJob id. Also, we have the ability to see that the entry failed without implementing an error failure storage mechanism
    Con: need to modify implementation of BatchStorage

These both have their pros and cons so I will leave this choice up to you.

Overall, I believe this design is safe, but please come up with suggestions and let me know if I overlooked something. 
