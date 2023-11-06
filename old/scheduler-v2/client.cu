#include "client.h"

void *Client::threadFunction(void *arg)
{
    Client *client = static_cast<Client *>(arg);
    return client->threadFunction();
}

void *Client::threadFunction()
{
    while (true)
    {
        setupKernel();

        // Send the CUDA kernel to the main thread for execution
        clientManager->requestLaunchKernel(&kernel);

        // Continue running application code inside thread

        finishKernel();
    }

    return nullptr;
}