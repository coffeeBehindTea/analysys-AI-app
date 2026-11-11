# Week 1 Engineering Retrospective

#### File Structure

The business content is in `app` folders, while the rest are in folders such as test scripts and documents in `app` directories.

```text
├app
 └─middleware
 └─routers
 └─schemas
 └─services
 └─utils
├docs
├samples
├tests
```

### Structure note

#### Network services

Network services are provided by `fastAPI` and `uvicorn`, which initiate services and provide network interfaces.

#### Data Contract

The `pydantic` library sets data structure specifications. The error format from the client's request format, LLM's output format to the thrown is specified. And it provides data validation functionality, which automatically throws errors when the data received does not meet the requirements.

#### Middle

The “package” of Middleware is the entire request, which can be processed as soon as the request arrives, and the content is then transferred internally by calling `await call_next(request)`. After Call_next, “post-treatment” can be performed and then removed from the current middleware range. If there are more than one middleware, it would be “last registered first” as if the later registered middleware company had carried the previous middleware. Middleware is responsible for tasks such as identification prior to the actual business and can return errors before arriving at Router.

#### router

The decorator before the router function shows the grouping in which it is placed, the examples of the return result and what requests will be received. The router layer, which is similar to the front desk, is not responsible for the actual business content, but instead directs the different requests to the business function in a manner consistent with business norms.

#### services

Servicees contains real business files such as creating a LLM client, formatting hints, etc.

#### utils

Tools and supporting scripts are available in the utils folder. In this project, for example, there is a definition of uniform error mapping and a script read by Handler env.

### Harder.

The entire architecture is still well designed and understood, but most of the libraries involved in it have not been reached. Even though it is clear what this step is, there is a feeling that the code will be “know why it did it, but not why”. It is therefore much more difficult to understand the code than expected. So I set up strict AI behavior:
I was guided by AI to do it manually (even if the code is created by AI, I create the file manually), and each code contains detailed notes, which are explained in detail when given. It is not just the code of interpretation, but the library and concept involved are to be explained together. After answering my questions and passing the test in every small part, we move forward.
