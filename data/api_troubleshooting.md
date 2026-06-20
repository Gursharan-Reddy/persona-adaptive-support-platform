# API Authentication & Troubleshooting Guide

### Bearer Token Header Format
All requests to our API endpoints must contain a valid Bearer Token in the HTTP authorization header. The syntax must follow exactly: Authorization: Bearer <your_access_token>.

### Common Errors
- **401 Unauthorized**: Occurs when the token is missing, expired, or malformed.
- **500 Internal Server Error**: Occurs when database query execution pools drop or connection timeouts expire.
