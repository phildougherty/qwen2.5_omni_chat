FROM nginx:alpine

# Copy static files
COPY . /usr/share/nginx/html/

# Copy a custom nginx configuration for the frontend
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Keep the container running
CMD ["nginx", "-g", "daemon off;"]