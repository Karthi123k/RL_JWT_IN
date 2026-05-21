1. first i did with normal iteration exution
2. with 3 times and with std added   # here 500 not done with sphincs  192 and 256 i did with 400 concurrency for that version 
#####
3. with linkerd we need to do it
4. compare the linkerd and with out linkerd results 
5. based on the results we  need to selected the algorithms for rl
6. in the Rl we have to do it Herustic and Q- learn and PPo and MAPPo



# auto scale approach

First install metrics server once only:

kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

Verify:

kubectl top nodes
kubectl top pods -n pqc-jwt

Then create HPA for auth-service:

kubectl autoscale deployment auth-service \
--cpu-percent=70 \
--min=1 \
--max=6 \
-n pqc-jwt

Create HPA for user-service:

kubectl autoscale deployment user-service \
--cpu-percent=70 \
--min=1 \
--max=6 \
-n pqc-jwt

Check both:

kubectl get hpa -n pqc-jwt

Expected output:

NAME            REFERENCE                      TARGETS   MINPODS   MAXPODS
auth-service    Deployment/auth-service       45%/70%   1          6
user-service    Deployment/user-service       30%/70%   1          6


# if i want delete autoscal 
To stop autoscaling, delete the HPA (Horizontal Pod Autoscaler).

For auth-service:

kubectl delete hpa auth-service -n pqc-jwt

For user-service:

kubectl delete hpa user-service -n pqc-jwt

Or delete all HPAs in the namespace:

kubectl delete hpa --all -n pqc-jwt

Verify:

kubectl get hpa -n pqc-jwt

If removed successfully:

No resources found in pqc-jwt namespace

After deleting HPA, Kubernetes stops automatic scaling, but it keeps the current number of pods.

For example:

Before delete:
HPA running
Current pods = 4

After delete:
Autoscaling stopped
Pods remain = 4

If you want to return to a fixed number (for example 1 pod):

kubectl scale deployment auth-service --replicas=1 -n pqc-jwt
kubectl scale deployment user-service --replicas=1 -n pqc-jwt

That gives you a static deployment again.



# for watch  autoscale watching 
Also watch pods in another terminal while the test runs:

watch kubectl get pods -n pqc-jwt

and:

watch kubectl top pods -n pqc-jwt

Expected behavior during load:




# for runtest.sh Seqnence to run commands


 # Start Docker
sudo systemctl start docker

# Start existing k3d cluster
k3d cluster start pqc-cluster

# Verify Kubernetes nodes
kubectl get nodes

# Verify pods
kubectl get pods -n pqc-jwt

# If pods are missing or not running
kubectl apply -f ~/Karthik/RL_JWT/k8s/

# Verify gateway works
curl -X POST http://localhost:8080/login

# Move to benchmark folder
cd ~/Karthik/RL_JWT/locust

# Make script executable (only needed if permissions changed)
chmod +x run_tests.sh

# Run benchmark
./run_tests.sh






# Use this full "from scratch" terminal flow:

# Start Docker
sudo systemctl start docker

# Go to project
cd ~/Karthik/RL_JWT

# Create k3d cluster
k3d cluster create pqc-cluster \
--servers 1 \
--agents 2 \
-p "8080:80@loadbalancer" \
-p "9090:30090@loadbalancer" \
-v "/home/karthik/Karthik/RL_JWT/certs:/mnt/certs@all"

# Build Docker images
docker build -t auth-service:1.0 ./auth-service

docker build -t user-service:1.0 ./user-service

# Import images into k3d
k3d image import auth-service:1.0 -c pqc-cluster

k3d image import user-service:1.0 -c pqc-cluster

# Deploy Kubernetes resources
kubectl apply -f k8s/

# Check nodes
kubectl get nodes

# Check pods
kubectl get pods -n pqc-jwt -w

# Verify gateway
curl -X POST http://localhost:8080/login

# Move to benchmark folder
cd locust

chmod +x run_tests.sh

# Run benchmark
./run_tests.sh

If you want to clean everything completely before recreating:

# Delete cluster
k3d cluster delete pqc-cluster

# Remove local images
docker rmi auth-service:1.0
docker rmi user-service:1.0

# Optional: remove dangling images/build cache
docker system prune -a

Use cases:

Only rebooted machine:

Start Docker
Start cluster
Run tests

Deleted cluster:

Create cluster
Import images
Apply YAML
Run tests

New machine:

Clone repo
Build images
Create cluster
Import images
Apply YAML
Run tests

You don't need to rebuild images after a simple restart, but you do after moving to a fresh system or deleting the cluster/images.


# Linkerd Setup for Current Project

Project architecture:

k3d
↓
k3s
↓
Traefik Ingress
↓
auth-service
↓
user-service

After Linkerd:

Locust
↓
Traefik
↓
┌────────────────────┐
│ Auth Pod           │
│ ├─ auth-service    │
│ └─ linkerd-proxy 🔒│
└────────────────────┘
↓
┌────────────────────┐
│ User Pod           │
│ ├─ user-service    │
│ └─ linkerd-proxy 🔒│
└────────────────────┘

====================================================
STEP 1: Verify Kubernetes cluster
=================================

kubectl get nodes

Expected:

NAME            STATUS
k3s-master      Ready
k3s-worker1     Ready
...

====================================================
STEP 2: Install Linkerd CLI
===========================

Install:

curl -fsL https://run.linkerd.io/install | sh

Add to PATH permanently:

echo 'export PATH=$PATH:$HOME/.linkerd2/bin' >> ~/.bashrc

source ~/.bashrc

Verify:

linkerd version

Expected:

Client version: stable-xx
Server version: unavailable

(Server unavailable is normal)

====================================================
STEP 3: Install Gateway API CRDs
================================

Required for new Linkerd versions.

kubectl apply --server-side -f 
https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml

Verify:

kubectl get crds | grep gateway

Expected:

gatewayclasses.gateway.networking.k8s.io
gateways.gateway.networking.k8s.io
httproutes.gateway.networking.k8s.io

====================================================
STEP 4: Install Linkerd CRDs
============================

linkerd install --crds | kubectl apply -f -
linkerd viz install | kubectl apply -f -
Verify:

kubectl get crds | grep linkerd

Expected:

authorizationpolicies.policy.linkerd.io
servers.policy.linkerd.io
...

====================================================
STEP 5: Install Linkerd control plane
=====================================

linkerd install | kubectl apply -f -

Verify:

kubectl get pods -n linkerd

Expected:

linkerd-controller
linkerd-destination
linkerd-identity
linkerd-proxy-injector

All should be:

Running

====================================================
STEP 6: Verify installation
===========================

linkerd check

Expected:

Status check results are √

====================================================
STEP 7: Enable sidecar injection
================================

Enable namespace:

kubectl annotate namespace pqc-jwt 
linkerd.io/inject=enabled

Verify:

kubectl get namespace pqc-jwt --show-labels

Expected:

linkerd.io/inject=enabled

====================================================
STEP 8: Restart application pods
================================

kubectl rollout restart deployment/auth-service 
-n pqc-jwt

kubectl rollout restart deployment/user-service 
-n pqc-jwt

kubectl rollout restart deployment/prometheus 
-n pqc-jwt

Wait:

kubectl rollout status deployment/auth-service 
-n pqc-jwt

kubectl rollout status deployment/user-service 
-n pqc-jwt

====================================================
STEP 9: Verify sidecars
=======================

kubectl get pods -n pqc-jwt

Before:

auth-service-xxx  1/1 Running
user-service-xxx  1/1 Running

After:

auth-service-xxx  2/2 Running
user-service-xxx  2/2 Running

2/2 means:

auth-service
+
linkerd-proxy

====================================================
STEP 10: Verify containers
==========================

kubectl describe pod <pod-name> 
-n pqc-jwt

Expected:

Containers:

auth-service
linkerd-proxy

====================================================
STEP 11: Install dashboard (optional)
=====================================

linkerd viz install | kubectl apply -f -

Verify:

linkerd check

Open dashboard:

linkerd viz dashboard

Dashboard shows:

* Request rate
* Latency
* Success rate
* Traffic graph
* mTLS status

====================================================
DISABLE LINKERD
===============

Disable namespace injection:

kubectl annotate namespace pqc-jwt 
linkerd.io/inject-

Restart:

kubectl rollout restart deployment/auth-service 
-n pqc-jwt

kubectl rollout restart deployment/user-service 
-n pqc-jwt

====================================================
FULL UNINSTALL
==============

Remove dashboard:

linkerd viz uninstall | kubectl delete -f -

Remove control plane:

linkerd uninstall | kubectl delete -f -

Verify:

kubectl get pods -A | grep linkerd

Expected:

(no output)

====================================================
FINAL FLOW
==========

Locust
↓
Traefik
↓
Linkerd Proxy 🔒
↓
Auth Service
↓
Linkerd Proxy 🔒
↓
User Service

Benefits:

✔ automatic mTLS
✔ service-to-service encryption
✔ traffic metrics
✔ retries
✔ observability
✔ no application code changes
