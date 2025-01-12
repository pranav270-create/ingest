terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = "us-west-2"
  profile = "default"
}

# Security group for GROBID
resource "aws_security_group" "grobid" {
  name        = "grobid-security-group"
  description = "Security group for GROBID server"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8070
    to_port     = 8070
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 instance
resource "aws_instance" "grobid" {
  ami           = "ami-0efcece6bed30fd98" # Ubuntu 22.04 with GPU support
  instance_type = "g4dn.xlarge"

  vpc_security_group_ids = [aws_security_group.grobid.id]

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              # Install Docker
              apt-get update
              apt-get install -y ca-certificates curl gnupg
              install -m 0755 -d /etc/apt/keyrings
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
              chmod a+r /etc/apt/keyrings/docker.gpg
              echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
              apt-get update
              apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

              # Install NVIDIA drivers and CUDA
              apt-get install -y linux-headers-$(uname -r)
              apt-get install -y nvidia-driver-525
              distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
              curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
              curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
              apt-get update && apt-get install -y nvidia-container-toolkit
              systemctl restart docker

              # Run GROBID
              docker run -d --rm --gpus all --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.1
              EOF

  tags = {
    Name = "grobid-server"
  }
}

# Output the public IP
output "public_ip" {
  value = aws_instance.grobid.public_ip
}