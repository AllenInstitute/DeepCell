import base64
import os
import subprocess
from pathlib import Path
from typing import Tuple

import boto3
import docker
import logging

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md   # noqa E501
# Needed to authenticate against this host to pull the image
AWS_DEEP_LEARNING_DOCKER_IMAGE_HOST = \
    '763104351884.dkr.ecr.us-west-2.amazonaws.com'


class ECRUploader:
    """Class for handling local build and upload of docker image to
    Elastic container service"""
    def __init__(self, repository_name: str,
                 image_tag: str, profile_name='default',
                 region_name='us-west-2'):
        """
        Parameters
        ----------
        repository_name
            Name of docker repository
        image_tag
            Image tag
        profile_name
            AWS profile to use
        region_name
            AWS region
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        self._boto_session = boto3.session.Session(
                profile_name=profile_name,
                region_name=region_name)
        self._region_name = region_name
        self._repository_host = self._get_repository_host()
        self._repository_name = repository_name
        self._image_tag = image_tag

    def build_and_push_container(
            self,
            dockerfile_path: Path,
    ) -> None:
        """
        Builds docker image locally and pushes to ECR
        Parameters
        ----------
        dockerfile_path
            Path to the dockerfile
        Notes
        ----------
        Note that if nothing changed to the Dockerfile compared with the
        same tag on ECR, then nothing is pushed
        """
        self._logger.info('Creating ecr repository')
        self._create_repository()

        self._logger.info('Building docker image')
        self._build_docker(path_to_dockerfile=dockerfile_path)

        self._logger.info('Tagging locally')
        self._tag_docker()

        self._logger.info('Pushing image to ecr')
        self._docker_push()

    @property
    def image_uri(self):
        host = self._get_repository_host()
        name = self._repository_name
        tag = self._image_tag
        return f'{host}/{name}:{tag}'

    def _get_repository_host(self):
        account = \
            self._boto_session.client('sts').get_caller_identity()['Account']
        return f'{account}.dkr.ecr.{self._region_name}.amazonaws.com'

    def _create_repository(self):
        ecr = self._boto_session.client('ecr', region_name=self._region_name)
        repos = ecr.describe_repositories()
        repos = repos['repositories']
        if self._repository_name in [x['repositoryName'] for x in repos]:
            self._logger.info(f'ECR repository {self._repository_name} '
                              f'already exists')
            return
        ecr.create_repository(repositoryName=self._repository_name)

    def _docker_login(self):
        username, password = self._get_ecr_credentials()
        client = docker.from_env()
        client.login(username=username, password=password,
                     registry=self._repository_host)

        # authenticate against private aws deep learning docker repository
        # in order to pull it
        # Need to execute the cmd directly rather than use the docker api;
        # docker api didn't work
        cmd = f'docker login --username AWS --password-stdin ' \
              f'{AWS_DEEP_LEARNING_DOCKER_IMAGE_HOST}'
        cmd = cmd.split(' ')
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   env=os.environ)
        stdout, stderr = process.communicate(input=password.encode('utf-8'))
        self._logger.info(stdout)
        if stderr:
            self._logger.error(stderr)
            raise RuntimeError(stderr)
        return username, password

    def _build_docker(self, path_to_dockerfile: Path):
        self._docker_login()
        docker_client = docker.APIClient()
        image_tag = f'{self._repository_name}:{self._image_tag}'
        build_res = docker_client.build(path=str(path_to_dockerfile.parent),
                                        tag=image_tag, rm=True, decode=True,
                                        nocache=True)
        for line in build_res:
            self._logger.info(line)
            if 'errorDetail' in line:
                raise RuntimeError(line['errorDetail']['message'])

    def _tag_docker(self):
        docker_client = docker.APIClient()
        full_repository_name = \
            f'{self._repository_host}/{self._repository_name}'
        tag_successful = docker_client.tag(image=self._repository_name,
                                           repository=full_repository_name,
                                           tag=self._image_tag)
        if not tag_successful:
            raise RuntimeError('Tagging was not successful')

    def _docker_push(self):
        ecr_username, ecr_password = self._docker_login()
        repository = f'{self._repository_host}/{self._repository_name}'
        auth_config = {
            'username': ecr_username,
            'password': ecr_password
        }
        docker_client = docker.APIClient()
        res = docker_client.push(repository=repository, tag=self._image_tag,
                                 auth_config=auth_config, decode=True,
                                 stream=True)
        for line in res:
            self._logger.info(line)
            if 'errorDetail' in line:
                raise RuntimeError(res['errorDetail']['message'])

    def _get_ecr_credentials(self) -> Tuple[str, str]:
        """
        Gets the login credentials for ECR
        """
        ecr = self._boto_session.client('ecr', region_name=self._region_name)
        auth = ecr.get_authorization_token()
        auth = auth['authorizationData'][0]
        password = base64.b64decode(auth['authorizationToken'])
        password = bytes.decode(password)
        password = password.replace('AWS:', '')
        return 'AWS', password
