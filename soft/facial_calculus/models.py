from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Document(models.Model):
    #docfile = models.FileField(upload_to='documents/%Y/%m/%d')
	docfile = models.FileField(upload_to='pictures')
	filename = models.CharField(max_length = 200, default = 'aaaa')

class DocumentOutput(models.Model):
	docfile = models.FileField(upload_to='output')
	filename = models.CharField(max_length = 200, default = 'aaaa')
	textfield = models.TextField(default = '')
	description = models.TextField ( default = '')