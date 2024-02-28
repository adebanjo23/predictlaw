import random
from django.utils import timezone
from django.db import models


class User(models.Model):
    username = models.CharField(max_length=255, null=False, unique=True)
    phone = models.CharField(max_length=22, null=False)
    email = models.EmailField(unique=True)
    is_verified = models.BooleanField(default=False)


class OTP(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    otp_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)

    objects = models.Manager()

    def __str__(self):
        # noinspection PyUnresolvedReferences
        return f"OTP for {self.user.email}"

    @staticmethod
    def generate_otp_code(length=6):
        numbers = '0123456789'
        return ''.join(random.choice(numbers) for i in range(length))

    def save(self, *args, **kwargs):
        if not self.otp_code:
            self.otp_code = self.generate_otp_code()
        super().save(*args, **kwargs)

    @property
    def is_expired(self):
        # Assuming OTP is valid for 10 minutes
        return timezone.now() - self.created_at > timezone.timedelta(minutes=10)
