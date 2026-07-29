# Acme Robotics — Information Security Policy

## Passwords & Authentication
All employee accounts must use passwords of at least 14 characters containing upper case, lower case, a number, and a symbol. Passwords must be rotated every 90 days. Multi-factor authentication (MFA) is mandatory for all systems that access customer data. Acme Robotics uses hardware security keys (YubiKeys) for engineering staff with production access.

## Data Classification
Acme Robotics classifies data into four tiers: Public, Internal, Confidential, and Restricted. Restricted data — which includes customer PII and robot telemetry logs — must be encrypted at rest using AES-256 and may only be accessed from company-managed devices on the corporate VPN.

## Device Security
Company laptops must have full-disk encryption enabled and the screen must auto-lock after 5 minutes of inactivity. Lost or stolen devices must be reported to the Security Operations Center within 1 hour of discovery. Personal USB storage devices are prohibited on all engineering workstations.

## Incident Response
Any suspected security incident must be reported to security@acmerobotics.example immediately. The Security Operations Center operates 24/7 and guarantees a first response within 15 minutes for critical incidents. Employees must never attempt to investigate or remediate an incident themselves before reporting it.
