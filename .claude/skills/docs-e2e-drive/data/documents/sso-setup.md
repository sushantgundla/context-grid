# Setting up single sign-on

Single sign-on is available on Growth and Enterprise plans. Northwind Cloud supports SAML 2.0
and OpenID Connect. SCIM provisioning is Enterprise only.

## Before you start

You need an account Owner, a verified domain, and admin access to your identity provider.
Domain verification is a DNS TXT record and usually propagates within an hour.

## SAML

Give your identity provider these two values, found under **Settings → Security → SSO**:

| Field | Value |
|---|---|
| ACS URL | `https://auth.northwind.cloud/saml/acs/<workspace-id>` |
| Entity ID | `https://auth.northwind.cloud/saml/metadata/<workspace-id>` |

We require the `NameID` to be the user's email address, in `emailAddress` format. A `NameID`
that is an opaque identifier will authenticate but will not match an existing user, which
shows up as a duplicate account rather than an error.

## Enforcing SSO

Once SSO is verified you can require it for everyone on your verified domains. Enforcement
does not apply to service tokens, and it deliberately does not apply to the account Owner who
enabled it — otherwise a misconfigured identity provider locks the whole company out with no
way back in.

## SCIM provisioning

SCIM 2.0 provisioning creates, updates and deactivates users automatically. Deactivating a
user in your identity provider removes their access within one minute and frees their seat at
the end of the billing period, not immediately.

## Common problems

- **"Invalid signature"** — your identity provider is signing with a certificate we do not
  have. Re-upload the metadata after a certificate rotation.
- **Users land on a new empty workspace** — the `NameID` format is wrong; see above.
- **Enforcement locked me out** — sign in as the Owner who enabled enforcement, who is always
  exempt.
