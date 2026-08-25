# Russian Trusted CA (НУЦ Минцифры)

Needed so the API container can verify TLS for `enter.tochka.com`
(certificate chain ends at **Russian Trusted Root CA**, not in the default Mozilla store).

| File | Source |
|------|--------|
| `russian_trusted_root_ca.crt` | https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt |
| `russian_trusted_sub_ca.crt` | https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt |

Installed into the image via `Dockerfile` (`update-ca-certificates` + append to `certifi`).
