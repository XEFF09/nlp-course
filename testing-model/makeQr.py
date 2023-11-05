import qrcode

# URL to convert to QR code
url = "https://colab.research.google.com/drive/10jOAZS7ehvPPUfK7lt-SrjpiCMdtTLRe"

# Generate a QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create a QR code image
img = qr.make_image(fill_color="black", back_color="white")

# Save the QR code image to a file
img.save("colab_qr_code.png")

# You can also display the QR code using a library like PIL
img.show()