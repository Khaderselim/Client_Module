
# Introduction

This project introduces a semi-automated PrestaShop module designed for e-commerce users to monitor and extract data from competitors’ product listings. The module allows users to select products from their catalog and input URLs of similar products offered by competitors. 

Once configured, the module automatically retrieves key information from the specified URLs — including product title, price, description, and stock status — providing valuable insights for pricing and inventory strategies. 

Data extraction is powered by a Python-based API running in a Dockerized server environment, ensuring scalability, portability, and ease of deployment. This integration empowers PrestaShop users to enhance their competitive intelligence and stay agile in a rapidly evolving market.

---

# Getting Started
## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
   - [API Setup with Docker](#1-api-setup-with-docker)
   - [Module Installation in PrestaShop](#2-module-installation-in-prestashop)
4. [Start Tracking Products](#4-start-tracking-products)

## Overview

Our goal with this project is to make product tracking simple and effortless. This guide will walk you through everything you need to get started with the module — from setting up your Docker server and installing the PrestaShop module to tracking your first competitor products. With just a few steps, you'll be able to monitor and extract valuable data from your competitors' listings with ease.

---

## Prerequisites

Ensure you have the following installed:

- **Python** (3.12)
- **PHP** (8.0)
- **PrestaShop** (1.7 or higher)
- **Docker**

---

# Installation

## 1. API Setup with Docker

For a smooth and quick installation, we recommend running the API using **Docker**.

### Steps

1. **Download the `server` folder** from the repository:
    - [here](https://github.com/Khaderselim/Client_Module.git)

2. **Open a terminal** in the project folder (where the `docker-compose.yml` file is located), then run:

    ```bash
    docker-compose up --build
    ```

3. The API will start and be accessible at:

    ```
    http://localhost:8000
    ```

---

### ⚙️ Port Customization

If port `8000` is already in use or you want to change it, open the `docker-compose.yml` file and update the following line:

```yaml
ports:
  - "8000:8000"  # Change HOST_PORT to any port you prefer (e.g., 8080:8000)
```

After making changes, restart the container:

```bash
docker-compose up --build
```

---

### 🔁 Restarting & Stopping

- **Stop the API**: Press `Ctrl + C` in the terminal.
- **Restart the API**: Run:

    ```bash
    docker-compose up
    ```

---

### 🐳 Optional: Use Docker Desktop

If you prefer a GUI, you can use **Docker Desktop** to:

- Start/Stop containers
- View logs
- Monitor ports and resource usage

---

## 2. Module Installation in PrestaShop

### Step 1: Download the Module

- Download the `client` folder from the repository:
    - [here](https://github.com/Khaderselim/Client_Module.git)

---

### Step 2: Move the Module to PrestaShop

- Move the `client` folder into the `modules/` directory of your PrestaShop installation.

**Example**: If using **Laragon**, the path might look like:

```
C:/laragon/www/prestashop/modules
```

---

### Step 3: Access PrestaShop Admin Panel

1. Open your PrestaShop admin panel in a browser. If running locally, it may look like:

    ```
    http://localhost/prestashop/admin
    ```

2. If it doesn't open:
    - Ensure your **local server** (like Laragon or XAMPP) is running.
    - Check the admin folder name — PrestaShop sometimes renames it (e.g., `admin123xyz`). You can rename it back to `admin` or find the correct folder name inside your PrestaShop directory.

---

### Step 4: Install the Module

1. In the PrestaShop back office, navigate to:

    ```
    Modules → Module Manager → Uninstalled Modules
    ```

2. Find the **Product Tracking** module.
3. Click **Install**.

---

## 3. Insert Your Product Table in the Database

### Step 1: Connect to the PrestaShop Database

Use one of the following tools or any other MySQL-compatible tool to connect :

- **phpMyAdmin** (typically included with Laragon/XAMPP)
- **MySQL Workbench**
- **Database tools in your IDE** (e.g., VS Code, PhpStorm, etc.)


---

### Step 2: Insert Your Product Table

Create and insert your custom product table into the PrestaShop database. Name the table appropriately (e.g., `my_products`) and keep track of the name for future steps.

---



## 4. Start Tracking Products

### Step 1: Create Categories

1. Go to `Catalog` → Click the ➕ (Add Category).
2. Enter the **category name**.
3. Set the **status** (active/inactive).
4. Click **Save**.

---

### Step 2: Add Competitors

1. Go to `Competitors` → Click the ➕ (Add Competitor).
2. Fill in:
    - **Name**
    - **URL**
    - **Priority**
    - **Status**

> ⚠️ On the first competitor addition, a popup will ask for a product URL to extract the HTML pattern for price, name, etc. You can update it later using the “Update Pattern” button.

---

### Step 3: Add Products

1. Go to `Products` → Click `Add New`.
2. Fill in:
    - **Product name**
    - **Category**
    - **Competitors**

3. For each competitor:
    - Input the **product URL** (you can add multiple URLs).

4. Click **Save**.

---

![ClientModule](https://github.com/user-attachments/assets/e665480e-5479-40d2-90c1-3f57aac1908f)


