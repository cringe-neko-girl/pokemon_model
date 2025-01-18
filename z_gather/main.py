import os
import aiohttp
import asyncio
import aiofiles
from tqdm import tqdm
from bs4 import BeautifulSoup
import logging

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Pokémon API URLs
pokemon_api_url = "https://pokeapi.co/api/v2/pokemon"

# Semaphore for limiting concurrent requests
semaphore = asyncio.Semaphore(10)  # Max concurrent requests

# Fetch all Pokémon names asynchronously
async def fetch_all_pokemon_names():
    """Fetch all Pokémon names asynchronously from the Pokémon API."""
    pokemon_names = []
    url = pokemon_api_url
    while url:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    for result in data["results"]:
                        pokemon_names.append(result["name"])
                    url = data.get("next")
                else:
                    logger.error("Failed to fetch Pokémon names.")
                    break
    return pokemon_names

# Create all folders for each Pokémon
def create_pokemon_folders(pokemon_names):
    """Create folders for each Pokémon sequentially."""
    for pokemon_name in pokemon_names:
        folder_name_train = f"t_data/train/{pokemon_name}"
        folder_name_test = f"t_data/test/{pokemon_name}"

        # Create directories if they don't exist
        os.makedirs(folder_name_train, exist_ok=True)
        os.makedirs(folder_name_test, exist_ok=True)
        logger.info(f"Created folders for {pokemon_name}.")

# Fetch images of a Pokémon with semaphore for concurrency control
async def fetch_pokemon_images(session, search_query, semaphore, num_images=20):
    """Download Pokémon images from Bing search asynchronously with semaphore."""
    async with semaphore:  # Limit the number of concurrent requests
        bing_url = f"https://www.bing.com/images/search?q={search_query.replace(' ', '+')}&FORM=HDRSC2"
        async with session.get(bing_url) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch images for {search_query} from Bing.")
                return []

            soup = BeautifulSoup(await response.text(), "html.parser")
            image_tags = soup.find_all("img", class_="mimg", limit=num_images)
            if not image_tags:
                logger.error(f"No images found for {search_query}.")
                return []

            # Extract image URLs
            img_urls = [img_tag.get("src") or img_tag.get("data-src") for img_tag in image_tags if img_tag.get("src") or img_tag.get("data-src")]
            return img_urls

# Download and save image asynchronously
async def download_image(session, img_url, folder_name, index):
    """Download and save a single image asynchronously."""
    try:
        async with session.get(img_url) as img_response:
            img_data = await img_response.read()
            async with aiofiles.open(os.path.join(folder_name, f"{index + 1}.jpg"), "wb") as img_file:
                await img_file.write(img_data)
            logger.info(f"Image {index + 1} saved successfully.")
    except Exception as e:
        logger.error(f"Failed to download image {index + 1}: {e}")

# Asynchronous task for fetching images of all Pokémon
async def fetch_images_for_pokemon(session, pokemon_name, semaphore):
    """Fetch images for a single Pokémon sequentially."""
    folder_name_train = f"t_data/train/{pokemon_name}"
    folder_name_test = f"t_data/test/{pokemon_name}"

    # Fetch images for training and testing sets
    train_images = await fetch_pokemon_images(session, pokemon_name, semaphore, num_images=10)
    test_images = await fetch_pokemon_images(session, pokemon_name, semaphore, num_images=10)

    # Download images sequentially for training and testing
    for i, img_url in enumerate(train_images):
        await download_image(session, img_url, folder_name_train, i)
    for i, img_url in enumerate(test_images):
        await download_image(session, img_url, folder_name_test, i)

# Main function for executing the scraping process
async def main():
    # Fetch all Pokémon names
    pokemon_names = await fetch_all_pokemon_names()
    logger.info(f"Fetched {len(pokemon_names)} Pokémon names.")
    
    # Create all necessary folders
    create_pokemon_folders(pokemon_names)

    # Create an aiohttp session for fetching images
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pokemon_name in pokemon_names:
            tasks.append(fetch_images_for_pokemon(session, pokemon_name, semaphore))
        print("This will take a while to gather all the data (noting the 0%), but once it's ready, it will rapidly compute.\nPlease wait.")
        # Use tqdm to display progress while downloading images
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading images"):
            await future

# Run the script asynchronously
if __name__ == "__main__":
    asyncio.run(main())
