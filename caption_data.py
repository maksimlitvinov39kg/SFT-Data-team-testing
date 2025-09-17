import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage
from PIL import Image

load_dotenv()
creds = os.getenv("GIGACHAT_CREDENTIALS")
scope = os.getenv('SCOPE')

giga = GigaChat(
    model="GigaChat-2-Pro",
    credentials=creds,
    scope=scope,
    verify_ssl_certs=False,
    profanity_check=True
)

def load_config():
    try:
        with open('data/json/character_names.json', 'r', encoding='utf-8') as f:
            character_names = json.load(f)
        
        with open('data/json/base_prompts.json', 'r', encoding='utf-8') as f:
            baseline_prompts = json.load(f)
            
        return character_names, baseline_prompts
        
    except FileNotFoundError as e:
        print(f"Ошибка: файл конфигурации не найден: {e}")
        print("Убедитесь, что файлы character_names.json и baseline_prompts.json находятся в папке data/json/")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения JSON файла: {e}")
        return None, None

def convert_webp_to_png(webp_path):
    """Конвертирует WebP изображение в PNG и возвращает путь к новому файлу"""
    try:
        # Создаем путь для PNG файла
        png_path = str(Path(webp_path).with_suffix('.png'))
        
        # Открываем WebP изображение
        with Image.open(webp_path) as img:
            # Конвертируем в RGB если изображение имеет альфа-канал
            if img.mode in ('RGBA', 'LA', 'P'):
                # Создаем белый фон
                background = Image.new('RGB', img.size, (255, 255, 255))
                # Если есть прозрачность, конвертируем правильно
                if img.mode == 'P':
                    img = img.convert('RGBA')
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # используем альфа-канал как маску
                else:
                    background.paste(img)
                img = background
            # Сохраняем как PNG
            img.save(png_path, 'PNG')
        
        print(f"  ✓ Конвертировано: {webp_path} -> {png_path}")
        return png_path
    except Exception as e:
        print(f"  ✗ Ошибка конвертации {webp_path}: {e}")
        return None

def get_character_name_from_path(image_path, character_names):
    path_obj = Path(image_path)

    folder_name = path_obj.parent.name.lower()
    if folder_name != 'images':
        if folder_name in character_names:
            return character_names[folder_name]

        for key, value in character_names.items():
            if key in folder_name or folder_name in key:
                return value

    file_name = path_obj.stem.lower()
    
    if file_name in character_names:
        return character_names[file_name]
    
    for key, value in character_names.items():
        if key in file_name or file_name in key:
            return value
    if folder_name != 'images':
        return folder_name.replace('_', ' ').title()
    else:
        return file_name.replace('_', ' ').title()

def process_image(image_path, character_names, baseline_prompts):
    try:
        # Проверяем, нужно ли конвертировать WebP в PNG
        original_path = image_path
        if Path(image_path).suffix.lower() == '.webp':
            print(f"  Обнаружен WebP файл, конвертируем...")
            converted_path = convert_webp_to_png(image_path)
            if converted_path:
                image_path = converted_path
            else:
                print(f"  ✗ Не удалось конвертировать WebP файл")
                return None
        
        character_name = get_character_name_from_path(image_path, character_names)
        
        random_question = random.choice(baseline_prompts)
        
        with open(image_path, "rb") as f:
            file = giga.upload_file(f)
        
        # Создаем более информативный промпт для описания персонажа
        caption_prompt = f"""Проанализируй изображение и ответь на вопрос: {random_question}
        
Имя персонажа: {character_name}
"""

        resp = giga.invoke([
            HumanMessage(
                content=caption_prompt,
                additional_kwargs={
                    'attachments': [file.id_]
                }
            )
        ],
        temperature=0.9)
        
        result = {
            "image": original_path,  # Сохраняем оригинальный путь
            "converted_image": image_path if image_path != original_path else None,  # Путь к конвертированному файлу
            "character_folder": Path(original_path).parent.name,
            "character_name": character_name,
            "question": random_question,
            "answer": resp.content,
            "detailed_analysis": {
                "character_name_provided": character_name,
                "image_format": Path(original_path).suffix.lower(),
                "was_converted": image_path != original_path
            }
        }
        
        print(f"✓ Обработано: {original_path} -> {character_name}")
        print(f"  Папка: {Path(original_path).parent.name}")
        print(f"  Вопрос: {random_question}")
        print(f"  Длина ответа: {len(resp.content)} символов")
        return result
        
    except Exception as e:
        print(f"✗ Ошибка при обработке {original_path}: {e}")
        return None

def save_results(results, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        return False

def load_existing_results(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Предупреждение: файл {output_file} повреждён, создаём новый")
            return []
    return []

def get_processed_images(existing_results):
    return {result['image'] for result in existing_results}

def find_all_images(data_folder):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(data_folder):
        for filename in files:
            if Path(filename).suffix.lower() in image_extensions:
                image_path = os.path.join(root, filename)
                image_files.append(image_path)
    
    return image_files

def cleanup_converted_files(results):
    """Очищает временные PNG файлы, созданные при конвертации WebP"""
    for result in results:
        converted_image = result.get('converted_image')
        if converted_image and converted_image != result['image']:
            try:
                if os.path.exists(converted_image):
                    os.remove(converted_image)
                    print(f"  Удален временный файл: {converted_image}")
            except Exception as e:
                print(f"  Ошибка при удалении временного файла {converted_image}: {e}")

def main():
    character_names, baseline_prompts = load_config()
    if not character_names or not baseline_prompts:
        return
    
    data_folder = "data/images"
    
    output_folder = "data/"
    os.makedirs(output_folder, exist_ok=True)
    
    output_file = os.path.join(output_folder, "annotation.json")
    
    existing_results = load_existing_results(output_file)
    processed_images = get_processed_images(existing_results)
    
    if not os.path.exists(data_folder):
        print(f"Папка {data_folder} не найдена!")
        return
    
    all_image_files = find_all_images(data_folder)

    image_files = [img for img in all_image_files if img not in processed_images]
    
    if not image_files:
        if processed_images:
            print(f"Все изображения в папке {data_folder} уже обработаны!")
        else:
            print(f"В папке {data_folder} не найдено новых изображений!")
        return
    
    print(f"Найдено {len(image_files)} новых изображений для обработки:")
    print(f"Уже обработано: {len(processed_images)} изображений")
    
    by_folder = {}
    for img in image_files:
        folder = Path(img).parent.name
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(img)
    
    print("\nИзображения по папкам:")
    for folder, files in by_folder.items():
        if files:
            sample_char_name = get_character_name_from_path(files[0], character_names)
            print(f"  {folder} ({sample_char_name}): {len(files)} файлов")
    
    new_results = []
    try:
        for i, image_path in enumerate(image_files, 1):
            print(f"\nОбработка {i}/{len(image_files)}: {image_path}")
            result = process_image(image_path, character_names, baseline_prompts)
            if result:
                new_results.append(result)
                all_results = existing_results + new_results
                if save_results(all_results, output_file):
                    print(f"  Промежуточное сохранение: {len(all_results)} записей")
    finally:
        # Очищаем временные файлы в конце работы
        if new_results:
            cleanup_converted_files(new_results)
    
    if new_results:
        all_results = existing_results + new_results
        if save_results(all_results, output_file):
            print(f"\n✓ Готово! Результаты сохранены в {output_file}")
            print(f"Обработано новых: {len(new_results)} изображений")
            print(f"Всего в датасете: {len(all_results)} записей")
            
            character_stats = {}
            for result in new_results:
                char = result['character_name']
                character_stats[char] = character_stats.get(char, 0) + 1
            
            print("\nСтатистика по персонажам (новые изображения):")
            for character, count in sorted(character_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {character}: {count} изображений")
            
            question_stats = {}
            for result in new_results:
                q = result['question']
                question_stats[q] = question_stats.get(q, 0) + 1
            
            print("\nСтатистика вопросов для новых изображений:")
            for question, count in sorted(question_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  '{question}': {count} раз")
            
            # Дополнительная статистика
            avg_response_length = sum(len(result['answer']) for result in new_results) / len(new_results)
            print(f"\nСредняя длина ответа: {avg_response_length:.0f} символов")
            
            conversion_count = sum(1 for result in new_results if result.get('detailed_analysis', {}).get('was_converted', False))
            if conversion_count > 0:
                print(f"Конвертировано WebP файлов: {conversion_count}")
        else:
            print("\n✗ Ошибка при сохранении результатов")
    else:
        print("\n✗ Не удалось обработать ни одного нового изображения")

if __name__ == "__main__":
    main()