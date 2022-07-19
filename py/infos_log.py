import colorama as col


def info_face_recognition(log: str, person_img_id: str = 0) -> None:
    if log == 'loading_path':
        print(col.Fore.GREEN + '|=========| PATH LOADED SUCCESSFULLY |=========|')

    if log == 'os.listdir_error':
        print(col.Fore.LIGHTRED_EX + 'There are no images in the directory!')

    if log == 'new_person':
        print(col.Fore.WHITE + '|=========| NEW PERSON SEARCH: ' + col.Fore.YELLOW + str(person_img_id) + " ", end='')

    if log == 'encoding_new_person':
        print(col.Fore.CYAN + "ENCODING >>>> ", end='')

    if log == 'complete_new_person':
        print(col.Fore.LIGHTGREEN_EX + "COMPLETE ")

    if log == 'face_not_found':
        print(col.Fore.LIGHTRED_EX + "UNENCODED FACE")

    if log == 'new_path':
        print(col.Fore.WHITE +
              '|-----------------------------------------------------------------------------------------------------|')
        print(col.Fore.WHITE + '|-----| NEW PATH >>>> ' + col.Fore.LIGHTGREEN_EX + 'CREATED')
        print(col.Fore.WHITE +
              '|-----------------------------------------------------------------------------------------------------|')

    if log == 'file_not_existing':
        print(col.Fore.YELLOW + 'File already identified or not existing')

    if log == 'search_data':
        print(col.Fore.WHITE +
              '|=========| ENCODING ' + col.Fore.CYAN + person_img_id + col.Fore.WHITE + " >>>> ", end='')

    if log == 'complete_search':
        print(col.Fore.LIGHTGREEN_EX + 'COMPLETE' + col.Fore.WHITE + '|------| ', end='')


def info_faceAuth(log: str, number_img: int = 0, img_read: str = None) -> None:
    if log == 'os.listdir_error':
        print(col.Fore.LIGHTRED_EX + 'There are no images in the directory!')
        print(col.Fore.RED + 'Run a face detection on the ' + col.Fore.CYAN + 'main.py ' + col.Fore.RED + 'file to' +
              'generate images.')

    if log == 'loading_path':
        print(col.Fore.GREEN + '|=========| PATH LOADED SUCCESSFULLY |=========|')

    if log == 'number_img':
        print(col.Fore.CYAN + "Archive amount: " + col.Fore.YELLOW + str(number_img))

    if log == 'removed_archives':
        print(col.Fore.YELLOW +
              '|--------------------------------------------------------------------------------------------------|')
        print('| ' + col.Fore.YELLOW + str(number_img) + col.Fore.MAGENTA + ' FILES WERE REMOVED')
        print(col.Fore.YELLOW +
              '|--------------------------------------------------------------------------------------------------|')

    if log == 'reading process':
        print(col.Fore.CYAN + 'IMG .png: ' + col.Fore.LIGHTMAGENTA_EX + img_read, end='')

    if log == 'processing_detector':
        print(col.Fore.CYAN + 'Processing -> ' + img_read, '|----------------| ', end='')


def info_decoder(log: str, path: str = None, img: str = None, dir_csv: str = None, person: str = None) -> None:
    if log == 'path_reading':
        print(col.Fore.CYAN + 'Path reading: ' + col.Fore.LIGHTYELLOW_EX + path)

    if log == 'img_encode':
        print(col.Fore.WHITE + 'Encoding image >>>> ' + img, end='')

    if log == 'complete':
        print(col.Fore.WHITE + ' |-------| ' + col.Fore.GREEN + 'COMPLETE' + col.Style.RESET_ALL)

    if log == 'csv_file':
        print(col.Fore.CYAN + ".csv file with the encoding created in: " + col.Fore.MAGENTA + dir_csv)

    if log == 'decoder':
        print(col.Fore.CYAN + 'Decoding facial recognition file from >>>> ' + col.Fore.LIGHTWHITE_EX + person, end='')

    if log == 'complete_decoder':
        print(col.Fore.WHITE + " |-------| " + col.Fore.GREEN + "COMPLETE" + col.Style.RESET_ALL)
