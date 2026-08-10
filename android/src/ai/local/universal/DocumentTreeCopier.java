package ai.local.universal;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public final class DocumentTreeCopier {
    private static final int MAX_DEPTH = 16;
    private static final int MAX_FILES = 8192;

    private DocumentTreeCopier() {}

    public static String copyTree(Context context, String treeUriText, String destinationText) {
        try {
            Uri treeUri = Uri.parse(treeUriText);
            String rootId = DocumentsContract.getTreeDocumentId(treeUri);
            Uri rootDocument = DocumentsContract.buildDocumentUriUsingTree(treeUri, rootId);
            File destination = new File(destinationText);
            File staging = new File(destinationText + ".importing");
            deleteRecursively(staging);
            if (!staging.mkdirs()) {
                return "Could not create private destination directory";
            }
            int[] fileCount = new int[] {0};
            copyDirectory(context.getContentResolver(), treeUri, rootDocument, staging, 0, fileCount);
            File previous = new File(destinationText + ".previous");
            deleteRecursively(previous);
            if (destination.exists() && !destination.renameTo(previous)) {
                deleteRecursively(staging);
                return "Could not rotate the existing private bundle";
            }
            if (!staging.renameTo(destination)) {
                previous.renameTo(destination);
                deleteRecursively(staging);
                return "Could not atomically finish the private bundle import";
            }
            deleteRecursively(previous);
            return "";
        } catch (Exception error) {
            String message = error.getMessage();
            return "Android document-tree copy failed: " +
                    (message == null || message.isEmpty() ? error.getClass().getSimpleName() : message);
        }
    }

    private static void copyDirectory(ContentResolver resolver, Uri treeUri, Uri documentUri,
                                      File destination, int depth, int[] fileCount) throws Exception {
        if (depth > MAX_DEPTH) throw new SecurityException("Bundle directory nesting is too deep");
        String parentId = DocumentsContract.getDocumentId(documentUri);
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId);
        String[] columns = new String[] {
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE
        };
        try (Cursor cursor = resolver.query(children, columns, null, null, null)) {
            if (cursor == null) throw new SecurityException("Document provider returned no directory listing");
            while (cursor.moveToNext()) {
                String documentId = cursor.getString(0);
                String displayName = safeName(cursor.getString(1));
                String mimeType = cursor.getString(2);
                Uri child = DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId);
                File output = new File(destination, displayName);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mimeType)) {
                    if (!output.exists() && !output.mkdirs()) {
                        throw new SecurityException("Could not create bundle subdirectory " + displayName);
                    }
                    copyDirectory(resolver, treeUri, child, output, depth + 1, fileCount);
                } else {
                    fileCount[0]++;
                    if (fileCount[0] > MAX_FILES) throw new SecurityException("Bundle contains too many files");
                    copyFile(resolver, child, output);
                }
            }
        }
    }

    private static void copyFile(ContentResolver resolver, Uri source, File destination) throws Exception {
        File temporary = new File(destination.getParentFile(), destination.getName() + ".importing");
        try (InputStream input = resolver.openInputStream(source);
             FileOutputStream output = new FileOutputStream(temporary, false)) {
            if (input == null) throw new SecurityException("Document provider could not open " + destination.getName());
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) output.write(buffer, 0, read);
            output.getFD().sync();
        } catch (Exception error) {
            temporary.delete();
            throw error;
        }
        if (destination.exists() && !destination.delete()) {
            temporary.delete();
            throw new SecurityException("Could not replace " + destination.getName());
        }
        if (!temporary.renameTo(destination)) {
            temporary.delete();
            throw new SecurityException("Could not finish " + destination.getName());
        }
    }

    private static String safeName(String input) {
        if (input == null || input.isEmpty()) return "unnamed";
        String value = input.replaceAll("[^A-Za-z0-9._-]", "_");
        if (value.equals(".") || value.equals("..")) return "unnamed";
        return value.length() > 180 ? value.substring(0, 180) : value;
    }

    private static void deleteRecursively(File file) throws Exception {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children == null) throw new SecurityException("Could not inspect private staging directory");
            for (File child : children) deleteRecursively(child);
        }
        if (!file.delete()) throw new SecurityException("Could not clear private staging directory");
    }
}
