import { NextResponse} from "next/server";
import fs from "fs";
import path from "path";

export async function  POST(req) {
    const form = await req.formData();
    const file = form.get("myFile");

    if (!file) {
        return NextResponse.json({ error: "No File"}, {status:400})
    }

    // convert file to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    //ensure uploaded folder exists
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, {recursive:true});

    //save file
    const filePath = path.join(uploadDir, file.name);
    fs.writeFileSync(filePath, buffer);

    //return URL
    return NextResponse.json({
        url: `/uploads/${file.name}`
    })
}