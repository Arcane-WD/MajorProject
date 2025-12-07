"use client"
import React, {useDebugValue, useState} from 'react'
import axios from 'axios'
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';

const Inputimgcard = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const onFileChange = (event) => {
    const selected = event.target.files[0];
    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };
  const onFileUpload = async () => {
    if(!file) {
      console.log("No File Uploaded");
      return;
    }

    const formData = new FormData();
		formData.append(
			"myFile",
			file,
			file.name
		);
    console.log(file);
    axios.post("api/uploadfile", formData);
  };
  const fileData = () => {
    return file ? (
      <div>
        <h4>File Details</h4>
        <p>File Name: {file.name}</p>
        <p>File Type: {file.type}</p>
      </div>
    ) : (
      <div>
        <br />
        <h4>Choose before pressing upload</h4>
      </div>
    )
  }

  return (
    <div>
      <div className='d-flex justify-content-around'>
        <Card style={{ width: '40rem' }}>
        {file ? (
          <div>
            <Card.Img variant="top" src={preview} width={600} height={400} />
          </div>
        ):(
          <div>
            <Card.Img variant="top" src="\placeholder.svg" width={600} />
          </div>
        )
        }
        <Card.Body>
          <Card.Title>Upload Architectural Image</Card.Title>
          <Card.Text>
              <input type="file" onChange={onFileChange}/>
              <br/>
          </Card.Text>
        <Button onClick={onFileUpload}>Upload</Button>
        {fileData()}
        </Card.Body>
      </Card>
      </div>
    </div>
  )
}

export default Inputimgcard