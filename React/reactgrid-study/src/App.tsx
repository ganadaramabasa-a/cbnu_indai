import { useState } from "react";
import {
  ReactGrid,
  Column,
  Row,
  Id,
  CellChange,
  TextCell,
} from "@silevis/reactgrid";
import "@silevis/reactgrid/styles.css";

interface Person {
  name: string;
  surname: string;
}

function getPeople(): Person[] {
  return [
    { name: "Thomas", surname: "Goldman" },
    { name: "Susie", surname: "Quattro" },
    { name: "", surname: "" },
  ];
}

function getColumns(): Column[] {
  return [
    { columnId: "name", width: 150 },
    { columnId: "surname", width: 150 },
  ];
}

const headerRow: Row = {
  rowId: "header",
  cells: [
    { type: "header", text: "Name" },
    { type: "header", text: "Surname" },
  ],
};

function getRows(people: Person[]): Row[] {
  return [
    headerRow,
    ...people.map<Row>((person, idx) => ({
      rowId: idx,
      cells: [
        { type: "text", text: person.name },
        { type: "text", text: person.surname },
      ],
    })),
  ];
}

function applyChangesToPeople(
  changes: CellChange<TextCell>[],
  prevPeople: Person[]
): Person[] {
  changes.forEach((change) => {
    const personIndex = change.rowId as number;
    const fieldName = change.columnId;
    prevPeople[personIndex][fieldName as keyof Person] = change.newCell.text;
  });
  return [...prevPeople];
}

export default function App() {
  const [people, setPeople] = useState<Person[]>(getPeople());

  const rows = getRows(people);
  const columns = getColumns();

  function handleChanges(changes: CellChange<TextCell>[]) {
    setPeople((prevPeople) => applyChangesToPeople(changes, prevPeople));
  }

  return (
    <ReactGrid rows={rows} columns={columns} onCellsChanged={handleChanges} />
  );
}
