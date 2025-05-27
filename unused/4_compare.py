import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

# Load files
generated_planogram_df = pd.read_csv(r'received_data\Planogram\planogram_2.csv')
expected_planogram_df = pd.read_excel(r'received_data\Planogram\planogram_empty.xlsx')

def split_row(df,index,empty):
    # Error checking 
    if index < 0 or index >=len(df):
        raise IndexError("Index our of range!")
    
    # Extract row that is empty - not used currently
    row_to_move = df.iloc[index]
    rack_id = row_to_move['rack_id']
    
    # Remove the row
    df_dropped = df.drop(index)

    # Create empty rows
    empty_rows = pd.DataFrame([{col: 'empty' for col in df.columns}] * empty)
    empty_rows['rack_id'] = rack_id
    
    # Index for inseting empty row
    insertion_index = index

    # Split dropped df to insert empty row
    top = df_dropped.iloc[:insertion_index]
    bottom = df_dropped[insertion_index:]
    
    # create new dataframe
    new_df = pd.concat([top, empty_rows, bottom])

    return new_df

def planogram_analysis(generated_df, expected_df):

    # Find the empty rows from generated_df
    empty_rows_index = generated_df[generated_df['class_name'] == 'empty'].index
    print(empty_rows_index)
    
    # Empty list to store index info
    neighbour_info = []
    

    # iterate index and keep index of before and after & error checking
    for idx in empty_rows_index:
        if idx - 1 >= 0:
            before_class = generated_df.loc[idx -1, 'class_name']
            rack_id = generated_df.loc[idx, 'rack_id']
        else: 
            None
        if idx + 1 < len(generated_df):
            after_class = generated_df.loc[idx + 1, 'class_name']
            rack_id = generated_df.loc[idx, 'rack_id']
        else:
            None

        neighbour_info.append([idx, before_class, after_class, rack_id])
        print(neighbour_info)

    df_to_change = generated_df
    last_index = 0
    empty_init = 0
    for idx, before_class, after_class, rack_id in neighbour_info:
        #! idx is index in generated_df 
        idx += last_index 
        
        matching_indices_before = expected_df[expected_df['class_name'] == before_class].index.tolist() # Find last occurance of before_class -> [-1]
        matching_indices_after = expected_df[expected_df['class_name'] == after_class].index.tolist()

        matching_indices_before_generated = generated_df[generated_df['class_name'] == before_class].index.tolist() # Find last occurance of before_class -> [-1]
        matching_indices_after_generated = generated_df[generated_df['class_name'] == after_class].index.tolist() # Find first occurance of after_class -> [0]
        
        if pd.isna(before_class) and after_class: # If first item is missing
            # Partially missing algo
            empty_init += 1
            next_item_index = matching_indices_after[-1]+1
            
            next_item_index_generated = matching_indices_after_generated[-1]+1
            
            item_count_expected = 0 # number of item counts for first in expected df 
            while pd.notna(expected_df.loc[next_item_index-item_count_expected]['class_name']):
                item_count_expected +=1

            item_count_generated = 0 # number of item counts for first in generated df
            while pd.notna(generated_df.loc[next_item_index_generated-item_count_generated]['class_name']) and not generated_df.loc[next_item_index_generated-item_count_generated]['class_name'] == 'empty':
                item_count_generated +=1
            
            # Compare generated and expected numbers
            # Insert empty row at top of row
            last_index += abs(item_count_expected-item_count_generated) - 1 
            df_to_change = split_row(df_to_change,idx,item_count_expected-item_count_generated) 
            df_to_change = df_to_change.reset_index(drop=True)
            
                #Fully missing algo
            

        elif pd.isna(after_class) and before_class:
            next_item_index = matching_indices_before[-1]-1

            next_item_index_generated = matching_indices_before_generated[0]-1

            item_count_expected = 0 # number of item counts for first in expected df 
            while pd.notna(expected_df.loc[next_item_index-item_count_expected]['class_name']):
                item_count_expected +=1

            item_count_generated = 0 # number of item counts for first in generated df
            while pd.notna(generated_df.loc[next_item_index_generated-item_count_generated]['class_name']) and not generated_df.loc[next_item_index_generated-item_count_generated]['class_name'] == 'empty':
                item_count_generated +=1

        # elif before_class == after_class:
        #within class missing
            # total_index_value = abs(matching_indices_before[-1]-matching_indices_after[0])
            # total_item_expected = total_index_value + 1
            # print(f"Quantity : {quantity_of_empty}")
            # last_index += quantity_of_empty - 1
            # df_to_change = split_row(df_to_change,idx,quantity_of_empty)
            # df_to_change = df_to_change.reset_index(drop=True)

        else:
            print(f"idx: {idx}")
            quantity_of_empty = abs(matching_indices_before[-1]-matching_indices_after[0])-1
            print(f"quantity_of_empty : {quantity_of_empty}")
            if quantity_of_empty == 0:
                quantity_of_empty = 1
            last_index += quantity_of_empty - 1
            df_to_change = split_row(df_to_change,idx,quantity_of_empty)
            df_to_change = df_to_change.reset_index(drop=True)
        
    df_to_change.to_csv('received_data/CSV/main_case.csv', index=False)
    return df_to_change

def check_planogram(generated_df, expected_df, output_excel_path=None):
    gen = generated_df.rename(columns={'class_name': 'generated_class_name'})
    exp = expected_df.rename(columns={'class_name': 'expected_class_name'})

    combined_rows = []
    rack_ids = gen['rack_id'].unique()

    for rack in rack_ids:
        gen_rack = gen[gen['rack_id'] == rack].reset_index(drop=True)
        exp_rack = exp[exp['rack_id'] == rack].reset_index(drop=True)

        max_len = max(len(gen_rack), len(exp_rack))

        # Pad shorter with empty rows
        if len(gen_rack) < max_len:
            empty_rows = pd.DataFrame({
                'rack_id': [rack] * (max_len - len(gen_rack)),
                'generated_class_name': ['empty'] * (max_len - len(gen_rack))
            })
            gen_rack = pd.concat([gen_rack, empty_rows], ignore_index=True)

        if len(exp_rack) < max_len:
            empty_rows = pd.DataFrame({
                'rack_id': [rack] * (max_len - len(exp_rack)),
                'expected_class_name': ['empty'] * (max_len - len(exp_rack))
            })
            exp_rack = pd.concat([exp_rack, empty_rows], ignore_index=True)

        for i in range(max_len):
            gen_cls = gen_rack.at[i, 'generated_class_name']
            exp_cls = exp_rack.at[i, 'expected_class_name']

            if gen_cls == 'empty' or exp_cls == 'empty':
                output = 'RED'
            elif gen_cls != exp_cls:
                output = 'YELLOW'
            else:
                output = ''

            combined_rows.append({
                'rack_id': rack,
                'generated_class_name': gen_cls,
                'expected_class_name': exp_cls,
                'output': output
            })

        combined_rows.append({'rack_id': '', 'generated_class_name': '', 'expected_class_name': '', 'output': ''})

    # Remove trailing blank row
    if combined_rows and all(value == '' for value in combined_rows[-1].values()):
        combined_rows.pop()

    result_df = pd.DataFrame(combined_rows)

    if output_excel_path:
        def highlight_output(s):
            colors = []
            for v in s:
                if v == 'RED':
                    colors.append('background-color: red')
                elif v == 'YELLOW':
                    colors.append('background-color: yellow')
                else:
                    colors.append('')
            return colors

        styled = result_df.style.apply(highlight_output, subset=['output'])
        styled.to_excel(output_excel_path, index=False)
        print(f"Saved colored output to {output_excel_path}")

    return result_df

if __name__ == "__main__":
    new_generated_df = planogram_analysis(generated_planogram_df, expected_planogram_df)
    comparison_df = check_planogram(new_generated_df, expected_planogram_df, "received_data/CSV/planogram_comparison.xlsx")
    
    
