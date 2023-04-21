@staticmethod
def df_wrangle(data=None, start=datetime.now(), config=None,
               team_df: pd.DataFrame = None, project_df: pd.DataFrame = None, epic_df: pd.DataFrame = None) -> pd:
    """
        Args:
            @param epic_df:
            @param team_df:
            @param start: Timestamp received to set on
            @param data: It's Dictionary instance to get appropriate formatting
            @param config: Config parameters received
            @param project_df:
    """

    st = tm.time()
    if config is None:
        config = []
    if data is None:
        data = []
    pd.options.display.max_columns = 200
    if len(data) == 0:
        return
    entity = config["entity"]
    df_hist = df_sprint_update = []
    df_data = JiraDataWrangler.__retrieve_parents(data=data, entity=entity)
    df_jira = pd.json_normalize(df_data)

    # Setting Title type format on all columns
    prev_size = round(df_jira.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    df_jira = df_jira.dropna(axis=1, how='all')
    df_jira.columns = [x.title() for x in df_jira.columns]
    for value in config["remove_prefix"]:
        df_jira.columns = df_jira.columns.str.removeprefix(value.title())

    # List fields parameter configured and add the fields to be mapped
    fields = [x.title() for x in ''.join(config["fields"]).split(",")]
    fields_mapping = [x.title() for x in [*config["fields_mapping"]]]
    dictionary = [x.title() for x in ''.join(config["dictionary"]).split(",")]
    all_fields = dictionary + fields + fields_mapping

    # Removing unnecessary columns from data retrieved from Jira, it's to improve performance
    df_jira = df_jira[df_jira.columns.intersection(all_fields)]
    curr_size = round(df_jira.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    logging.debug(f"  === Size of PD DataFrame purged, from {prev_size} to {curr_size} MBs - ({len(df_jira)}) "
                  f"rows.")

    if config['entity'] == "Teams":
        config.update({'entity': 'Jira_Team_Names'})
        config.update({'jql': None})
        df_data = JiraDataWrangler.extract_from_jira(**config)
        team_names = pd.json_normalize(df_data)
        df_jira['Name'] = None
        config.update({'entity': 'Teams'})

        # Retrieve on Subtask, SubTask parent's keys from same USs Dataframe: SubTasks ==> Subtask's Parent
        df_jira = JiraDataWrangler.merge_parent_entity(df=df_jira, parent_df=team_names, cols="Key,Name",
                                                       left_col='Summary', right_col='Key', prefix='Team_',
                                                       merge_cols=True, force=True)
    if config["entity"] == "Epics":

        # To find each epic ultimate top parent (Financial Projects or last Key on issues linked)
        df_jira = JiraDataWrangler.__find_ultimate_parents(df=df_jira, entity=config["entity"], child_col='Key',
                                                           parent_col='Outwardissue.Key', ultimate_col='Projectkey')
        df_jira = JiraDataWrangler.merge_parent_entity(df=df_jira, parent_df=project_df.copy(),
                                                       cols=config["project_cols"], left_col='Projectkey',
                                                       right_col='Key')
    elif config["entity"] == "Userstories":
        us_list = df_jira.loc[df_jira['Issuetype.Subtask'] == False].copy()
        # Retrieve on Subtask, SubTask parent's keys from same USs Dataframe: SubTasks ==> Subtask's Parent
        df_jira = JiraDataWrangler.merge_parent_entity(df=df_jira, parent_df=us_list, cols="key,customfield_10103",
                                                       left_col='Parent.Key', right_col='Key', prefix='Sub_Task_',
                                                       merge_cols=True, force=True)
        # Retrieve Project and Epic details from Epics Dataframe: USs ==> Epics ==> Projects
        df_jira = JiraDataWrangler.merge_parent_entity(df=df_jira, parent_df=epic_df,
                                                       cols=config['epic_cols'], left_col='customfield_10103',
                                                       right_col='Key', prefix='Epic_')
        df_jira = JiraDataWrangler.merge_parent_entity(df=df_jira, parent_df=team_df.copy(),
                                                       cols=config['team_cols'], left_col='project.key',
                                                       right_col='Key', prefix='Team_')

        # Setting "Is Waste" Flag to True if tickets have any of the waste statuses
        if 'waste_statuses' in config and config['waste_statuses']:
            waste_statuses = ''.join(config["waste_statuses"]).split(",")
            df_jira['Is_Waste'] = np.where(df_jira['Status.Name'].isin(waste_statuses), True, False)
        # Setting Is Estimating Flag to True if no Estimation is set on items
        if 'Is_Estimated' not in df_jira.columns and 'Customfield_10002' in df_jira.columns:
            df_jira['Is_Estimated'] = np.where(df_jira['Customfield_10002'].isnull() |
                                               df_jira['Customfield_10002'].isna(), False, True)
        df_jira = JiraDataWrangler.filtering_default_values(df=df_jira, config=config, field='Customfield_10002',
                                                            parameter='default_effort_issue-type',
                                                            field_filtered='Issuetype.Name')
    if df_jira is None:
        return None

    fields = ''.join(config['fields']).split(",")
    fields = [x.title() for x in fields]
    if 'Key' not in fields:
        df_jira = df_jira.rename(columns={'Key': 'key'})
    # Re-mapping Fields according to entity configuration (fields_mapping)
    fields_mapping = dict((k.title(), v.title()) for k, v in config['fields_mapping'].items())
    re_map = dict((k.title(), v.title()) for k, v in fields_mapping.items())
    df_jira.rename(columns=re_map, inplace=True)

    # Explode Fields on multiple column if these are having dictionary of data, then this will duplicate rows per
    # each dictionary item found
    if 'expand' in config and 'Changelog.Histories' in df_jira.columns and config["entity"] != "Userstories":
        df_jira = df_jira.explode('Changelog.Histories')
        df_jira = pd.concat([df_jira.drop(['Changelog.Histories'], axis=1),
                             df_jira['Changelog.Histories'].apply(pd.Series)], axis=1)

    # Converting column dictionary into single value
    if 'dict_cols_to_explode' in config and config['dict_cols_to_explode']:
        for value in config['dict_cols_to_explode']:
            for k, v in config['dict_cols_to_explode'][value].items():
                try:
                    if value in df_jira.columns:
                        mask = (~df_jira[value].isnull())
                        df_jira.loc[mask, v] = [x[0][k] for x in df_jira.loc[mask, value]]

                except KeyError:
                    logging.WARN(f" --- Dictionary on Col '{value}', doesn't have '{k}' element, validate "
                                 f"'dict_cols_to_explode' parameter on config file for {config['entity']}")
    if 'jira_mask' in config and config['jira_mask']:
        mask = None
        temp = df_jira.copy()
        # for value in config['jira_mask']:
        for item in config['jira_mask']:
            value = config['jira_mask'][item]
            print(config['jira_mask'][item])
            if isinstance(value, str):
                mask = df_jira[df_jira[item] != value]
                df_jira = df_jira[df_jira[item] != value].copy()
            if isinstance(value, list):
                mask = df_jira[~df_jira[item].isin(value)]
                df_jira = df_jira[df_jira[item].isin(value)].copy()
            elif value is None:
                mask = df_jira[~df_jira[item].isnull()]
                df_jira = df_jira[~df_jira[item].isnull()].copy()
        if entity == "Teams_Capacity_History":
            df_jira = JiraDataWrangler.__merge_teams_head_count(df_head_cnt=df_jira, df_team=team_df)

    # Merging fields setup from 'merge_fields' config parameter
    if 'merge_fields' in config and config['merge_fields']:
        # for field in config['merge_fields']:
        for key, value in config['merge_fields'].items():
            try:
                if key not in df_jira.columns and value in df_jira.columns:
                    df_jira[key] = None
                if pd.Series([key, value]).isin(df_jira.columns).all():
                    mask = (df_jira[key].isnull() & ~df_jira[value].isnull())
                    df_jira.loc[mask, key] = df_jira.loc[mask, value]
                else:
                    logging.error(
                        f" --- Dictionary on Column '{value}', doesn't have '{key}' element, validate "
                        f"'merge_fields' parameter on config file for {config['entity']}")

            except KeyError:
                logging.error(
                    f" --- Dictionary on Column '{value}', doesn't have '{key}' element, validate 'merge_fields'"
                    f" parameter on config file for {config['entity']}")

    # Apply filtering of records from filter_by parameter, eg: issuetype
    if 'filter_by' in config and config['filter_by']:
        for key, value in config['filter_by'].items():
            try:
                if key in df_jira.columns:
                    df_jira = df_jira.drop(df_jira[(df_jira[key] != value)].index)
                else:
                    logging.error(
                        f" --- Dictionary on Column '{value}', doesn't have '{key}' element, validate "
                        f"'filter_by' parameter on config file for {config['entity']}")
            except KeyError:
                logging.error(f" --- Dictionary on Column '{value}', doesn't have '{key}' element, validate "
                              f"'filter_by' parameter on config file for {config['entity']}")

    # Dropping columns that are not in dictionary AND adding missing ones
    df_jira = df_jira[df_jira.columns.intersection(dictionary)]
    col_list = list(set().union(df_jira.columns, dictionary))
    df_jira = df_jira.reindex(columns=col_list, fill_value=None)
    df_jira = df_jira[dictionary]

    # Formatting Datetime columns
    if 'date_fields' in config and config['date_fields']:
        date_columns = ''.join(config['date_fields']).split(",")
        date_columns = [x.title() for x in date_columns] if date_columns else []
        for col in date_columns:
            try:
                if col in df_jira.columns:
                    df_jira[col] = pd.to_datetime(df_jira[col], utc=True, errors='coerce')
                    df_jira[col] = pd.to_datetime(df_jira[col], format=config['jira_date_format'])
                    df_jira[col] = df_jira[col].dt.tz_localize(None)

                else:
                    logging.error(f" --- DataFrame doesn't have '{col}' column (date type), validate "
                                  f"'date_fields' parameter on config file for {config['entity']}")
            except KeyError:
                logging.error(f" --- DataFrame doesn't have '{col}' column (date type), validate "
                              f"'date_fields' parameter on config file for {config['entity']}")

        # Fill Null Start Date with Create date if item has End Date
        if config['entity'] in ('Userstories', 'Epics') and \
                pd.Series(["Start_Date", "Created", "End_Date"]).isin(df_jira.columns).all():
            mask = df_jira['Start_Date'].isnull() & ~df_jira['End_Date'].isnull()

        # Set null Completed Dates as End date if Ticket is completed
        if config['entity'] == 'Userstories' and pd.Series(["Completed", "End_Date"]).isin(
                df_jira.columns).all():
            mask = df_jira['Completed'].isnull() & ~df_jira['End_Date'].isnull()
            df_jira.loc[mask, 'Completed'] = df_jira.loc[mask, 'End_Date']

    # Pre-Populate Fields from 'jira_pre_populate' config Parameter
    if 'jira_pre_populate' in config:
        for key, value in config['jira_pre_populate'].items():
            try:
                if key in df_jira.columns:
                    mask = df_jira[key].isnull()
                    df_jira.loc[mask, key] = value
                else:
                    logging.error(f" --- Dataframe column '{key}' doesn't have exist '{value}' element, validate "
                                  f"'merge_fields' parameter on config file for {config['entity']}")
            except KeyError:
                logging.error(f" --- Dictionary on Column '{key}', doesn't have '{value}' element, validate "
                              f"'merge_fields' parameter on config file for {config['entity']}")

    # Formatting integer fields
    int_cols = ''.join(config['int_fields']).split(",")
    int_cols = [x.title() for x in int_cols]
    df_jira.update(df_jira[int_cols].fillna(0))

    df_jira[int_cols] = df_jira[int_cols].astype('float')
    df_jira[int_cols] = df_jira[int_cols].round().astype('Int64')

    # Converting list into string values
    if 'list_to_string' in config and config['list_to_string']:
        list_2_str = ''.join(config['list_to_string']).split(",")
        list_2_str = [x.title() for x in list_2_str] if list_2_str else []
        for value in list_2_str:
            mask = (~df_jira[value].isnull())
            temp = pd.DataFrame(df_jira.loc[mask, value])
            temp[value] = temp.explode(value)
            temp[value] = temp[value].str.replace('[ (@*&?].*[)@*&?]', '', regex=True)
            df_jira[value] = temp[value]

    df_jira = ut.set_debugging_fields(df=df_jira, start_tst=start, config=config)

    # Processing Historical Updates into another dataframe
    if config["entity"] == "Teams":
        df_jira['Progressing_Statuses'] = df_jira['Progressing_Statuses'].apply(lambda x: x.title())
        df_jira['Statuses'] = df_jira['Statuses'].apply(lambda x: x.title())

    # Processing Historical Updates into another dataframe
    if config["entity"] == "Userstories" and 'Changelog.Histories' in df_jira.columns:
        df_hist = df_jira.copy()
        df_hist = df_hist.explode('Changelog.Histories').reset_index().drop("index", 1)
        df_hist = pd.concat([df_hist.drop(['Changelog.Histories'], axis=1),
                             df_hist['Changelog.Histories'].apply(pd.Series)], axis=1)

        # Explode the dictionary column
        df_hist = df_hist.explode('items').reset_index().drop("index", 1)
        df_hist = df_hist.reset_index(drop=True)
        df_hist = pd.concat([df_hist.drop(['items'], axis=1), pd.json_normalize(df_hist['items'])], axis=1)

        # Getting Sprint Update Historicals
        df_sprint_update: df_hist = df_hist.loc[df_hist['field'] == 'Sprint'].copy()

        # Getting State Transition historicals
        df_hist = df_hist[df_hist['field'] == 'status']

        et = tm.time()
        run_time = round(((et - st) / 60), 1)
        run = f"{run_time} minutes" if run_time >= 1 else f"{round(et - st, 1)} seconds"
        logging.info(f"\n== JIRA '{config['entity']}' processed prior Historical Transformation, "
                     f"({len(df_hist)} records - Total run: {run}).\n{ut.get_separator(level=3)}")

        df_hist = JiraDataWrangler.__process_state_transitions(df_hist=df_hist,
                                                               team_df=team_df.copy(),
                                                               config=config)

        # Group df by Cycle Time calculated and calculate the sum of 'Cycle Time' from Historicals
        total_ct = df_hist.groupby('Id')['Cycle_Time'].sum().reset_index()
        total_ct['Cycle_Time'] = total_ct['Cycle_Time'].astype('float')
        total_ct['Cycle_Time'].round(2)
        df_jira = df_jira.merge(total_ct, on='Id', how='left')
        df_jira['Cycletime'] = df_jira['Cycle_Time']
        # Selecting Start Date out of Historicals
        filtered_start_dates = df_hist[df_hist['Start_Date'].notna()]
        min_start_date = filtered_start_dates.groupby('Id')['Start_Date'].min().reset_index()
        min_start_date.rename(columns={'Start_Date': 'Start_Date_Hist'}, inplace=True)
        df_jira = df_jira.merge(min_start_date, on='Id', how='left')
        df_jira['Start_Date'] = df_jira['Start_Date_Hist']
        # If USs is finished but Start data is Null, we'll add it as End Date
        mask = df_jira['Start_Date'].isnull() & ~df_jira['End_Date'].isnull()
        df_jira.loc[mask, 'Start_Date'] = df_jira.loc[mask, 'End_Date']

        # Adding title case to All Columns
        df_hist.columns = [x.title() for x in df_hist.columns]
        dictionary = ''.join(config['dictionary_hist']).split(",")
        dictionary = [x.title() for x in dictionary]
        df_hist = df_hist[df_hist.columns.intersection(dictionary)]
        col_list = list(set().union(df_hist.columns, dictionary))
        df_hist = df_hist.reindex(columns=col_list, fill_value=None)
        df_hist = df_hist[dictionary]
        df_hist = ut.set_debugging_fields(df=df_hist, start_tst=start, config=config,
                                          file_name="uki_tp_uss_state_transitions_SUBFIX.csv")

        # Dropping Extra columns added on main USs Jira Dataframe related to Historical records
        df_jira = df_jira.drop([col for col in ''.join(config['us_drop_hist_fields']).split(",") if col in
                                df_jira.columns], axis=1)

    temp_fields = [x.title() for x in ''.join(config['temp_fields']).split(",")] if 'temp_fields' in config else []
    if len(temp_fields) > 0:
        df_jira = df_jira.drop([col for col in temp_fields if col in df_jira.columns], axis=1)

    # Setting Date_SKs fields, used to link dim_date/dim_date_range tables (Improve performance on RS queries)
    if 'date_sks' in config and config['date_sks']:
        date_sks = ''.join(config['date_sks']).split(",") if config['date_sks'] else None
        date_sks = [x.title() for x in date_sks] if date_sks else []
        for col in date_sks:
            new_col = f"{col}_Sk"
            if not df_jira[col].isnull().values.all():
                df_jira[new_col] = pd.to_datetime(df_jira[col], format=config['jira_date_format']).dt.strftime(
                    config['date_sks_format']).astype('Int64', errors='ignore')
            else:
                df_jira[new_col] = None
            df_jira[new_col] = df_jira[new_col].fillna(0)

    # Calculating Measures Leadtime, Cycletime, etc
    if 'time_range_measures' in config and config['time_range_measures']:
        for field in config['time_range_measures']:
            if ("Start" and "End") in config['time_range_measures'][field]:
                start_date = config['time_range_measures'][field]["Start"]
                end_date = config['time_range_measures'][field]["End"]
                maskDaterange = (df_jira[start_date].notnull() & df_jira[end_date].notnull())
                tempDF: df_jira = df_jira.loc[maskDaterange].copy()
                tempDF[start_date] = tempDF[start_date].dt.normalize()
                tempDF[end_date] = tempDF[end_date].dt.normalize()
                tempDF[field] = np.busday_count(
                    pd.to_datetime(tempDF[start_date]).values.astype('datetime64[D]'),
                    pd.to_datetime(tempDF[end_date]).values.astype('datetime64[D]')) + 1
                df_jira[field] = tempDF.reset_index(drop=True)[field]

                if field == "Cycletime" and 'Completed' in df_jira.columns and "End_Prev" \
                        in config['time_range_measures'][field]:
                    end_date = config['time_range_measures'][field]["End_Prev"]
                    maskDaterange = (
                            (df_jira[start_date].notnull() & df_jira[end_date].notnull()) & (
                            df_jira["Is_Native"] is True))
                    tempDF: df_jira = df_jira.loc[maskDaterange].copy()
                    if len(tempDF) > 0:
                        tempDF[start_date] = tempDF[start_date].dt.normalize()
                        tempDF[end_date] = tempDF[end_date].dt.normalize()
                        tempDF[field] = np.busday_count(
                            pd.to_datetime(tempDF[start_date]).values.astype('datetime64[D]'),
                            pd.to_datetime(tempDF[end_date]).values.astype('datetime64[D]')) + 1
                        df_jira[field] = tempDF[field]

    df_jira = ut.order_by(df=df_jira, config=config)
    # get the end time
    et = tm.time()
    run_time = round(((et - st) / 60), 1)
    run = f"{run_time} minutes" if run_time >= 1 else f"{round(et - st, 1)} seconds"
    logging.info(f"\n== JIRA '{config['entity']}' data wrangled, ({len(df_jira)} "
                 f"records - Total run: {run}).\n{ut.get_separator(level=3)}")
    if (isinstance(df_hist, pd.DataFrame) or isinstance(df_hist, pd.Series)) and \
            (isinstance(df_sprint_update, pd.DataFrame) or isinstance(df_sprint_update, pd.Series)):
        return df_jira, df_hist.copy(), df_sprint_update.copy()
    else:
        return df_jira, df_hist, df_sprint_update
