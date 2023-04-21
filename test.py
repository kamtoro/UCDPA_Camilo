import base64
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, time, date, timedelta
# from business_duration import businessDuration
from concurrent.futures import ThreadPoolExecutor, as_completed
from business_duration import businessDuration
import holidays as pyholidays

import pytz
from dateutil.relativedelta import relativedelta
from jira import JIRA, JIRAError
import time as tm
import config.retry_decorator as retrydec
import config.utils as ut

logger = logging.getLogger(__name__)


class JiraDataWrangler:

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

    @staticmethod
    def business_hours(start=None, end=None, tz='Europe/London', starttime=8, endtime=18):
        # UK public holidays
        UK_holiday_list = pyholidays.UK()
        # Business open hour must be in standard python time format-Hour,Min,Sec
        biz_open_time = time(starttime, 0, 0)
        # Business close hour must be in standard python time format-Hour,Min,Sec
        biz_close_time = time(endtime, 0, 0)

        #
        # start_date = start.replace(tzinfo=None)
        # end_date = end.replace(tzinfo=None)

        tz_delta = timedelta(hours=pytz.timezone(tz).utcoffset(datetime.now()).total_seconds() / 3600)
        start_date = start + tz_delta
        end_date = end + tz_delta

        # hours = businessDuration(start_date, end_date).seconds // 3600
        hours = businessDuration(startdate=start_date,
                                 enddate=end_date,
                                 starttime=biz_open_time,
                                 endtime=biz_close_time,
                                 weekendlist=[5, 6],
                                 holidaylist=UK_holiday_list,
                                 unit='hour'
                                 )

        return round(hours, 2)

    @staticmethod
    def __find_ultimate_parents(df: pd.DataFrame = None, entity=None, child_col=None, parent_col=None,
                                ultimate_col=None, copy_cols=[]) -> pd:
        """
        @param child_col: Field from data set to be used as the Indexing field to search parents.
        @param parent_col: Field from data set to be used to find ultimate parents.
        @param ultimate_col: The ultimate parent found from the dataframe.
        """
        if not ultimate_col:
            ultimate_col = 'ultimate_parent'
        # Make a copy of df, using 'id' as the index so we can lookup parent ids
        df2 = df.set_index(df[child_col]).copy()
        df2[ultimate_col] = df2[parent_col]

        # Next-parent-2 not null - fake it for now
        np2nn = df2[ultimate_col].notnull()
        while np2nn.any():
            # Lookup df2[parent-id], since the index is now by id. Get the
            # parent-id (of the parent-id), put that value in nextpar2.
            # So basically, if row B.nextpar has A, nextpar2 has (parent-of-A), or Nan.

            # Set na_action='ignore' so any Nan doesn't bother looking up, just copies
            # the Nan to the next generation.
            df2['nextpar2'] = df2[ultimate_col].map(df2[parent_col], na_action='ignore')
            # for col in copy_cols:
            #     df2[f'{col}_temp'] = df2[col].map(df2[parent_col], na_action='ignore')

            # df2[[copy_cols]] = df2[[copy_cols]].map(df2[parent_col], na_action='ignore')

            # Re-evaluate who is a Nan in the nextpar2 column.
            np2nn = df2['nextpar2'].notnull()

            # Only update nextpar from nextpar2 if nextpar2 is not a Nan. Thus, stop
            # at the root.
            df2.loc[np2nn, ultimate_col] = df2[np2nn]['nextpar2']
            # for col in copy_cols:
            #     df2.loc[np2nn, ultimate_col] = df2[np2nn]['nextpar2']
            #     df2.loc[np2nn, col] = df2[np2nn][f'{col}_temp']

        # At this point, we've run out of parents to look up. df2[ultimate_col] has
        # the "ultimate" parents.

        df = df2.copy()
        # df = df.reset_index()
        df = df.reset_index(level=0, drop=True).reset_index()
        return df

    @staticmethod
    def __process_state_transitions(df_hist: pd.DataFrame = [], team_df: pd.DataFrame = None, config=None) -> pd:
        """
        Description: Process USs Historicals data retrieved
            @param:df_hist: Dataframe (pandas): dataframe which contains data from entity.
            @param:config: Config file to retrieve parameters configured
        entity:
            Entity to extract (str) the Tribe from.
        Returns:
            Pandas Dataframe
        """
        import csv
        curr_row = modify_date = prev_state_list = rework_transition = prev_row = next_state = next_team_state = \
            next_team_state_id = next_state_priority = next_team_state_priority = next_state_id = next_state_id \
            = create_date = None

        with open('State_transition_not_in_progressing_statuses.csv', 'w', encoding='utf-8') as fd:
            fd.write(f"Key,Modify_Date,Prev_State,Next_State,Team_Key,Statuses,Progressing_Statuses")

        with open('State_transition_not_in_statuses.csv', 'w', encoding='utf-8') as fd:
            fd.write(f"Key,Modify_Date,Prev_State,Next_State,Team_Key,Statuses,Progressing_Statuses")

        timezone = {
            'UK': 'Europe/London',
            'PT': 'Europe/Lisbon',
            'RO': 'Europe/Bucharest',
            'CO': 'Europe/Dublin',
            'MT': 'Europe/Malta',
            'BG': 'Europe/Sofia',
            'EXT': 'Europe/London',
            'Unknown': 'Europe/London',
            'WW': 'Europe/London'
        }
        is_rework = False
        min_duration = 30

        day_start = config['day_start']
        day_end = config['day_end']

        # Dropping unused fields
        df_hist = df_hist.drop([col for col in ''.join(config['drop_hist_fields']).split(",") if col in
                                df_hist.columns], axis=1)
        df_hist.drop(0, inplace=True, axis=1)
        # Renaming exploded fields
        fields_mapping = dict((k, v) for k, v in config['hist_fields_mapping'].items())
        df_hist.rename(columns=fields_mapping, inplace=True)

        # Convert to datetime objects and align to UTC
        df_hist['Modify_Date'] = pd.to_datetime(df_hist['Modify_Date'], utc=True)
        df_hist['Modify_Date'] = df_hist['Modify_Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_hist['Modify_Date'] = pd.to_datetime(df_hist['Modify_Date'])  # Reconverting series to Timestamp

        df_hist = ut.order_by(df=df_hist, config=config, field='hist_order_by')

        # iterate through each row and select each transition update
        raw_df = df_hist.copy()  # To be deleted
        # df_dic = df_hist.to_dict('records')

        df_hist['Duration_In_State'] = None
        df_hist['Rework_Duration'] = 0
        df_hist['Is_Rework'] = False
        df_hist['Cycle_Time'] = None
        # Stripping any non alphanumeric character from Statuses Columns
        team_df['Statuses'] = team_df['Statuses'].astype(str)
        team_df['Progressing_Statuses'] = team_df['Progressing_Statuses'].astype(str)
        # Remove any non-alphabetic characters from the start or end of the 'Name' column
        team_df['Statuses'] = team_df['Statuses'].str.replace(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '',
                                                              regex=True)
        team_df['Progressing_Statuses'] = team_df['Progressing_Statuses'].str.replace(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '',
                                                                                      regex=True)
        team_df['Statuses'] = team_df['Statuses'].str.title()
        team_df['Progressing_Statuses'] = team_df['Progressing_Statuses'].str.title()
        df_hist['Prev_State'] = df_hist['Prev_State'].str.title()
        df_hist['Next_State'] = df_hist['Next_State'].str.title()
        team_df.rename(columns={'Key': 'Temp_Key'}, inplace=True)
        df_hist = df_hist.merge(team_df[['Temp_Key', 'Statuses', 'Progressing_Statuses', 'Location']],
                                left_on='Team_Key', right_on='Temp_Key', how="left")
        iid = None
        ticket_start = None
        for idx, row in df_hist.iterrows():
            if iid is None or iid != row['Id']:
                start = df_hist.at[idx, 'State_Changed_From'] = row['Created']
                iid = row['Id']
                ticket_start = None
            else:
                df_hist.at[idx, 'State_Changed_From'] = start = modify_date
            modify_date = row['Modify_Date']
            location = row['Location']
            workflow_statuses = ''.join(row['Statuses']).split(",") if 'Statuses' in row else None
            progressing_statuses = ''.join(row['Progressing_Statuses']).split(",") if 'Progressing_Statuses' in row \
                else None
            tz = timezone.get(location, 'Europe/London')
            df_hist.at[idx, 'Duration_In_State'] = JiraDataWrangler.business_hours(start=start,
                                                                                   end=modify_date,
                                                                                   tz=tz,
                                                                                   starttime=day_start,
                                                                                   endtime=day_end)

            row['State_Transition'] = f"{row['Prev_State']} >> {row['Next_State']}"
            if row['Prev_State'] in workflow_statuses and row['Next_State'] in workflow_statuses:
                prev_state_index = workflow_statuses.index(row['Prev_State']) if row['Prev_State'] \
                                                                                 in workflow_statuses else 0
                next_state_index = workflow_statuses.index(row['Next_State']) if row['Next_State'] \
                                                                                 in workflow_statuses else 0

                if next_state_index < prev_state_index:
                    df_hist.at[idx, 'Rework_Duration'] = df_hist.at[idx, 'Duration_In_State']
                    row['State_Transition'] = f"{row['Prev_State']} >> {row['Next_State']} (RW)"
                    row['Is_Rework'] = True

            if row['Prev_State'] in progressing_statuses:
                df_hist.at[idx, 'Cycle_Time'] = JiraDataWrangler.business_hours(start=start,
                                                                                end=modify_date,
                                                                                tz=tz,
                                                                                starttime=config['day_start'],
                                                                                endtime=config['day_end'])

                df_hist.at[idx, 'Start_Date'] = modify_date if ticket_start is None else ticket_start

            if row['Prev_State'] not in workflow_statuses or row['Next_State'] not in workflow_statuses:
                with open('State_transition_not_in_progressing_statuses.csv', 'a', encoding='utf-8') as fd:
                    fd.write(f"\n{row['Key']},{modify_date},{row['Prev_State']},{row['Next_State']},"
                             f"{row['Team_Key']},\"{workflow_statuses}\",\"{progressing_statuses}\"")

            if row['Prev_State'] not in progressing_statuses or row['Next_State'] not in progressing_statuses:
                with open('State_transition_not_in_statuses.csv', 'a', encoding='utf-8') as fd:
                    fd.write(f"\n{row['Key']},{modify_date},{row['Prev_State']},{row['Next_State']},"
                             f"{row['Team_Key']},\"{workflow_statuses}\",\"{progressing_statuses}\"")
            start = df_hist.at[idx, 'State_Changed_To'] = modify_date

        return df_hist

    @staticmethod
    def __merge_teams_head_count(df_head_cnt: pd.DataFrame = [], df_team: pd.DataFrame = []) -> pd:
        """
        Description:
            @param:df_head_cnt: Dataframe (pandas): dataframe which contains data from entity.
            @param:df_team DF Teams at SBG
        Returns:
            Pandas Dataframe

        """
        # if entity == "Initiative":
        count_users = df_head_cnt['To_Key'].nunique()
        count_teams = df_head_cnt['Project_Team_Key'].nunique()
        logging.debug(f"Count Users: {count_users}/ Count Teams: {count_teams}")
        head_count = round((count_users / count_teams) * 2) / 2
        df_team[['Team_Size', 'Team_Size_M1', 'Team_Size_M2', 'Team_Size_M3']] = head_count
        df_team['Month_History'] = datetime.today().replace(day=1) - relativedelta(months=0)
        df_team['Month_History'] = pd.to_datetime(df_team['Month_History'].values.astype('datetime64[M]'))

        i = 0
        temp_df = df_team.copy()
        # Adding Dataset for the next 3 months
        while i < 3:
            i += 1
            temp_temp = temp_df.copy()
            temp_temp['Month_History'] = temp_temp['Month_History'].apply(lambda x: x + relativedelta(months=i))
            df_team = pd.concat([df_team, temp_temp])

        return df_team.copy()

    @staticmethod
    def __retrieve_parents(data=None, entity=None) -> list:
        # Retrieving issues' parent epics (Projects) from Issue_Links
        row = None
        idx = None

        with open('parent_entities_non_epics.txt', 'w', encoding='utf-8') as fd:
            fd.write(f"Parent entities not found as epics")

        if entity == 'Epics' and isinstance(data, list):
            # try:
            # for row in data:
            try:
                for idx, row in enumerate(data):
                    # Gets Epic parent link
                    issue_type = row['fields']['issuetype']['name']
                    # print(f"{row['id']}: {issue_type}; ({row['fields']['issuelinks']})")
                    if 'issuelinks' in row['fields']:
                        for link in row['fields']['issuelinks']:
                            if 'outwardIssue' in link:
                                if link['type']['outward'] == 'child of':
                                    current_key = link['outwardIssue']['key']
                                    if current_key.startswith('FP'):
                                        # row['epic_parent'] = current_key
                                        data[idx]['outwardIssue.id'] = link['outwardIssue']['id']
                                        data[idx]['outwardIssue.key'] = link['outwardIssue']['key']
                                        data[idx]['outwardIssue.summary'] = link['outwardIssue']['fields']['summary']
                                        data[idx]['outwardIssue.status'] = link['outwardIssue']['fields']['status'] \
                                            ['name']
                                        data[idx]['outwardIssue.status_category'] = link['outwardIssue']['fields'] \
                                            ['status']['statusCategory']['name']
                                        data[idx]['outwardIssue.issuetype'] = link['outwardIssue']['fields'] \
                                            ['issuetype']['name']
                                        break
                                    else:
                                        if link['outwardIssue']['fields']['issuetype']['name'] in ('Epic',
                                                                                                   'New Work Request'):
                                            data[idx]['parent_epic.id'] = link['outwardIssue']['id']
                                            data[idx]['parent_epic.key'] = link['outwardIssue']['key']
                                            data[idx]['parent_epic.summary'] = link['outwardIssue']['fields']['summary']
                                            data[idx]['parent_epic.status'] = link['outwardIssue']['fields']['status'] \
                                                ['name']
                                            data[idx]['parent_epic.status_category'] = link['outwardIssue']['fields'] \
                                                ['status']['statusCategory']['name']

                                            data[idx]['parent_epic.status_category'] = link['outwardIssue']['fields'] \
                                                ['status']['statusCategory']['name']

                                            # Added on 30/12/22
                                            data[idx]['outwardIssue.id'] = link['outwardIssue']['id']
                                            data[idx]['outwardIssue.key'] = link['outwardIssue']['key']
                                            data[idx]['outwardIssue.summary'] = link['outwardIssue']['fields'][
                                                'summary']
                                            data[idx]['outwardIssue.status'] = link['outwardIssue']['fields']['status'][
                                                'name']
                                            data[idx]['outwardIssue.status_category'] = link['outwardIssue']['fields'] \
                                                ['status']['statusCategory']['name']
                                            data[idx]['outwardIssue.issuetype'] = \
                                                link['outwardIssue']['fields']['issuetype'] \
                                                    ['name']

                                        else:
                                            with open('parent_entities_non_epics.txt', 'a') as fd:
                                                fd.write(f"\n{link['outwardIssue']['key']}: Parent issue type ("
                                                         f"{link['outwardIssue']['key']} - "
                                                         f"{link['outwardIssue']['fields']['issuetype']['name']})")


            except:
                logging.error(f"No data retrieved for this type entity \"{row}{idx}\" ******",
                              exc_info=True)
            # except KeyError:
            #     logging.WARN(f" --- Dictionary on Col '{value}', doesn't have '{key}' element, validate "
            #                  f"'merge_fields' parameter on config file for {kwargs['entity']}")
        return data

    @staticmethod
    def extract_from_jira(**kwargs):
        wrapper_all_exceptions = retrydec.retry(Exception, total_tries=6, initial_wait=1, backoff_factor=3,
                                                logger=logger)
        wrapped_request_func = wrapper_all_exceptions(jira_data_request)
        return wrapped_request_func(**kwargs)

    @staticmethod
    def filtering_default_values(df: pd.DataFrame = [], config=None, parameter=None, field=None, field_filtered=None) \
            -> pd:
        """
            This function is meant to set default values using conditions to slice data and set default values.
            An example of this is, for Jira and effort null on Tickets, certain issue types must have different default
            effort, eg: Help issue type must have 0.5 SP and Support issue types, 1 SP
            @param df: Pandas Dataframe to update
            @param config: Config file received
            @param parameter: Parameter from config file to extract filter criteria and default values
            @param field: Field from DF to set default values
            @param field_filtered: Is simply, what field will be added on mask to filter values, eg, Issue_Type: Key ==
            Help
        Returns:
            Pandas Dataframe
        """
        if parameter and parameter in config:
            for key, value in config[parameter].items():
                try:
                    if field in df.columns and field_filtered in df.columns:
                        df[field] = np.where((df[field].isnull() | df[field].isna()) & df[field_filtered].eq(key),
                                             value, df[field])
                    else:
                        logging.error(f" --- Dictionary on Column '{key}', doesn't have '{value}' element, validate "
                                      f"{parameter} parameter on config file for {config['entity']}")
                except KeyError:
                    logging.error(f" --- Dictionary on Column '{key}', doesn't have '{value}' element, validate "
                                  f"{parameter} parameter on config file for {config['entity']}")
        return df

    @staticmethod
    def merge_parent_entity(df: pd.DataFrame = [], parent_df: pd.DataFrame = [], cols=[], left_col=None,
                            right_col=None, prefix=None, merge_cols=False, force=False) -> pd:
        """
        This method is simply to Merge parent entities's data with the entity dataset being transformed,
        Eg: Projects -> Epics; USs -> SubTask; Epics -> USs
        @param df: Main Dataframe from Child entity being processed.
        @param parent_df: Parent's dataframe to get data to be merged with the main DF.
        @param cols: Columns to be merged, these must exist with same name in both Dataframes.
        @param left_col: Column key from Main entity (left) to search data on Parent.
        @param right_col: Column PK from Parent's DF.
        @param prefix: Prefix to be set on columns to be merged.
        """

        columns = None
        # if 'project_cols' in kwargs and kwargs['project_cols']:
        if cols and len(df) and len(parent_df):
            columns = ''.join(cols).split(",")
            columns = [x.title() for x in columns] if columns else []
            parent_df.drop(parent_df.columns.difference(columns), axis=1, inplace=True)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            parent_df = parent_df.sort_values(by=right_col.title(), ascending=True)
        else:
            logger.error("merge_parent_entity: parent_df received, isn't a Dataframe.")
            return
        prefix = prefix.title() if prefix else 'Parent_'
        parent_df = parent_df.add_prefix(prefix)
        df = df.merge(parent_df, left_on=left_col.title(), right_on=f'{prefix}{right_col.title()}', how="left")
        if merge_cols:
            if right_col in columns:
                columns.remove(f'{right_col}')
            for col in columns:
                if col in df.columns:
                    mask = (~df[f'{prefix}{right_col}'].isnull()) if force else \
                        (df[col].isnull() & ~df[f'{prefix}{right_col}'].isnull())
                    df.loc[mask, col] = df.loc[mask, f'{prefix}{col}']
        return df


def jira_data_request(*args, **kwargs):
    # get the start time
    st = tm.time()

    jira = get_jira_client(kwargs['connection'])
    kwargs['startAt'] = kwargs['startAt'] if 'startAt' in kwargs else 0

    fromRow = 0
    maxResults = kwargs['maxResults'] if 'maxResults' in kwargs else None
    issues = []
    queries = []
    jira.projects()
    expand = None if 'expand' not in kwargs else kwargs['expand']
    if 'list_projects' in kwargs and '{project}' in kwargs['jql']:
        projects = ''.join(kwargs['list_projects']).split(",")
        for project in projects:
            jql = kwargs['jql']
            queries.append(jql.replace("{project}", project))
    elif 'jql' in kwargs and kwargs['jql']:
        queries = [kwargs['jql']]
    elif kwargs['entity'] == 'Jira_Team_Names':
        for project in jira.projects():
            issues.append({'Key': project.key, 'Name': project.name})
        return issues


    # Initialize ThreadPoolExecutor
    num_workers = 15  # Adjust this value based on your desired level of parallelism
    executor = ThreadPoolExecutor(max_workers=num_workers)

    # Create a list of futures
    futures = []
    for query in queries:
        future = executor.submit(fetch_issues, query, jira, expand, kwargs)
        futures.append(future)

    # Collect the results as they complete
    for future in as_completed(futures):
        issues += future.result()

    # get the end time
    et = tm.time()
    run_time = round(((et - st) / 60), 1)
    run = f"{run_time} minutes" if run_time >= 1 else f"{round(et - st, 1)} seconds"
    logging.info(f"\n== JIRA '{kwargs['entity']}' extraction completed, ({len(issues)} "
                 f"records - Total run: {run}).\n{ut.get_separator(level=3)}")
    return issues


def fetch_issues(query, jira, expand, kwargs):
    i = fromRow = 0
    maxResults = kwargs['maxResults'] if 'maxResults' in kwargs else None
    issues = []
    while True:
        i += 1
        response = jira.search_issues(jql_str=query,
                                      expand=expand,
                                      startAt=fromRow,
                                      maxResults=maxResults,
                                      fields=kwargs['fields'],
                                      json_result=True)

        issues += response['issues']
        fromRow = fromRow + maxResults
        if len(response['issues']) == 0 or fromRow > response['total']:
            query_up = query.upper()
            project = ""
            if 'PROJECT =' in query_up:
                project = f"Project {project}: {(query_up.split('PROJECT ='))[1].split(' AND')[0]} "
            log = f"  -- {project} Issues extracted {response['total']} - ({i} Loops done)"
            logging.debug(log)
            break
    return issues


def jira_data_request_old(*args, **kwargs):
    # get the start time
    st = tm.time()

    jira = get_jira_client(kwargs['connection'])
    kwargs['startAt'] = kwargs['startAt'] if 'startAt' in kwargs else 0

    fromRow = 0
    maxResults = kwargs['maxResults'] if 'maxResults' in kwargs else None
    issues = []
    queries = []
    jira.projects()
    expand = None if 'expand' not in kwargs else kwargs['expand']
    if 'list_projects' in kwargs and '{project}' in kwargs['jql']:
        projects = ''.join(kwargs['list_projects']).split(",")
        for project in projects:
            jql = kwargs['jql']
            queries.append(jql.replace("{project}", project))
    elif 'jql' in kwargs and kwargs['jql']:
        queries = [kwargs['jql']]
    elif kwargs['entity'] == 'Jira_Team_Names':
        for project in jira.projects():
            issues.append({'Key': project.key, 'Name': project.name})
        return issues
    for query in queries:
        try:
            i = 0
            fromRow = 0
            while True:
                i += 1
                response = jira.search_issues(jql_str=query,
                                              expand=expand,
                                              startAt=fromRow,
                                              maxResults=maxResults,
                                              fields=kwargs['fields'],
                                              json_result=True)

                issues += response['issues']
                fromRow = fromRow + maxResults
                if len(response['issues']) == 0 or fromRow > response['total']:
                    maxResults = kwargs['maxResults']

                    query_up = query.upper()
                    project = ""
                    if 'PROJECT =' in query_up:
                        project = f"Project {project}: {(query_up.split('PROJECT ='))[1].split(' AND')[0]} "
                    log = f"  -- {project} Issues extracted {response['total']} - ({i} Loops done)"
                    logging.debug(log)
                    break

        except JIRAError as e:
            logging.error("Jira query error with: {}\n{}".format(query, e))
            # return []

    # get the end time
    et = tm.time()
    run_time = round(((et - st) / 60), 1)
    run = f"{run_time} minutes" if run_time >= 1 else f"{round(et - st, 1)} seconds"
    logging.info(f"\n== JIRA '{kwargs['entity']}' extraction completed, ({len(issues)} "
                 f"records - Total run: {run}).\n{ut.get_separator(level=3)}")
    return issues


def get_jira_client(connection):
    jira_connection = None
    url = connection['jira_api_url']
    token = connection['authorization_basic']
    try:
        verify = connection['verify']
    except KeyError:  # Not found in yaml configuration file
        verify = True  # Default should be to verify the certificates to Jira server

    if token:
        username, password = base64.b64decode(token).decode('utf-8').split(':')
    else:
        username = connection['username']
        password = connection['password']

    # logging.DEBUG("Connecting to ", url)

    if username is None:
        raise KeyError(f'No Username or token provided on configuration file.')

    if password is None:
        raise KeyError(f'No password or token provided on configuration file.')

    params = {'access_token': token}
    if len(username + password) > 10:
        if connection['native']:
            response = requests.get(url=url, params=params)
        else:
            jira_connection = JIRA(options={'server': url, 'verify': verify}, basic_auth=(username, password))
    else:
        if connection['native']:
            response = requests.get(url=url, params=params)
        else:
            jira_connection = JIRA(options={'server': url, 'verify': verify})
    return jira_connection
