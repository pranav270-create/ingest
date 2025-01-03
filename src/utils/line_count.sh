#!/bin/bash

#inspired by https://gist.github.com/ezamelczyk/78b9c0dd095f8706a3f6a41e8eae0afd

if [ -z "$1" ]; then
    # Use current month
    set -- "$(date +%b)"
fi

if [ -z "$2" ]; then
    # Use current year
    set -- "$1" "$(date +%Y)"
fi

echo "$1 $2"

# Function to get contributions for a specific month
get_contributions() {
    local month=$1
    local year=$2
    
    case $(uname -s) in
    Darwin*)
        pm=$(date -v-1m -jf "%b %Y" "$month $year" +%m)
        pms=$(date -v-1m -jf "%b %Y" "$month $year" +%b)
        ld_pm=$(date -v-1d -jf "%b 1 %Y" "$month 1 $year" +%d)
        nm=$(date -v+1m -jf "%b %Y" "$month $year" +%m)
        ;;
    *)
        pm=$(date -d "$month 1, $year -1 month" +%m)
        pms=$(date -d "$month 1, $year -1 month" +%b)
        ld_pm=$(date -d "$month 1, $year -1 day" +%d)
        nm=$(date -d "$month 1, $year +1 month" +%m)
        ;;
    esac

    after_year=$year
    before_year=$year
    if [ "$pm" = "12" ]; then
        ((after_year--))
    fi
    if [ "$nm" = "01" ]; then
        ((before_year++))
    fi
    
    filter="--after=${after_year}-${pm}-${ld_pm} --before=${before_year}-${nm}-01"
    
    git log --shortstat $filter -- '*.py' '*.ts' '*.js' '*.jsx' | grep -E "(Author: )(\b\s*([a-zA-Z]\w+)){1,2}|fil(e|es) changed" | awk -v month="$month" -v year="$year" '
    {
        if($1 ~ /Author/) {
            author = $2" "$3
        }
        else {
            files[author]+=$1
            inserted[author]+=$4
            deleted[author]+=$6
            delta[author]+=$4-$6
        }
    }
    END { 
        for (key in files) { 
            print month, year, key, files[key], inserted[key], deleted[key], delta[key]
        } 
    }'
}

# Get contributions for the last 6 months
current_month=$(date +%m)
current_year=$(date +%Y)

echo "Month Year Author Files_Changed Lines_Inserted Lines_Deleted Lines_Delta"
for i in {0..5}
do
    if [ "$(uname -s)" = "Darwin" ]; then
        month=$(date -v-${i}m +%b)
        year=$(date -v-${i}m +%Y)
    else
        month=$(date -d "$current_year-$current_month-01 -$i month" +%b)
        year=$(date -d "$current_year-$current_month-01 -$i month" +%Y)
    fi
    get_contributions $month $year
done | sort -k3,3 -k1,1M -k2,2n

echo "" ; echo "Generating overall contribution log ..." ; echo ""

if [ "$1" != "all" ]; then
    # Filter by month and year
    filter="--after=$2-$pm-$ld_pm --before=$before_year-$nm-01"
    msg="for $1 $2"
else
    # Get all logs
    filter=""
    msg="as of $(date "+%b %Y")"
fi

echo "Overall contributions $msg:"
git log --shortstat $filter -- '*.py' '*.ts' '*.js' '*.jsx' | grep -E "(Author: )(\b\s*([a-zA-Z]\w+)){1,2}|fil(e|es) changed" | awk '
{
    if($1 ~ /Author/) {
        author = $2" "$3
    }
    else {
        files[author]+=$1
        inserted[author]+=$4
        deleted[author]+=$6
        delta[author]+=$4-$6
    }
}
END { for (key in files) { print "Author: " key "\n\tfiles changed: ", files[key], "\n\tlines inserted: ", inserted[key], "\n\tlines deleted: ", deleted[key], "\n\tlines delta: ", delta[key] } }
'

git ls-files | grep -v '.json' | grep -v '.csv' | grep -v '.txt' | xargs cloc

git diff --shortstat `git hash-object -t tree /dev/null`