function isThanksgiving(date: Date, region: string) {
  const month = date.getMonth()
  const day = date.getDate()
  const weekday = date.getDay()
  return (region === 'US' && month === 10 && weekday === 4 && day >= 22 && day <= 28)
    || (region === 'CA' && month === 9 && weekday === 1 && day >= 8 && day <= 14)
}

export function greetingFor(date: Date, locale = 'en-US'): string {
  const day = date.getDate()
  const month = date.getMonth()
  const weekday = date.getDay()
  const hour = date.getHours()
  const region = new Intl.Locale(locale).maximize().region
  if (month === 0 && day === 1) return 'Happy New Year!'
  if (month === 9 && day === 31) return 'Happy Halloween!'
  if (month === 11 && day === 25) return 'Merry Christmas!'
  if (month === 11 && day === 31) return 'Happy New Year’s Eve!'
  if (isThanksgiving(date, region ?? '')) return 'Happy Thanksgiving!'
  if (day === 1) {
    const monthName = new Intl.DateTimeFormat('en', { month: 'long' }).format(date)
    return `Hello, ${monthName}.`
  }
  if (hour < 5 || hour >= 23) return 'Up late?'
  if (weekday === 5 && hour >= 12 && hour < 19) return 'Happy Friday.'
  if ((weekday === 0 || weekday === 6) && hour >= 9 && hour < 15) return 'Happy weekend.'
  if (weekday === 1 && hour >= 5 && hour < 12) return 'Happy Monday.'
  if (hour < 12) return 'Good morning.'
  if (hour < 18) return 'Good afternoon.'
  return 'Good evening.'
}
