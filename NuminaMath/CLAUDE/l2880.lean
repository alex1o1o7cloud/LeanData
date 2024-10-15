import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_a_l2880_288063

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 2

-- State the theorem
theorem solve_for_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 8 * x + 4 * a) → 
  deriv (f a) 2 = 20 → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l2880_288063


namespace NUMINAMATH_CALUDE_uncool_parents_count_l2880_288008

theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 12 := by
  sorry

#check uncool_parents_count

end NUMINAMATH_CALUDE_uncool_parents_count_l2880_288008


namespace NUMINAMATH_CALUDE_two_number_difference_l2880_288054

theorem two_number_difference (a b : ℕ) (h1 : b = 10 * a) (h2 : a + b = 23320) : b - a = 19080 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l2880_288054


namespace NUMINAMATH_CALUDE_rain_probability_l2880_288071

theorem rain_probability (p_monday p_tuesday p_neither : ℝ) 
  (h1 : p_monday = 0.7)
  (h2 : p_tuesday = 0.55)
  (h3 : p_neither = 0.35)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1)
  (h5 : 0 ≤ p_tuesday ∧ p_tuesday ≤ 1)
  (h6 : 0 ≤ p_neither ∧ p_neither ≤ 1) :
  p_monday + p_tuesday - (1 - p_neither) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l2880_288071


namespace NUMINAMATH_CALUDE_thursday_temperature_l2880_288087

/-- Calculates the temperature on Thursday given the temperatures for the other days of the week and the average temperature. -/
def temperature_on_thursday (sunday monday tuesday wednesday friday saturday average : ℝ) : ℝ :=
  7 * average - (sunday + monday + tuesday + wednesday + friday + saturday)

/-- Theorem stating that the temperature on Thursday is 82° given the specified conditions. -/
theorem thursday_temperature :
  temperature_on_thursday 40 50 65 36 72 26 53 = 82 := by
  sorry

end NUMINAMATH_CALUDE_thursday_temperature_l2880_288087


namespace NUMINAMATH_CALUDE_prime_power_sum_l2880_288005

theorem prime_power_sum (w x y z : ℕ) :
  3^w * 5^x * 7^y * 11^z = 2310 →
  3*w + 5*x + 7*y + 11*z = 26 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l2880_288005


namespace NUMINAMATH_CALUDE_problem_solution_l2880_288084

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 1) 
  (h2 : b - a = 4) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2880_288084


namespace NUMINAMATH_CALUDE_max_stickers_for_player_l2880_288033

theorem max_stickers_for_player (num_players : ℕ) (avg_stickers : ℕ) (min_stickers : ℕ) :
  num_players = 25 →
  avg_stickers = 4 →
  min_stickers = 1 →
  ∃ (max_stickers : ℕ), max_stickers = 76 ∧
    ∀ (player_stickers : ℕ),
      (player_stickers * num_players ≤ num_players * avg_stickers) ∧
      (∀ (i : ℕ), i < num_players → min_stickers ≤ player_stickers) →
      player_stickers ≤ max_stickers :=
by sorry

end NUMINAMATH_CALUDE_max_stickers_for_player_l2880_288033


namespace NUMINAMATH_CALUDE_min_distance_squared_l2880_288038

/-- The line on which point P(x,y) moves --/
def line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The distance function from point (x,y) to (2,2) --/
def distance_squared (x y : ℝ) : ℝ := (x - 2)^2 + (y - 2)^2

/-- Theorem stating the minimum value of the distance function --/
theorem min_distance_squared :
  ∃ (min : ℝ), min = 1/2 ∧ 
  (∀ x y : ℝ, line x y → distance_squared x y ≥ min) ∧
  (∃ x y : ℝ, line x y ∧ distance_squared x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2880_288038


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l2880_288056

/-- Proves that the initial volume of a mixture is 150 liters, given the conditions of the problem -/
theorem mixture_volume_proof (initial_water_percentage : Real) 
                              (added_water : Real) 
                              (final_water_percentage : Real) : 
  initial_water_percentage = 0.1 →
  added_water = 30 →
  final_water_percentage = 0.25 →
  ∃ (initial_volume : Real),
    initial_volume * initial_water_percentage + added_water = 
    (initial_volume + added_water) * final_water_percentage ∧
    initial_volume = 150 := by
  sorry


end NUMINAMATH_CALUDE_mixture_volume_proof_l2880_288056


namespace NUMINAMATH_CALUDE_contractor_absent_days_l2880_288060

/-- Proves that given the specified contract conditions, the number of absent days is 8 -/
theorem contractor_absent_days 
  (total_days : ℕ) 
  (daily_pay : ℚ) 
  (daily_fine : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 490) :
  ∃ (days_absent : ℕ), 
    days_absent = 8 ∧ 
    (daily_pay * (total_days - days_absent) - daily_fine * days_absent = total_amount) :=
by sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l2880_288060


namespace NUMINAMATH_CALUDE_equivalent_discount_l2880_288083

/-- Proves that a single discount of 23.5% on $1200 results in the same final price
    as successive discounts of 15% and 10%. -/
theorem equivalent_discount (original_price : ℝ) (discount1 discount2 single_discount : ℝ) :
  original_price = 1200 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - single_discount) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l2880_288083


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2880_288021

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 255 → (¬(11 ∣ (y + 9)) ∨ ¬(24 ∣ (y + 9)))) ∧ 
  (11 ∣ (255 + 9)) ∧ 
  (24 ∣ (255 + 9)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2880_288021


namespace NUMINAMATH_CALUDE_somu_age_problem_l2880_288082

/-- Represents the problem of finding when Somu was one-fifth of his father's age -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 14 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 7 := by
  sorry

end NUMINAMATH_CALUDE_somu_age_problem_l2880_288082


namespace NUMINAMATH_CALUDE_selection_ways_eq_55_l2880_288098

/-- The number of ways to select 5 students out of 5 male and 3 female students,
    ensuring both male and female students are included. -/
def selection_ways : ℕ :=
  Nat.choose 8 5 - Nat.choose 5 5

/-- Theorem stating that the number of ways to select 5 students
    out of 5 male and 3 female students, ensuring both male and
    female students are included, is equal to 55. -/
theorem selection_ways_eq_55 : selection_ways = 55 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_55_l2880_288098


namespace NUMINAMATH_CALUDE_tangent_at_negative_one_range_of_a_l2880_288039

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the condition for the shared tangent line
def shared_tangent (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ (x₂ : ℝ), f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- Theorem for part 1
theorem tangent_at_negative_one (a : ℝ) :
  shared_tangent a (-1) → a = 3 := by sorry

-- Theorem for part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ : ℝ, shared_tangent a x₁) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_tangent_at_negative_one_range_of_a_l2880_288039


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l2880_288031

/-- Proves that 850 billion yuan is equal to 8.5 × 10^11 yuan -/
theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  850 * billion = 8.5 * 10^11 := by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l2880_288031


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2880_288034

theorem abs_neg_two_equals_two : abs (-2 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l2880_288034


namespace NUMINAMATH_CALUDE_series_sum_l2880_288029

/-- The series defined by the problem -/
def series (n : ℕ) : ℚ :=
  if n % 3 = 1 then 1 / (2^n)
  else if n % 3 = 0 then -1 / (2^n)
  else -1 / (2^n)

/-- The sum of the series -/
noncomputable def S : ℚ := ∑' n, series n

/-- The theorem to be proved -/
theorem series_sum : S / (10 * 81) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l2880_288029


namespace NUMINAMATH_CALUDE_oranges_problem_l2880_288012

def oranges_left (mary jason tom sarah : ℕ) : ℕ :=
  let initial_total := mary + jason + tom + sarah
  let increased_total := (initial_total * 110 + 50) / 100  -- Rounded up
  (increased_total * 85 + 50) / 100  -- Rounded down

theorem oranges_problem (mary jason tom sarah : ℕ) 
  (h_mary : mary = 122)
  (h_jason : jason = 105)
  (h_tom : tom = 85)
  (h_sarah : sarah = 134) :
  oranges_left mary jason tom sarah = 417 := by
  sorry

end NUMINAMATH_CALUDE_oranges_problem_l2880_288012


namespace NUMINAMATH_CALUDE_boys_meeting_time_l2880_288061

/-- Two boys running on a circular track meet after a specific time -/
theorem boys_meeting_time (track_length : Real) (speed1 speed2 : Real) :
  track_length = 4800 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 100 * (1000 / 3600) →
  track_length / (speed1 + speed2) = 108 := by
  sorry

end NUMINAMATH_CALUDE_boys_meeting_time_l2880_288061


namespace NUMINAMATH_CALUDE_sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l2880_288064

theorem sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one :
  (Real.sqrt 2)^2 - Real.sqrt 9 + (8 : ℝ)^(1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_minus_sqrt_nine_plus_cube_root_eight_equals_one_l2880_288064


namespace NUMINAMATH_CALUDE_moon_temperature_difference_l2880_288090

theorem moon_temperature_difference : 
  let noon_temp : ℤ := 10
  let midnight_temp : ℤ := -150
  noon_temp - midnight_temp = 160 := by
sorry

end NUMINAMATH_CALUDE_moon_temperature_difference_l2880_288090


namespace NUMINAMATH_CALUDE_rotate90_of_4_minus_2i_l2880_288017

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_of_4_minus_2i : 
  rotate90 (4 - 2 * Complex.I) = 2 + 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotate90_of_4_minus_2i_l2880_288017


namespace NUMINAMATH_CALUDE_train_crossing_time_l2880_288041

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 120 →
  train_speed_kmh = 18 →
  (2 * train_length) / (2 * train_speed_kmh * (1000 / 3600)) = 24 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2880_288041


namespace NUMINAMATH_CALUDE_simplified_expression_equals_negative_three_l2880_288072

theorem simplified_expression_equals_negative_three :
  let a : ℚ := -4
  (1 / (a - 1) + 1) / (a / (a^2 - 1)) = -3 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_negative_three_l2880_288072


namespace NUMINAMATH_CALUDE_charcoal_for_900ml_l2880_288032

/-- Given a ratio of charcoal to water and a volume of water, calculate the amount of charcoal needed. -/
def charcoal_needed (charcoal_ratio : ℚ) (water_volume : ℚ) : ℚ :=
  water_volume / (30 / charcoal_ratio)

/-- Theorem: The amount of charcoal needed for 900 ml of water is 60 grams, given the ratio of 2 grams of charcoal per 30 ml of water. -/
theorem charcoal_for_900ml :
  charcoal_needed 2 900 = 60 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_for_900ml_l2880_288032


namespace NUMINAMATH_CALUDE_bella_soccer_goals_l2880_288045

def goals_first_6 : List Nat := [5, 3, 2, 4, 1, 6]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem bella_soccer_goals :
  ∀ (g7 g8 : Nat),
    g7 < 10 →
    g8 < 10 →
    is_integer ((goals_first_6.sum + g7) / 7) →
    is_integer ((goals_first_6.sum + g7 + g8) / 8) →
    g7 * g8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bella_soccer_goals_l2880_288045


namespace NUMINAMATH_CALUDE_square_root_of_1024_l2880_288096

theorem square_root_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 1024) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l2880_288096


namespace NUMINAMATH_CALUDE_student_ticket_price_is_correct_l2880_288009

/-- Represents the ticket sales data for a single day -/
structure DaySales where
  senior : ℕ
  student : ℕ
  adult : ℕ
  total : ℚ

/-- Represents the price changes for a day -/
structure PriceChange where
  senior : ℚ
  student : ℚ
  adult : ℚ

/-- Finds the initial price of a student ticket given the sales data and price changes -/
def find_student_ticket_price (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) : ℚ :=
  sorry

/-- The main theorem stating that the initial price of a student ticket is approximately $8.83 -/
theorem student_ticket_price_is_correct (sales : Vector DaySales 5) (day4_change : PriceChange) (day5_change : PriceChange) :
  let price := find_student_ticket_price sales day4_change day5_change
  abs (price - 8.83) < 0.01 := by sorry

end NUMINAMATH_CALUDE_student_ticket_price_is_correct_l2880_288009


namespace NUMINAMATH_CALUDE_compound_interest_problem_l2880_288019

/-- Proves that given the conditions, the principal amount is 20,000 --/
theorem compound_interest_problem (P : ℝ) : 
  P * ((1 + 0.1)^4 - (1 + 0.2)^2) = 482 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l2880_288019


namespace NUMINAMATH_CALUDE_find_y_value_l2880_288058

theorem find_y_value (x y : ℝ) (h1 : x^2 - x + 3 = y + 3) (h2 : x = -5) (h3 : y > 0) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l2880_288058


namespace NUMINAMATH_CALUDE_third_anthill_population_l2880_288073

/-- Calculates the number of ants in the next anthill given the current number of ants -/
def next_anthill_population (current_ants : ℕ) : ℕ :=
  (current_ants * 4) / 5

/-- Represents the forest with three anthills -/
structure Forest where
  anthill1 : ℕ
  anthill2 : ℕ
  anthill3 : ℕ

/-- Creates a forest with three anthills, where each subsequent anthill has 20% fewer ants -/
def create_forest (initial_ants : ℕ) : Forest :=
  let anthill2 := next_anthill_population initial_ants
  let anthill3 := next_anthill_population anthill2
  { anthill1 := initial_ants, anthill2 := anthill2, anthill3 := anthill3 }

/-- Theorem stating that in a forest with 100 ants in the first anthill, 
    the third anthill will have 64 ants -/
theorem third_anthill_population : 
  (create_forest 100).anthill3 = 64 := by sorry

end NUMINAMATH_CALUDE_third_anthill_population_l2880_288073


namespace NUMINAMATH_CALUDE_problem_solution_l2880_288003

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) + x

def g (x : ℝ) : ℝ := x - 1

def h (m : ℝ) (f' : ℝ → ℝ) (x : ℝ) : ℝ := m * f' x + g x + 1

theorem problem_solution (a : ℝ) :
  (∀ x, deriv (f a) x = a / (x - 1) + 1) →
  deriv (f a) 2 = 2 →
  (a = 1 ∧
   (∀ x, g x = x - 1) ∧
   (∀ m, (∀ x ∈ Set.Icc 2 4, h m (deriv (f a)) x > 0) → m > -1 ∧ ∀ y > -1, ∃ x ∈ Set.Icc 2 4, h y (deriv (f a)) x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2880_288003


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2880_288093

theorem cube_volume_problem (a : ℝ) (h : a > 0) :
  (a^3 : ℝ) - ((a - 1) * a * (a + 1)) = 5 →
  a^3 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2880_288093


namespace NUMINAMATH_CALUDE_max_product_sum_l2880_288014

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) ∧
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l2880_288014


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2880_288015

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The conditions for the three-digit number -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  n.units + n.hundreds = n.tens ∧
  7 * n.hundreds = n.units + n.tens + 2 ∧
  n.units + n.tens + n.hundreds = 14

/-- The theorem stating that 275 is the only three-digit number satisfying the conditions -/
theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ 
    n.hundreds = 2 ∧ n.tens = 7 ∧ n.units = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2880_288015


namespace NUMINAMATH_CALUDE_anna_cannot_afford_tour_l2880_288089

/-- Calculates the future value of an amount with compound interest -/
def futureValue (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents Anna's initial savings -/
def initialSavings : ℝ := 40000

/-- Represents the initial cost of the tour package -/
def initialCost : ℝ := 45000

/-- Represents the annual interest rate -/
def interestRate : ℝ := 0.05

/-- Represents the annual inflation rate -/
def inflationRate : ℝ := 0.05

/-- Represents the time period in years -/
def timePeriod : ℕ := 3

/-- Theorem stating that Anna cannot afford the tour package after 3 years -/
theorem anna_cannot_afford_tour : 
  futureValue initialSavings interestRate timePeriod < 
  futureValue initialCost inflationRate timePeriod := by
  sorry

end NUMINAMATH_CALUDE_anna_cannot_afford_tour_l2880_288089


namespace NUMINAMATH_CALUDE_dans_grocery_items_l2880_288007

/-- Represents the items bought at the grocery store -/
structure GroceryItems where
  eggs : ℕ
  flour : ℕ
  butter : ℕ
  vanilla : ℕ

/-- Calculates the total number of individual items -/
def totalItems (items : GroceryItems) : ℕ :=
  items.eggs + items.flour + items.butter + items.vanilla

/-- Theorem stating the total number of items Dan bought -/
theorem dans_grocery_items : ∃ (items : GroceryItems), 
  items.eggs = 9 * 12 ∧ 
  items.flour = 6 ∧ 
  items.butter = 3 * 24 ∧ 
  items.vanilla = 12 ∧ 
  totalItems items = 198 := by
  sorry


end NUMINAMATH_CALUDE_dans_grocery_items_l2880_288007


namespace NUMINAMATH_CALUDE_min_a_for_probability_half_or_more_l2880_288001

/-- Represents a deck of cards numbered from 1 to 60 -/
def Deck := Finset (Fin 60)

/-- Represents the probability function p(a,b) -/
noncomputable def p (a b : ℕ) : ℚ :=
  let remaining_cards := 58
  let total_ways := Nat.choose remaining_cards 2
  let lower_team_ways := Nat.choose (a - 1) 2
  let higher_team_ways := Nat.choose (48 - a) 2
  (lower_team_ways + higher_team_ways : ℚ) / total_ways

/-- The main theorem to prove -/
theorem min_a_for_probability_half_or_more (deck : Deck) :
  (∀ a < 13, p a (a + 10) < 1/2) ∧ 
  p 13 23 = 473/551 ∧
  p 13 23 ≥ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_min_a_for_probability_half_or_more_l2880_288001


namespace NUMINAMATH_CALUDE_sector_central_angle_l2880_288022

/-- The central angle of a sector with radius 1 cm and arc length 2 cm is 2 radians. -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (central_angle : ℝ) : 
  radius = 1 → arc_length = 2 → arc_length = radius * central_angle → central_angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2880_288022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2880_288057

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ = -5 and the common difference is 3,
    prove that a₁ = -8 -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)
  (h_arith : IsArithmeticSequence a)
  (h_a2 : a 2 = -5)
  (h_d : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2880_288057


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2880_288069

theorem composition_equation_solution :
  let δ : ℝ → ℝ := λ x ↦ 4 * x + 9
  let φ : ℝ → ℝ := λ x ↦ 9 * x + 6
  ∃ x : ℝ, δ (φ x) = 10 ∧ x = -23/36 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2880_288069


namespace NUMINAMATH_CALUDE_rower_upstream_speed_man_rowing_upstream_speed_l2880_288035

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
theorem rower_upstream_speed (v_still : ℝ) (v_downstream : ℝ) : 
  v_still > 0 → v_downstream > v_still → 
  2 * v_still - v_downstream = v_still - (v_downstream - v_still) := by
  sorry

/-- The specific problem instance -/
theorem man_rowing_upstream_speed : 
  let v_still : ℝ := 33
  let v_downstream : ℝ := 40
  2 * v_still - v_downstream = 26 := by
  sorry

end NUMINAMATH_CALUDE_rower_upstream_speed_man_rowing_upstream_speed_l2880_288035


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_19_l2880_288002

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_19 :
  ∃ (p : ℕ), is_prime p ∧ digit_sum p = 19 ∧
  ∀ (q : ℕ), is_prime q → digit_sum q = 19 → p ≤ q :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_19_l2880_288002


namespace NUMINAMATH_CALUDE_problem_statement_l2880_288094

open Real

theorem problem_statement :
  (∀ x ∈ Set.Ioo (-π/2) 0, sin x > x) ∧
  ¬(Set.Ioo 0 1 = {x | log (1 - x) / log 10 < 1}) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2880_288094


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2880_288026

theorem gcd_of_three_numbers : Nat.gcd 1734 (Nat.gcd 816 1343) = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2880_288026


namespace NUMINAMATH_CALUDE_seventh_term_is_28_l2880_288004

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℝ
  -- Common difference of the sequence
  d : ℝ
  -- Sum of first three terms is 9
  sum_first_three : a + (a + d) + (a + 2 * d) = 9
  -- Third term is 8
  third_term : a + 2 * d = 8

/-- The seventh term of the arithmetic sequence is 28 -/
theorem seventh_term_is_28 (seq : ArithmeticSequence) : 
  seq.a + 6 * seq.d = 28 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_is_28_l2880_288004


namespace NUMINAMATH_CALUDE_zeta_power_sum_l2880_288067

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 9) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 179 := by
  sorry

end NUMINAMATH_CALUDE_zeta_power_sum_l2880_288067


namespace NUMINAMATH_CALUDE_members_playing_both_l2880_288028

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  badminton : ℕ
  tennis : ℕ
  neither : ℕ

/-- Calculate the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - (club.total - club.neither)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub) 
  (h1 : club.total = 30)
  (h2 : club.badminton = 18)
  (h3 : club.tennis = 19)
  (h4 : club.neither = 2) :
  playBoth club = 9 := by
  sorry

#eval playBoth { total := 30, badminton := 18, tennis := 19, neither := 2 }

end NUMINAMATH_CALUDE_members_playing_both_l2880_288028


namespace NUMINAMATH_CALUDE_four_solutions_l2880_288025

/-- The number of positive integer solutions to the equation 3x + y = 15 -/
def solution_count : Nat :=
  (Finset.filter (fun p : Nat × Nat => 3 * p.1 + p.2 = 15 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 15) (Finset.range 15))).card

/-- Theorem stating that there are exactly 4 pairs of positive integers (x, y) satisfying 3x + y = 15 -/
theorem four_solutions : solution_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_solutions_l2880_288025


namespace NUMINAMATH_CALUDE_sin_arithmetic_is_geometric_ratio_l2880_288053

def is_arithmetic_sequence (α : ℕ → ℝ) (β : ℝ) :=
  ∀ n, α (n + 1) = α n + β

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem sin_arithmetic_is_geometric_ratio (α : ℕ → ℝ) (β : ℝ) :
  is_arithmetic_sequence α β →
  (∃ q, is_geometric_sequence (fun n ↦ Real.sin (α n)) q) →
  ∃ q, (q = 1 ∨ q = -1) ∧ is_geometric_sequence (fun n ↦ Real.sin (α n)) q :=
by sorry

end NUMINAMATH_CALUDE_sin_arithmetic_is_geometric_ratio_l2880_288053


namespace NUMINAMATH_CALUDE_circle_center_fourth_quadrant_l2880_288013

/-- Given a real number a, if the equation x^2 + y^2 - 2ax + 4ay + 6a^2 - a = 0 
    represents a circle with its center in the fourth quadrant, then 0 < a < 1. -/
theorem circle_center_fourth_quadrant (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x + 4*a*y + 6*a^2 - a = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∀ x' y' : ℝ, (x' - a)^2 + (y' + 2*a)^2 = r^2) →
  (a > 0 ∧ -2*a < 0) →
  0 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_fourth_quadrant_l2880_288013


namespace NUMINAMATH_CALUDE_tom_not_in_middle_seat_l2880_288066

-- Define the people
inductive Person : Type
| Andy : Person
| Jen : Person
| Sally : Person
| Mike : Person
| Tom : Person

-- Define a seating arrangement as a function from seat number to person
def Seating := Fin 5 → Person

-- Andy is not beside Jen
def AndyNotBesideJen (s : Seating) : Prop :=
  ∀ i : Fin 4, s i ≠ Person.Andy ∨ s i.succ ≠ Person.Jen

-- Sally is beside Mike
def SallyBesideMike (s : Seating) : Prop :=
  ∃ i : Fin 4, (s i = Person.Sally ∧ s i.succ = Person.Mike) ∨
               (s i = Person.Mike ∧ s i.succ = Person.Sally)

-- The middle seat is the third seat
def MiddleSeat : Fin 5 := ⟨2, by norm_num⟩

-- Theorem: Tom cannot sit in the middle seat
theorem tom_not_in_middle_seat :
  ∀ s : Seating, AndyNotBesideJen s → SallyBesideMike s →
  s MiddleSeat ≠ Person.Tom :=
by sorry

end NUMINAMATH_CALUDE_tom_not_in_middle_seat_l2880_288066


namespace NUMINAMATH_CALUDE_max_value_product_sum_l2880_288076

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l2880_288076


namespace NUMINAMATH_CALUDE_simplify_expression_l2880_288092

theorem simplify_expression (x : ℝ) : 120 * x - 52 * x = 68 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2880_288092


namespace NUMINAMATH_CALUDE_isabel_paper_left_l2880_288059

/-- The number of pieces of paper Isabel bought in her first purchase -/
def first_purchase : ℕ := 900

/-- The number of pieces of paper Isabel bought in her second purchase -/
def second_purchase : ℕ := 300

/-- The number of pieces of paper Isabel used for a school project -/
def school_project : ℕ := 156

/-- The number of pieces of paper Isabel used for her artwork -/
def artwork : ℕ := 97

/-- The number of pieces of paper Isabel used for writing letters -/
def letters : ℕ := 45

/-- The theorem stating that Isabel has 902 pieces of paper left -/
theorem isabel_paper_left : 
  first_purchase + second_purchase - (school_project + artwork + letters) = 902 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_left_l2880_288059


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_difference_l2880_288000

theorem arithmetic_sequence_sum_difference : 
  let seq1 := List.range 93
  let seq2 := List.range 93
  let sum1 := (List.sum (seq1.map (fun i => 2001 + i)))
  let sum2 := (List.sum (seq2.map (fun i => 201 + i)))
  sum1 - sum2 = 167400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_difference_l2880_288000


namespace NUMINAMATH_CALUDE_get_ready_time_l2880_288044

/-- The time it takes Jack to put on his own shoes, in minutes. -/
def jack_shoes_time : ℕ := 4

/-- The additional time it takes Jack to help a toddler with their shoes, in minutes. -/
def additional_toddler_time : ℕ := 3

/-- The number of toddlers Jack needs to help. -/
def number_of_toddlers : ℕ := 2

/-- The total time it takes for Jack and his toddlers to get ready, in minutes. -/
def total_time : ℕ := jack_shoes_time + number_of_toddlers * (jack_shoes_time + additional_toddler_time)

theorem get_ready_time : total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_get_ready_time_l2880_288044


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2880_288088

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) → 
  (∀ x : ℝ, f x = 1/2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2880_288088


namespace NUMINAMATH_CALUDE_f_unbounded_above_l2880_288024

/-- The function f(x, y) = 2x^2 + 4xy + 5y^2 + 8x - 6y -/
def f (x y : ℝ) : ℝ := 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y

/-- Theorem: The function f is unbounded above -/
theorem f_unbounded_above : ∀ M : ℝ, ∃ x y : ℝ, f x y > M := by
  sorry

end NUMINAMATH_CALUDE_f_unbounded_above_l2880_288024


namespace NUMINAMATH_CALUDE_calculator_sequence_101_l2880_288065

def calculator_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 7
  | n + 1 => 1 / (1 - calculator_sequence n)

theorem calculator_sequence_101 : calculator_sequence 101 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_calculator_sequence_101_l2880_288065


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l2880_288049

theorem probability_at_least_one_red (prob_red_A prob_red_B : ℝ) :
  prob_red_A = 1/3 →
  prob_red_B = 1/2 →
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l2880_288049


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l2880_288080

-- Define a function to get the last two digits of 8^n
def lastTwoDigits (n : ℕ) : ℕ := 8^n % 100

-- Define the cycle of last two digits
def lastTwoDigitsCycle : List ℕ := [8, 64, 12, 96, 68, 44, 52, 16, 28, 24]

-- Theorem statement
theorem tens_digit_of_8_pow_2023 :
  (lastTwoDigits 2023 / 10) % 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l2880_288080


namespace NUMINAMATH_CALUDE_smallest_z_minus_x_is_444_l2880_288046

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_z_minus_x_is_444 :
  ∃ (x y z : ℕ+),
    (x.val * y.val * z.val = factorial 10) ∧
    (x < y) ∧ (y < z) ∧
    (∀ (a b c : ℕ+),
      (a.val * b.val * c.val = factorial 10) → (a < b) → (b < c) →
      ((z.val - x.val : ℤ) ≤ (c.val - a.val))) ∧
    (z.val - x.val = 444) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_minus_x_is_444_l2880_288046


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2880_288085

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_diff_l2880_288085


namespace NUMINAMATH_CALUDE_horse_race_equation_l2880_288040

/-- The speed of the good horse in miles per day -/
def good_horse_speed : ℕ := 200

/-- The speed of the slow horse in miles per day -/
def slow_horse_speed : ℕ := 120

/-- The number of days the slow horse starts earlier -/
def head_start : ℕ := 10

/-- The number of days it takes for the good horse to catch up -/
def catch_up_days : ℕ := sorry

theorem horse_race_equation :
  good_horse_speed * catch_up_days = slow_horse_speed * catch_up_days + slow_horse_speed * head_start :=
by sorry

end NUMINAMATH_CALUDE_horse_race_equation_l2880_288040


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2880_288016

theorem cube_volume_ratio : 
  let cube1_side_length : ℝ := 2  -- in meters
  let cube2_side_length : ℝ := 100 / 100  -- 100 cm converted to meters
  let cube1_volume := cube1_side_length ^ 3
  let cube2_volume := cube2_side_length ^ 3
  cube1_volume / cube2_volume = 8 := by sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2880_288016


namespace NUMINAMATH_CALUDE_min_value_of_function_equality_condition_min_value_exists_l2880_288055

theorem min_value_of_function (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x ≥ 6 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 2 + 4*x + 1/x = 6 ↔ x = 1/2 :=
sorry

theorem min_value_exists : ∃ x : ℝ, x > 0 ∧ 2 + 4*x + 1/x = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_equality_condition_min_value_exists_l2880_288055


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2880_288081

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 1 - a 3 = -3) :
  a 4 = -8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2880_288081


namespace NUMINAMATH_CALUDE_smallest_n_with_four_trailing_zeros_l2880_288051

def is_divisible_by_10000 (n : ℕ) : Prop :=
  (n.choose 4) % 10000 = 0

theorem smallest_n_with_four_trailing_zeros : 
  ∀ k : ℕ, k ≥ 4 ∧ k < 8128 → ¬(is_divisible_by_10000 k) ∧ is_divisible_by_10000 8128 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_trailing_zeros_l2880_288051


namespace NUMINAMATH_CALUDE_legs_minus_twice_heads_diff_l2880_288062

/-- Represents the number of legs for each animal type -/
def legs_per_animal : Nat → Nat
| 0 => 2  -- Chicken
| 1 => 4  -- Cow
| _ => 0  -- Other animals (not used in this problem)

/-- Calculates the total number of legs in the group -/
def total_legs (num_chickens num_cows : Nat) : Nat :=
  legs_per_animal 0 * num_chickens + legs_per_animal 1 * num_cows

/-- Calculates the total number of heads in the group -/
def total_heads (num_chickens num_cows : Nat) : Nat :=
  num_chickens + num_cows

/-- The main theorem stating the difference between legs and twice the heads -/
theorem legs_minus_twice_heads_diff (num_chickens : Nat) : 
  total_legs num_chickens 7 - 2 * total_heads num_chickens 7 = 14 := by
  sorry

#check legs_minus_twice_heads_diff

end NUMINAMATH_CALUDE_legs_minus_twice_heads_diff_l2880_288062


namespace NUMINAMATH_CALUDE_choose_four_from_six_eq_fifteen_l2880_288048

/-- The number of ways to choose 4 items from a set of 6 items, where the order doesn't matter -/
def choose_four_from_six : ℕ := Nat.choose 6 4

/-- Theorem stating that choosing 4 items from a set of 6 items results in 15 combinations -/
theorem choose_four_from_six_eq_fifteen : choose_four_from_six = 15 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_six_eq_fifteen_l2880_288048


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2880_288037

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem equal_area_rectangles_width (carol_rect jordan_rect : Rectangle) 
  (h1 : carol_rect.length = 15)
  (h2 : carol_rect.width = 20)
  (h3 : jordan_rect.length = 6 * 12)
  (h4 : area carol_rect = area jordan_rect) :
  jordan_rect.width = 300 / (6 * 12) := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2880_288037


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l2880_288043

open Matrix

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = !![1, 4; -2, -7]) : 
  (B^2)⁻¹ = !![(-7), (-24); 12, 41] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l2880_288043


namespace NUMINAMATH_CALUDE_first_five_valid_numbers_l2880_288086

def is_valid (n : ℕ) : Bool :=
  n ≥ 0 ∧ n ≤ 499

def random_sequence : List ℕ :=
  [164, 785, 916, 955, 567, 199, 810, 507, 185, 128, 673, 580, 744, 395]

def first_five_valid (seq : List ℕ) : List ℕ :=
  seq.filter is_valid |> List.take 5

theorem first_five_valid_numbers :
  first_five_valid random_sequence = [164, 199, 185, 128, 395] := by
  sorry

end NUMINAMATH_CALUDE_first_five_valid_numbers_l2880_288086


namespace NUMINAMATH_CALUDE_unique_integer_sum_pair_l2880_288011

theorem unique_integer_sum_pair (a : ℕ → ℝ) (h1 : 1 < a 1 ∧ a 1 < 2) 
  (h2 : ∀ k : ℕ, a (k + 1) = a k + k / a k) :
  ∃! (i j : ℕ), i ≠ j ∧ ∃ m : ℤ, (a i + a j : ℝ) = m := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_sum_pair_l2880_288011


namespace NUMINAMATH_CALUDE_representations_of_2022_l2880_288010

/-- Represents a sequence of consecutive natural numbers. -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ

/-- The sum of a consecutive sequence of natural numbers. -/
def sum_consecutive (seq : ConsecutiveSequence) : ℕ :=
  seq.length * (2 * seq.start + seq.length - 1) / 2

/-- Checks if a consecutive sequence sums to a given target. -/
def is_valid_representation (seq : ConsecutiveSequence) (target : ℕ) : Prop :=
  sum_consecutive seq = target

theorem representations_of_2022 :
  ∀ (seq : ConsecutiveSequence),
    is_valid_representation seq 2022 ↔
      (seq.start = 673 ∧ seq.length = 3) ∨
      (seq.start = 504 ∧ seq.length = 4) ∨
      (seq.start = 163 ∧ seq.length = 12) :=
by sorry

end NUMINAMATH_CALUDE_representations_of_2022_l2880_288010


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2880_288097

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -13/2 < x ∧ x < 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2880_288097


namespace NUMINAMATH_CALUDE_maintenance_scheduling_methods_l2880_288052

/-- Represents the days of the week --/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents the monitoring points --/
inductive MonitoringPoint
  | A
  | B
  | C
  | D
  | E
  | F
  | G
  | H

/-- A schedule is a function from MonitoringPoint to Day --/
def Schedule := MonitoringPoint → Day

/-- Checks if a schedule is valid according to the given conditions --/
def isValidSchedule (s : Schedule) : Prop :=
  (s MonitoringPoint.A = Day.Monday) ∧
  (s MonitoringPoint.B = Day.Tuesday) ∧
  (s MonitoringPoint.C = s MonitoringPoint.D) ∧
  (s MonitoringPoint.D = s MonitoringPoint.E) ∧
  (s MonitoringPoint.F ≠ Day.Friday) ∧
  (∀ d : Day, ∃ p : MonitoringPoint, s p = d)

/-- The total number of valid schedules --/
def totalValidSchedules : ℕ := sorry

theorem maintenance_scheduling_methods :
  totalValidSchedules = 60 := by sorry

end NUMINAMATH_CALUDE_maintenance_scheduling_methods_l2880_288052


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l2880_288095

/-- Given points A, B, and C in a 2D plane, with specific conditions on their coordinates and the lines connecting them, prove that the sum of certain coordinate values is 1. -/
theorem point_coordinate_sum (a b : ℝ) : 
  let A : ℝ × ℝ := (a, 5)
  let B : ℝ × ℝ := (2, 2 - b)
  let C : ℝ × ℝ := (4, 2)
  (A.2 = B.2) →  -- AB is parallel to x-axis
  (A.1 = C.1) →  -- AC is parallel to y-axis
  a + b = 1 := by
sorry


end NUMINAMATH_CALUDE_point_coordinate_sum_l2880_288095


namespace NUMINAMATH_CALUDE_multiply_polynomial_l2880_288036

theorem multiply_polynomial (x : ℝ) : 
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_l2880_288036


namespace NUMINAMATH_CALUDE_kellys_games_l2880_288077

/-- Kelly's Nintendo games problem -/
theorem kellys_games (initial_games given_away_games : ℕ) : 
  initial_games = 121 → given_away_games = 99 → 
  initial_games - given_away_games = 22 := by
  sorry

end NUMINAMATH_CALUDE_kellys_games_l2880_288077


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2880_288023

theorem binomial_coefficient_congruence (n p : ℕ) (h_prime : Nat.Prime p) (h_n_gt_p : n > p) :
  (n.choose p) ≡ (n / p : ℕ) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2880_288023


namespace NUMINAMATH_CALUDE_billion_yuan_eq_scientific_notation_l2880_288042

/-- Represents the value in billions of yuan -/
def billion_yuan : ℝ := 98.36

/-- Represents the same value in scientific notation -/
def scientific_notation : ℝ := 9.836 * (10 ^ 9)

/-- Theorem stating that the billion yuan value is equal to its scientific notation -/
theorem billion_yuan_eq_scientific_notation : billion_yuan * (10 ^ 9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_yuan_eq_scientific_notation_l2880_288042


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2880_288047

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2880_288047


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l2880_288020

/-- Given a function f(x) = a*sin(x) + b*x^3 + 5 where f(1) = 3, prove that f(-1) = 7 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * x^3 + 5) 
  (h2 : f 1 = 3) : 
  f (-1) = 7 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l2880_288020


namespace NUMINAMATH_CALUDE_correct_average_marks_l2880_288018

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℚ) (incorrect_mark correct_mark : ℚ) :
  n = 10 ∧ incorrect_avg = 100 ∧ incorrect_mark = 50 ∧ correct_mark = 10 →
  (n * incorrect_avg - (incorrect_mark - correct_mark)) / n = 96 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l2880_288018


namespace NUMINAMATH_CALUDE_power_three_times_three_l2880_288070

theorem power_three_times_three (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_three_times_three_l2880_288070


namespace NUMINAMATH_CALUDE_diagonal_sum_is_161_l2880_288050

/-- Represents a multiplication grid with missing factors --/
structure MultiplicationGrid where
  /-- Products in the grid --/
  wp : ℕ
  xp : ℕ
  wr : ℕ
  zr : ℕ
  xs : ℕ
  vs : ℕ
  vq : ℕ
  yq : ℕ
  yt : ℕ

/-- The sum of diagonal elements in the multiplication grid --/
def diagonalSum (grid : MultiplicationGrid) : ℕ :=
  let p := 3  -- Derived from wp and xp
  let w := grid.wp / p
  let x := grid.xp / p
  let r := grid.wr / w
  let z := grid.zr / r
  let s := grid.xs / x
  let v := grid.vs / s
  let q := grid.vq / v
  let y := grid.yq / q
  let t := grid.yt / y
  v * p + w * q + x * r + y * s + z * t

/-- Theorem stating that the diagonal sum is 161 for the given grid --/
theorem diagonal_sum_is_161 (grid : MultiplicationGrid) 
  (h1 : grid.wp = 15) (h2 : grid.xp = 18) (h3 : grid.wr = 40) 
  (h4 : grid.zr = 56) (h5 : grid.xs = 60) (h6 : grid.vs = 20) 
  (h7 : grid.vq = 10) (h8 : grid.yq = 20) (h9 : grid.yt = 24) : 
  diagonalSum grid = 161 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_is_161_l2880_288050


namespace NUMINAMATH_CALUDE_components_exceed_quarter_square_l2880_288099

/-- Represents a square grid of size n × n -/
structure Grid (n : ℕ) where
  size : n > 8

/-- Represents a diagonal in a cell of the grid -/
inductive Diagonal
  | TopLeft
  | TopRight

/-- Represents the configuration of diagonals in the grid -/
def DiagonalConfig (n : ℕ) := Fin n → Fin n → Diagonal

/-- Represents a connected component in the grid -/
structure Component (n : ℕ) where
  cells : Set (Fin n × Fin n)
  is_connected : True  -- Simplified connectivity condition

/-- The number of connected components in a given diagonal configuration -/
def num_components (n : ℕ) (config : DiagonalConfig n) : ℕ := sorry

/-- Theorem stating that the number of components can exceed n²/4 for n > 8 -/
theorem components_exceed_quarter_square {n : ℕ} (grid : Grid n) :
  ∃ (config : DiagonalConfig n), num_components n config > n^2 / 4 := by sorry

end NUMINAMATH_CALUDE_components_exceed_quarter_square_l2880_288099


namespace NUMINAMATH_CALUDE_notes_count_l2880_288074

theorem notes_count (total_amount : ℕ) (denominations : Fin 3 → ℕ) : 
  total_amount = 480 ∧ 
  denominations 0 = 1 ∧ 
  denominations 1 = 5 ∧ 
  denominations 2 = 10 ∧ 
  (∃ x : ℕ, (denominations 0 * x + denominations 1 * x + denominations 2 * x = total_amount)) →
  (∃ x : ℕ, x + x + x = 90) :=
by sorry

end NUMINAMATH_CALUDE_notes_count_l2880_288074


namespace NUMINAMATH_CALUDE_probability_all_different_at_most_one_odd_l2880_288075

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (all different numbers with at most one odd) -/
def favorableOutcomes : ℕ := 60

/-- The probability of rolling three dice and getting all different numbers with at most one odd number -/
def probabilityAllDifferentAtMostOneOdd : ℚ := favorableOutcomes / totalOutcomes

theorem probability_all_different_at_most_one_odd :
  probabilityAllDifferentAtMostOneOdd = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_different_at_most_one_odd_l2880_288075


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2880_288030

/-- An isosceles triangle with side lengths satisfying a specific equation has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧
    (a = b ∨ a + a = b ∨ b + b = a)) →  -- isosceles condition
  |x^2 - 9| + (y - 4)^2 = 0 →
  (x + y + min x y = 10) ∨ (x + y + min x y = 11) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2880_288030


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2880_288091

theorem trigonometric_problem (α : ℝ) 
  (h1 : Real.sin (α + π/3) + Real.sin α = 9 * Real.sqrt 7 / 14)
  (h2 : 0 < α)
  (h3 : α < π/3) :
  (Real.sin α = 2 * Real.sqrt 7 / 7) ∧ 
  (Real.cos (2*α - π/4) = (4 * Real.sqrt 6 - Real.sqrt 2) / 14) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2880_288091


namespace NUMINAMATH_CALUDE_second_train_length_problem_l2880_288027

/-- Calculates the length of the second train given the conditions of the problem -/
def second_train_length (first_train_length : ℝ) (first_train_speed : ℝ) (second_train_speed : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := first_train_speed - second_train_speed
  let total_distance := relative_speed * time_to_cross
  total_distance - first_train_length

/-- Theorem stating that given the problem conditions, the length of the second train is 299.9440044796417 m -/
theorem second_train_length_problem :
  let first_train_length : ℝ := 400
  let first_train_speed : ℝ := 72 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 36 * 1000 / 3600 -- Convert km/h to m/s
  let time_to_cross : ℝ := 69.99440044796417
  second_train_length first_train_length first_train_speed second_train_speed time_to_cross = 299.9440044796417 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_problem_l2880_288027


namespace NUMINAMATH_CALUDE_gcf_64_144_l2880_288078

theorem gcf_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end NUMINAMATH_CALUDE_gcf_64_144_l2880_288078


namespace NUMINAMATH_CALUDE_no_solution_factorial_equation_l2880_288079

theorem no_solution_factorial_equation :
  ∀ (m n : ℕ), m.factorial + 48 ≠ 48 * (m + 1) * n := by
  sorry

end NUMINAMATH_CALUDE_no_solution_factorial_equation_l2880_288079


namespace NUMINAMATH_CALUDE_container_weight_problem_l2880_288068

theorem container_weight_problem (x y z : ℝ) 
  (h1 : x + y = 234)
  (h2 : y + z = 241)
  (h3 : z + x = 255) :
  x + y + z = 365 := by
sorry

end NUMINAMATH_CALUDE_container_weight_problem_l2880_288068


namespace NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_l2880_288006

/-- A quadrilateral with two parallel sides of different lengths is a trapezoid -/
def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ : ℝ), l₁ ≠ l₂ ∧ 
  (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2) ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2

/-- The quadratic equation whose roots are the lengths of AB and CD -/
def length_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - 3*m*x + 2*m^2 + m - 2 = 0

theorem quadrilateral_is_trapezoid (A B C D : ℝ × ℝ) (m : ℝ) :
  (∃ (l₁ l₂ : ℝ), 
    length_equation m l₁ ∧ 
    length_equation m l₂ ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = l₁^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = l₂^2 ∧
    (B.1 - A.1) / (B.2 - A.2) = (D.1 - C.1) / (D.2 - C.2)) →
  is_trapezoid A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_trapezoid_l2880_288006
