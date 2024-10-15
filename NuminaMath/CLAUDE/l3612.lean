import Mathlib

namespace NUMINAMATH_CALUDE_square_root_sum_difference_l3612_361233

theorem square_root_sum_difference (x y : ℝ) : 
  x = Real.sqrt 7 + Real.sqrt 3 →
  y = Real.sqrt 7 - Real.sqrt 3 →
  x * y = 4 ∧ x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_difference_l3612_361233


namespace NUMINAMATH_CALUDE_classroom_capacity_l3612_361244

theorem classroom_capacity (total_students : ℕ) (num_classrooms : ℕ) 
  (h1 : total_students = 390) (h2 : num_classrooms = 13) :
  total_students / num_classrooms = 30 := by
  sorry

end NUMINAMATH_CALUDE_classroom_capacity_l3612_361244


namespace NUMINAMATH_CALUDE_remainder_n_squared_plus_2n_plus_3_l3612_361249

theorem remainder_n_squared_plus_2n_plus_3 (n : ℤ) (a : ℤ) (h : n = 100 * a - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_n_squared_plus_2n_plus_3_l3612_361249


namespace NUMINAMATH_CALUDE_train_length_l3612_361227

/-- The length of a train given its speed and time to pass a fixed point --/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 ∧ time = 28 → speed * time * (1000 / 3600) = 280 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3612_361227


namespace NUMINAMATH_CALUDE_metal_rods_for_fence_l3612_361211

/-- Calculates the number of metal rods needed for a fence with given specifications. -/
theorem metal_rods_for_fence (
  sheets_per_panel : ℕ)
  (beams_per_panel : ℕ)
  (panels : ℕ)
  (rods_per_sheet : ℕ)
  (rods_per_beam : ℕ)
  (h1 : sheets_per_panel = 3)
  (h2 : beams_per_panel = 2)
  (h3 : panels = 10)
  (h4 : rods_per_sheet = 10)
  (h5 : rods_per_beam = 4)
  : sheets_per_panel * panels * rods_per_sheet + beams_per_panel * panels * rods_per_beam = 380 := by
  sorry

#check metal_rods_for_fence

end NUMINAMATH_CALUDE_metal_rods_for_fence_l3612_361211


namespace NUMINAMATH_CALUDE_x_eighth_is_zero_l3612_361213

theorem x_eighth_is_zero (x : ℝ) (h : (1 - x^4)^(1/4) + (1 + x^4)^(1/4) = 1) : x^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_is_zero_l3612_361213


namespace NUMINAMATH_CALUDE_inequality_proof_l3612_361260

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3612_361260


namespace NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3612_361243

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def weekly_pattern : List Nat := [5, 3, 6, 4]
def finalization_hours : Nat := 10

def total_sewing_hours : Nat := dress_hours.sum
def total_hours : Nat := total_sewing_hours + finalization_hours
def cycle_hours : Nat := weekly_pattern.sum

def weeks_to_complete : Nat :=
  let full_cycles := (total_hours + cycle_hours - 1) / cycle_hours
  full_cycles * 4 - 3

theorem bridesmaid_dresses_completion_time :
  weeks_to_complete = 37 := by sorry

end NUMINAMATH_CALUDE_bridesmaid_dresses_completion_time_l3612_361243


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3612_361212

/-- A trapezoid with an inscribed circle -/
structure InscribedTrapezoid where
  -- The length of segment BL
  BL : ℝ
  -- The length of segment CL
  CL : ℝ
  -- The length of side AB
  AB : ℝ
  -- Assumption that BL is positive
  BL_pos : BL > 0
  -- Assumption that CL is positive
  CL_pos : CL > 0
  -- Assumption that AB is positive
  AB_pos : AB > 0

/-- The area of a trapezoid with an inscribed circle -/
def area (t : InscribedTrapezoid) : ℝ :=
  -- Define the area function here
  sorry

/-- Theorem: The area of the specific trapezoid is 6.75 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedTrapezoid,
  t.BL = 4 → t.CL = 1/4 → t.AB = 6 →
  area t = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3612_361212


namespace NUMINAMATH_CALUDE_greatest_XPM_l3612_361283

/-- A function that checks if a number is a two-digit number with equal digits -/
def is_two_digit_equal (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 11 = 0

/-- A function that checks if a number is a one-digit prime -/
def is_one_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

/-- The main theorem -/
theorem greatest_XPM :
  ∀ M N XPM : ℕ,
  is_two_digit_equal M →
  is_one_digit_prime N →
  N ≠ M / 11 →
  M * N = XPM →
  100 ≤ XPM ∧ XPM ≤ 999 →
  XPM ≤ 462 :=
sorry

end NUMINAMATH_CALUDE_greatest_XPM_l3612_361283


namespace NUMINAMATH_CALUDE_interest_rate_increase_specific_interest_rate_increase_l3612_361221

-- Define the initial interest rate
def last_year_rate : ℝ := 9.90990990990991

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Theorem to prove
theorem interest_rate_increase (last_year_rate : ℝ) (increase_percent : ℝ) :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

-- Apply the theorem to our specific values
theorem specific_interest_rate_increase :
  last_year_rate * (1 + increase_percent / 100) = 10.9009009009009 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_increase_specific_interest_rate_increase_l3612_361221


namespace NUMINAMATH_CALUDE_evaluate_nested_root_l3612_361270

theorem evaluate_nested_root : (((4 ^ (1/3)) ^ 4) ^ (1/2)) ^ 6 = 256 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_root_l3612_361270


namespace NUMINAMATH_CALUDE_arun_age_proof_l3612_361232

theorem arun_age_proof (A S G M : ℕ) : 
  A - 6 = 18 * G →
  G + 2 = M →
  M = 5 →
  S = A - 8 →
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_proof_l3612_361232


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_a_l3612_361223

def i : ℂ := Complex.I

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a (a : ℝ) :
  is_pure_imaginary ((2 + a * i) / (2 - i)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_a_l3612_361223


namespace NUMINAMATH_CALUDE_reflect_P_x_axis_l3612_361245

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The original point P -/
def P : Point :=
  { x := -2, y := 4 }

/-- Theorem: Reflecting point P(-2, 4) across the x-axis results in (-2, -4) -/
theorem reflect_P_x_axis :
  reflect_x P = { x := -2, y := -4 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_x_axis_l3612_361245


namespace NUMINAMATH_CALUDE_twenty_one_less_than_sixty_thousand_l3612_361201

theorem twenty_one_less_than_sixty_thousand : 60000 - 21 = 59979 := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_less_than_sixty_thousand_l3612_361201


namespace NUMINAMATH_CALUDE_impossibility_of_all_prime_combinations_l3612_361284

/-- A digit is a natural number from 0 to 9 -/
def Digit := {n : ℕ // n < 10}

/-- A two-digit number formed from two digits -/
def TwoDigitNumber (d1 d2 : Digit) : ℕ := d1.val * 10 + d2.val

/-- Predicate to check if a natural number is prime -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem impossibility_of_all_prime_combinations :
  ∀ (d1 d2 d3 d4 : Digit),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 →
    ∃ (i j : Fin 4),
      i ≠ j ∧
      ¬(IsPrime (TwoDigitNumber (([d1, d2, d3, d4].get i) : Digit) (([d1, d2, d3, d4].get j) : Digit))) :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_all_prime_combinations_l3612_361284


namespace NUMINAMATH_CALUDE_jaime_score_l3612_361218

theorem jaime_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) (jaime_score : ℚ) :
  n = 20 →
  avg_without = 85 →
  avg_with = 86 →
  (n - 1) * avg_without + jaime_score = n * avg_with →
  jaime_score = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_jaime_score_l3612_361218


namespace NUMINAMATH_CALUDE_silver_zinc_battery_properties_l3612_361250

/-- Represents an electrode in the battery -/
inductive Electrode
| Zinc
| SilverOxide

/-- Represents the direction of current flow -/
inductive CurrentFlow
| FromZincToSilverOxide
| FromSilverOxideToZinc

/-- Represents the change in OH⁻ concentration -/
inductive OHConcentrationChange
| Increase
| Decrease
| NoChange

/-- Models a silver-zinc battery -/
structure SilverZincBattery where
  negativeElectrode : Electrode
  positiveElectrode : Electrode
  zincReaction : String
  silverOxideReaction : String
  currentFlow : CurrentFlow
  ohConcentrationChange : OHConcentrationChange

/-- Theorem about the properties of a silver-zinc battery -/
theorem silver_zinc_battery_properties (battery : SilverZincBattery) 
  (h1 : battery.zincReaction = "Zn + 2OH⁻ - 2e⁻ = Zn(OH)₂")
  (h2 : battery.silverOxideReaction = "Ag₂O + H₂O + 2e⁻ = 2Ag + 2OH⁻") :
  battery.negativeElectrode = Electrode.Zinc ∧
  battery.positiveElectrode = Electrode.SilverOxide ∧
  battery.ohConcentrationChange = OHConcentrationChange.Increase ∧
  battery.currentFlow = CurrentFlow.FromSilverOxideToZinc :=
sorry

end NUMINAMATH_CALUDE_silver_zinc_battery_properties_l3612_361250


namespace NUMINAMATH_CALUDE_park_conditions_l3612_361276

-- Define the basic conditions
def temperature_at_least_70 : Prop := sorry
def is_sunny : Prop := sorry
def park_is_packed : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_70 ∧ is_sunny → park_is_packed) →
  (¬park_is_packed → ¬temperature_at_least_70 ∨ ¬is_sunny) :=
by sorry

end NUMINAMATH_CALUDE_park_conditions_l3612_361276


namespace NUMINAMATH_CALUDE_book_cost_price_l3612_361251

theorem book_cost_price (cost : ℝ) : 
  (1.15 * cost - 1.10 * cost = 100) → cost = 2000 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l3612_361251


namespace NUMINAMATH_CALUDE_total_wheels_count_l3612_361274

theorem total_wheels_count (num_bicycles num_tricycles : ℕ) 
  (wheels_per_bicycle wheels_per_tricycle : ℕ) :
  num_bicycles = 24 →
  num_tricycles = 14 →
  wheels_per_bicycle = 2 →
  wheels_per_tricycle = 3 →
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_count_l3612_361274


namespace NUMINAMATH_CALUDE_dice_tosses_probability_l3612_361266

/-- The number of sides on the dice -/
def num_sides : ℕ := 8

/-- The probability of rolling a 3 on a single toss -/
def p_roll_3 : ℚ := 1 / num_sides

/-- The target probability of rolling a 3 at least once -/
def target_prob : ℚ := 111328125 / 1000000000

/-- The number of tosses -/
def num_tosses : ℕ := 7

theorem dice_tosses_probability :
  1 - (1 - p_roll_3) ^ num_tosses = target_prob := by sorry

end NUMINAMATH_CALUDE_dice_tosses_probability_l3612_361266


namespace NUMINAMATH_CALUDE_correct_distribution_ways_l3612_361206

/-- The number of ways to distribute four distinct balls into two boxes -/
def distribution_ways : ℕ := 10

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- The minimum number of balls required in box 1 -/
def min_box1 : ℕ := 1

/-- The minimum number of balls required in box 2 -/
def min_box2 : ℕ := 2

/-- A function that calculates the number of ways to distribute balls -/
def calculate_distribution_ways (n : ℕ) (k : ℕ) (min1 : ℕ) (min2 : ℕ) : ℕ := sorry

/-- Theorem stating that the number of ways to distribute the balls is correct -/
theorem correct_distribution_ways :
  calculate_distribution_ways num_balls num_boxes min_box1 min_box2 = distribution_ways := by sorry

end NUMINAMATH_CALUDE_correct_distribution_ways_l3612_361206


namespace NUMINAMATH_CALUDE_two_pairs_more_likely_than_three_of_a_kind_l3612_361290

def num_dice : ℕ := 5
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def two_pairs_outcomes : ℕ := 
  num_dice * faces_per_die * (num_dice - 1).choose 2 * (faces_per_die - 1) * (faces_per_die - 2)

def three_of_a_kind_outcomes : ℕ := 
  num_dice.choose 3 * faces_per_die * (faces_per_die - 1) * (faces_per_die - 2)

theorem two_pairs_more_likely_than_three_of_a_kind :
  (two_pairs_outcomes : ℚ) / total_outcomes > (three_of_a_kind_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_more_likely_than_three_of_a_kind_l3612_361290


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l3612_361225

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 2015 ≥ 1987 := by sorry

theorem quadratic_minimum_achieved : ∃ x : ℝ, 7 * x^2 - 28 * x + 2015 = 1987 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achieved_l3612_361225


namespace NUMINAMATH_CALUDE_haleigh_leggings_needed_l3612_361291

/-- The number of pairs of leggings needed for pets -/
def leggings_needed (num_dogs : ℕ) (num_cats : ℕ) (legs_per_animal : ℕ) (legs_per_legging : ℕ) : ℕ :=
  ((num_dogs + num_cats) * legs_per_animal) / legs_per_legging

/-- Theorem: Haleigh needs 14 pairs of leggings for her pets -/
theorem haleigh_leggings_needed :
  leggings_needed 4 3 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_haleigh_leggings_needed_l3612_361291


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sum_of_sqrts_l3612_361230

theorem sqrt_sum_equals_sum_of_sqrts : 
  Real.sqrt (36 + 14 * Real.sqrt 6 + 14 * Real.sqrt 5 + 6 * Real.sqrt 30) = 
  Real.sqrt 15 + Real.sqrt 10 + Real.sqrt 8 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sum_of_sqrts_l3612_361230


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3612_361298

theorem smallest_n_for_candy_purchase : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * n) ∧
  (∀ (m : ℕ), m > 0 → 
    (∀ (r g b : ℕ), r > 0 → g > 0 → b > 0 → 10 * r = 18 * g ∧ 18 * g = 20 * b ∧ 20 * b = 24 * m) → 
    n ≤ m) ∧
  n = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l3612_361298


namespace NUMINAMATH_CALUDE_service_cost_calculation_l3612_361261

/-- The service cost per vehicle at a fuel station -/
def service_cost_per_vehicle : ℝ := 2.20

/-- The cost of fuel per liter -/
def fuel_cost_per_liter : ℝ := 0.70

/-- The capacity of a mini-van's fuel tank in liters -/
def minivan_tank_capacity : ℝ := 65

/-- The capacity of a truck's fuel tank in liters -/
def truck_tank_capacity : ℝ := minivan_tank_capacity * 2.2

/-- The number of mini-vans filled up -/
def num_minivans : ℕ := 3

/-- The number of trucks filled up -/
def num_trucks : ℕ := 2

/-- The total cost for filling up all vehicles -/
def total_cost : ℝ := 347.7

/-- Theorem stating that the service cost per vehicle is correct given the problem conditions -/
theorem service_cost_calculation :
  service_cost_per_vehicle * (num_minivans + num_trucks : ℝ) +
  fuel_cost_per_liter * (num_minivans * minivan_tank_capacity + num_trucks * truck_tank_capacity) =
  total_cost :=
by sorry

end NUMINAMATH_CALUDE_service_cost_calculation_l3612_361261


namespace NUMINAMATH_CALUDE_day_50_of_prev_year_is_tuesday_l3612_361224

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a year -/
structure Year where
  number : ℤ
  isLeapYear : Bool

/-- Returns the day of the week for a given day number in a year -/
def dayOfWeek (y : Year) (dayNumber : ℕ) : DayOfWeek := sorry

/-- Returns the next year -/
def nextYear (y : Year) : Year := sorry

/-- Returns the previous year -/
def prevYear (y : Year) : Year := sorry

theorem day_50_of_prev_year_is_tuesday 
  (N : Year)
  (h1 : dayOfWeek N 250 = DayOfWeek.Friday)
  (h2 : dayOfWeek (nextYear N) 150 = DayOfWeek.Friday)
  (h3 : (nextYear N).isLeapYear = false) :
  dayOfWeek (prevYear N) 50 = DayOfWeek.Tuesday := by sorry

end NUMINAMATH_CALUDE_day_50_of_prev_year_is_tuesday_l3612_361224


namespace NUMINAMATH_CALUDE_pond_draining_time_l3612_361220

/-- The time taken by the first pump to drain one-half of the pond -/
def first_pump_time : ℝ := 5

/-- The time taken by the second pump to drain the entire pond alone -/
def second_pump_time : ℝ := 1.1111111111111112

/-- The time taken by both pumps to drain the remaining half of the pond -/
def combined_time : ℝ := 0.5

theorem pond_draining_time : 
  (1 / (2 * first_pump_time) + 1 / second_pump_time) * combined_time = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_pond_draining_time_l3612_361220


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l3612_361241

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l3612_361241


namespace NUMINAMATH_CALUDE_complex_magnitude_l3612_361272

theorem complex_magnitude (z : ℂ) (h : 1 + z * Complex.I = 2 * Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3612_361272


namespace NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l3612_361209

theorem square_difference_divided_by_eleven : (131^2 - 120^2) / 11 = 251 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_eleven_l3612_361209


namespace NUMINAMATH_CALUDE_digit_addition_puzzle_l3612_361296

/-- Represents a four-digit number ABCD --/
def FourDigitNumber (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem digit_addition_puzzle :
  ∃ (possible_d : Finset ℕ),
    (∀ a b c d : ℕ,
      a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- Digits are distinct
      FourDigitNumber a a b c + FourDigitNumber b c a d = FourDigitNumber d b c d →  -- AABC + BCAD = DBCD
      d ∈ possible_d) ∧
    possible_d.card = 9  -- There are 9 possible values for D
  := by sorry

end NUMINAMATH_CALUDE_digit_addition_puzzle_l3612_361296


namespace NUMINAMATH_CALUDE_speed_in_still_water_problem_l3612_361242

/-- Calculates the speed in still water given the downstream speed and current speed. -/
def speed_in_still_water (downstream_speed current_speed : ℝ) : ℝ :=
  downstream_speed - current_speed

/-- Theorem: Given the conditions from the problem, the speed in still water is 30 kmph. -/
theorem speed_in_still_water_problem :
  let downstream_distance : ℝ := 0.24 -- 240 meters in km
  let downstream_time : ℝ := 24 / 3600 -- 24 seconds in hours
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let current_speed : ℝ := 6
  speed_in_still_water downstream_speed current_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_problem_l3612_361242


namespace NUMINAMATH_CALUDE_phone_call_probability_l3612_361292

/-- The probability of answering a phone call at the first ring -/
def p_first : ℝ := 0.1

/-- The probability of answering a phone call at the second ring -/
def p_second : ℝ := 0.3

/-- The probability of answering a phone call at the third ring -/
def p_third : ℝ := 0.4

/-- The probability of answering a phone call at the fourth ring -/
def p_fourth : ℝ := 0.1

/-- The events of answering at each ring are mutually exclusive -/
axiom mutually_exclusive : True

/-- The probability of answering within the first four rings -/
def p_within_four : ℝ := p_first + p_second + p_third + p_fourth

theorem phone_call_probability :
  p_within_four = 0.9 :=
sorry

end NUMINAMATH_CALUDE_phone_call_probability_l3612_361292


namespace NUMINAMATH_CALUDE_unique_number_theorem_l3612_361288

def A₁ (n : ℕ) : Prop := n < 12
def A₂ (n : ℕ) : Prop := ¬(7 ∣ n)
def A₃ (n : ℕ) : Prop := 5 * n < 70

def B₁ (n : ℕ) : Prop := 12 * n > 1000
def B₂ (n : ℕ) : Prop := 10 ∣ n
def B₃ (n : ℕ) : Prop := n > 100

def C₁ (n : ℕ) : Prop := 4 ∣ n
def C₂ (n : ℕ) : Prop := 11 * n < 1000
def C₃ (n : ℕ) : Prop := 9 ∣ n

def D₁ (n : ℕ) : Prop := n < 20
def D₂ (n : ℕ) : Prop := Nat.Prime n
def D₃ (n : ℕ) : Prop := 7 ∣ n

def at_least_one_true (p q r : Prop) : Prop := p ∨ q ∨ r
def at_least_one_false (p q r : Prop) : Prop := ¬p ∨ ¬q ∨ ¬r

theorem unique_number_theorem (n : ℕ) : 
  (at_least_one_true (A₁ n) (A₂ n) (A₃ n)) ∧ 
  (at_least_one_false (A₁ n) (A₂ n) (A₃ n)) ∧
  (at_least_one_true (B₁ n) (B₂ n) (B₃ n)) ∧ 
  (at_least_one_false (B₁ n) (B₂ n) (B₃ n)) ∧
  (at_least_one_true (C₁ n) (C₂ n) (C₃ n)) ∧ 
  (at_least_one_false (C₁ n) (C₂ n) (C₃ n)) ∧
  (at_least_one_true (D₁ n) (D₂ n) (D₃ n)) ∧ 
  (at_least_one_false (D₁ n) (D₂ n) (D₃ n)) →
  n = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_theorem_l3612_361288


namespace NUMINAMATH_CALUDE_largest_number_l3612_361268

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.986) 
  (hb : b = 0.9851) 
  (hc : c = 0.9869) 
  (hd : d = 0.9807) 
  (he : e = 0.9819) : 
  max a (max b (max c (max d e))) = c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3612_361268


namespace NUMINAMATH_CALUDE_sqrt_equation_l3612_361255

theorem sqrt_equation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l3612_361255


namespace NUMINAMATH_CALUDE_equation_solution_is_origin_l3612_361208

theorem equation_solution_is_origin (x y : ℝ) : 
  (x + y)^2 = 2 * (x^2 + y^2) ↔ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_is_origin_l3612_361208


namespace NUMINAMATH_CALUDE_triangle_side_length_l3612_361237

namespace TriangleProof

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ
  cos_A : ℝ
  cos_B : ℝ
  h_cos_A : cos_A = 3/5
  h_cos_B : cos_B = 5/13
  h_AC : AC = 3

/-- The main theorem to prove -/
theorem triangle_side_length (t : Triangle) : t.AB = 14/5 := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_triangle_side_length_l3612_361237


namespace NUMINAMATH_CALUDE_square_park_area_l3612_361256

theorem square_park_area (side_length : ℝ) (h : side_length = 200) :
  side_length * side_length = 40000 := by
  sorry

end NUMINAMATH_CALUDE_square_park_area_l3612_361256


namespace NUMINAMATH_CALUDE_f_passes_through_quadrants_234_l3612_361253

/-- A linear function f(x) = kx + b passes through the second, third, and fourth quadrants if and only if k < 0 and b < 0 -/
def passes_through_quadrants_234 (k b : ℝ) : Prop :=
  k < 0 ∧ b < 0

/-- The specific linear function f(x) = -2x - 1 -/
def f (x : ℝ) : ℝ := -2 * x - 1

/-- Theorem stating that f(x) = -2x - 1 passes through the second, third, and fourth quadrants -/
theorem f_passes_through_quadrants_234 :
  passes_through_quadrants_234 (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_f_passes_through_quadrants_234_l3612_361253


namespace NUMINAMATH_CALUDE_inequality_proof_l3612_361281

theorem inequality_proof (a b : ℝ) : a^2 + a*b + b^2 ≥ 3*(a + b - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3612_361281


namespace NUMINAMATH_CALUDE_division_result_approx_point_zero_seven_l3612_361286

-- Define the approximation tolerance
def tolerance : ℝ := 0.001

-- Define the condition that 35 divided by x is approximately 500
def divisionApprox (x : ℝ) : Prop := 
  abs (35 / x - 500) < tolerance

-- Theorem statement
theorem division_result_approx_point_zero_seven :
  ∃ x : ℝ, divisionApprox x ∧ abs (x - 0.07) < tolerance :=
sorry

end NUMINAMATH_CALUDE_division_result_approx_point_zero_seven_l3612_361286


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l3612_361210

-- Define a function to represent a three-digit number
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Define a function to represent a two-digit number
def two_digit_number (a b : ℕ) : ℕ := 10 * a + b

-- Define the condition for the problem
def satisfies_condition (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  three_digit_number a b c = 
    two_digit_number a b + two_digit_number a c +
    two_digit_number b a + two_digit_number b c +
    two_digit_number c a + two_digit_number c b

-- State the theorem
theorem three_digit_sum_theorem :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ 
    (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
    (a = 2 ∧ b = 6 ∧ c = 4) ∨ 
    (a = 3 ∧ b = 9 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l3612_361210


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3612_361240

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x, (deriv f) x - 2 * f x < 0) ∧ 
  f 0 = 1

/-- The main theorem -/
theorem solution_set_characterization (f : ℝ → ℝ) (h : satisfies_conditions f) :
  ∀ x, f x > Real.exp (2 * x) ↔ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3612_361240


namespace NUMINAMATH_CALUDE_total_amount_received_l3612_361269

/-- The amount John won in the lottery -/
def lottery_winnings : ℚ := 155250

/-- The number of top students receiving money -/
def num_students : ℕ := 100

/-- The fraction of the winnings given to each student -/
def fraction_given : ℚ := 1 / 1000

theorem total_amount_received (lottery_winnings : ℚ) (num_students : ℕ) (fraction_given : ℚ) :
  (lottery_winnings * fraction_given) * num_students = 15525 :=
sorry

end NUMINAMATH_CALUDE_total_amount_received_l3612_361269


namespace NUMINAMATH_CALUDE_pencil_store_theorem_l3612_361246

/-- Represents the store's pencil purchases and sales -/
structure PencilStore where
  first_purchase_cost : ℝ
  first_purchase_quantity : ℝ
  second_purchase_cost : ℝ
  second_purchase_quantity : ℝ
  selling_price : ℝ

/-- The conditions of the pencil store problem -/
def pencil_store_conditions (s : PencilStore) : Prop :=
  s.first_purchase_cost * s.first_purchase_quantity = 600 ∧
  s.second_purchase_cost * s.second_purchase_quantity = 600 ∧
  s.second_purchase_cost = (5/4) * s.first_purchase_cost ∧
  s.second_purchase_quantity = s.first_purchase_quantity - 30

/-- The profit calculation for the pencil store -/
def profit (s : PencilStore) : ℝ :=
  s.selling_price * (s.first_purchase_quantity + s.second_purchase_quantity) -
  (s.first_purchase_cost * s.first_purchase_quantity + s.second_purchase_cost * s.second_purchase_quantity)

/-- The main theorem about the pencil store problem -/
theorem pencil_store_theorem (s : PencilStore) :
  pencil_store_conditions s →
  s.first_purchase_cost = 4 ∧
  (∀ p, profit { s with selling_price := p } ≥ 420 → p ≥ 6) :=
by sorry


end NUMINAMATH_CALUDE_pencil_store_theorem_l3612_361246


namespace NUMINAMATH_CALUDE_square_circle_distance_sum_constant_l3612_361265

/-- Given a square ABCD with side length 2a and a circle k centered at the center of the square with radius R, 
    the sum of squared distances from any point P on the circle to the vertices of the square is constant. -/
theorem square_circle_distance_sum_constant 
  (a R : ℝ) 
  (A B C D : ℝ × ℝ) 
  (k : Set (ℝ × ℝ)) 
  (h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a))
  (h_circle : k = {P : ℝ × ℝ | P.1^2 + P.2^2 = R^2}) :
  ∀ P ∈ k, 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 + 
    (P.1 - B.1)^2 + (P.2 - B.2)^2 + 
    (P.1 - C.1)^2 + (P.2 - C.2)^2 + 
    (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4*R^2 + 8*a^2 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_distance_sum_constant_l3612_361265


namespace NUMINAMATH_CALUDE_quick_multiply_correct_l3612_361278

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- The quick multiplication formula for two-digit numbers with reversed digits -/
def quickMultiply (x y : Nat) : Nat :=
  101 * x * y + 10 * (x^2 + y^2)

/-- Theorem stating that the quick multiplication formula is correct -/
theorem quick_multiply_correct (x y : Nat) (h1 : x < 10) (h2 : y < 10) :
  (TwoDigitNumber x y) * (TwoDigitNumber y x) = quickMultiply x y :=
by
  sorry

end NUMINAMATH_CALUDE_quick_multiply_correct_l3612_361278


namespace NUMINAMATH_CALUDE_caravan_keeper_count_l3612_361287

/-- Represents the number of keepers in the caravan -/
def num_keepers : ℕ := 10

/-- Represents the number of hens in the caravan -/
def num_hens : ℕ := 60

/-- Represents the number of goats in the caravan -/
def num_goats : ℕ := 35

/-- Represents the number of camels in the caravan -/
def num_camels : ℕ := 6

/-- Represents the number of feet for a hen -/
def hen_feet : ℕ := 2

/-- Represents the number of feet for a goat or camel -/
def goat_camel_feet : ℕ := 4

/-- Represents the number of feet for a keeper -/
def keeper_feet : ℕ := 2

/-- Represents the difference between total feet and total heads -/
def feet_head_difference : ℕ := 193

theorem caravan_keeper_count :
  num_keepers * keeper_feet +
  num_hens * hen_feet +
  num_goats * goat_camel_feet +
  num_camels * goat_camel_feet =
  (num_keepers + num_hens + num_goats + num_camels + feet_head_difference) :=
by sorry

end NUMINAMATH_CALUDE_caravan_keeper_count_l3612_361287


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3612_361285

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_seven_balls_three_boxes :
  distribute_balls 7 3 = 36 := by
  sorry


end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l3612_361285


namespace NUMINAMATH_CALUDE_brothers_age_difference_l3612_361216

theorem brothers_age_difference (a b : ℕ) : 
  a > 0 → b > 0 → a + b = 60 → 3 * b = 2 * a → a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l3612_361216


namespace NUMINAMATH_CALUDE_eunji_remaining_confetti_l3612_361200

def initial_green_confetti : ℕ := 9
def initial_red_confetti : ℕ := 1
def confetti_given_away : ℕ := 4

theorem eunji_remaining_confetti :
  initial_green_confetti + initial_red_confetti - confetti_given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_eunji_remaining_confetti_l3612_361200


namespace NUMINAMATH_CALUDE_students_per_group_l3612_361231

theorem students_per_group (total_students : Nat) (num_teachers : Nat) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) : 
  total_students / num_teachers = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3612_361231


namespace NUMINAMATH_CALUDE_gcd_p4_minus_1_l3612_361202

theorem gcd_p4_minus_1 (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) : 
  ∃ k : ℕ, p^4 - 1 = 240 * k := by
sorry

end NUMINAMATH_CALUDE_gcd_p4_minus_1_l3612_361202


namespace NUMINAMATH_CALUDE_fruit_problem_solution_l3612_361217

def fruit_problem (cost_A cost_B : ℝ) (weight_diff : ℝ) (total_weight : ℝ) 
  (selling_price_A selling_price_B : ℝ) : Prop :=
  ∃ (weight_A weight_B cost_per_kg_A cost_per_kg_B : ℝ),
    cost_A = weight_A * cost_per_kg_A ∧
    cost_B = weight_B * cost_per_kg_B ∧
    cost_per_kg_B = 1.5 * cost_per_kg_A ∧
    weight_A = weight_B + weight_diff ∧
    (∀ a b, a + b = total_weight ∧ a ≥ 3 * b →
      (13 - cost_per_kg_A) * a + (20 - cost_per_kg_B) * b ≤
      (13 - cost_per_kg_A) * 75 + (20 - cost_per_kg_B) * 25) ∧
    cost_per_kg_A = 10 ∧
    cost_per_kg_B = 15

theorem fruit_problem_solution :
  fruit_problem 300 300 10 100 13 20 :=
sorry

end NUMINAMATH_CALUDE_fruit_problem_solution_l3612_361217


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3612_361235

theorem quadratic_equation_solution (t s : ℝ) : t = 8 * s^2 + 2 * s → t = 5 →
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3612_361235


namespace NUMINAMATH_CALUDE_smallest_abb_value_l3612_361273

theorem smallest_abb_value (A B : Nat) : 
  A ≠ B →
  1 ≤ A ∧ A ≤ 9 →
  1 ≤ B ∧ B ≤ 9 →
  10 * A + B = (100 * A + 11 * B) / 7 →
  ∀ (X Y : Nat), 
    X ≠ Y →
    1 ≤ X ∧ X ≤ 9 →
    1 ≤ Y ∧ Y ≤ 9 →
    10 * X + Y = (100 * X + 11 * Y) / 7 →
    100 * A + 11 * B ≤ 100 * X + 11 * Y →
  100 * A + 11 * B = 466 :=
sorry

end NUMINAMATH_CALUDE_smallest_abb_value_l3612_361273


namespace NUMINAMATH_CALUDE_min_packages_correct_l3612_361203

/-- The minimum number of packages Mary must deliver to cover the cost of her bicycle -/
def min_packages : ℕ :=
  let bicycle_cost : ℕ := 800
  let revenue_per_package : ℕ := 12
  let maintenance_cost_per_package : ℕ := 4
  let profit_per_package : ℕ := revenue_per_package - maintenance_cost_per_package
  (bicycle_cost + profit_per_package - 1) / profit_per_package

theorem min_packages_correct : min_packages = 100 := by
  sorry

end NUMINAMATH_CALUDE_min_packages_correct_l3612_361203


namespace NUMINAMATH_CALUDE_evaluate_expression_l3612_361299

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3612_361299


namespace NUMINAMATH_CALUDE_exercise_book_count_l3612_361279

/-- Given a ratio of pencils to pens to exercise books and the number of pencils,
    calculate the number of exercise books. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 14) (h2 : pen_ratio = 4) (h3 : book_ratio = 3) 
    (h4 : pencil_count = 140) : 
    (pencil_count * book_ratio) / pencil_ratio = 30 := by
  sorry

end NUMINAMATH_CALUDE_exercise_book_count_l3612_361279


namespace NUMINAMATH_CALUDE_pizza_topping_combinations_l3612_361236

theorem pizza_topping_combinations (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by sorry

end NUMINAMATH_CALUDE_pizza_topping_combinations_l3612_361236


namespace NUMINAMATH_CALUDE_y_value_proof_l3612_361239

theorem y_value_proof (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l3612_361239


namespace NUMINAMATH_CALUDE_direction_vector_b_l3612_361257

/-- A line passing through two points with a specific direction vector form -/
def Line (p1 p2 : ℝ × ℝ) (dir : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = (p1.1 + t * dir.1, p1.2 + t * dir.2)

theorem direction_vector_b (b : ℝ) :
  Line (-3, 0) (0, 3) (3, b) → b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_direction_vector_b_l3612_361257


namespace NUMINAMATH_CALUDE_first_job_men_l3612_361262

/-- The number of men who worked on the first job -/
def M : ℕ := 250

/-- The number of days for the first job -/
def days_job1 : ℕ := 16

/-- The number of men working on the second job -/
def men_job2 : ℕ := 600

/-- The number of days for the second job -/
def days_job2 : ℕ := 20

/-- The ratio of work between the second and first job -/
def work_ratio : ℕ := 3

theorem first_job_men :
  M * days_job1 * work_ratio = men_job2 * days_job2 := by
  sorry

#check first_job_men

end NUMINAMATH_CALUDE_first_job_men_l3612_361262


namespace NUMINAMATH_CALUDE_inequality_reversal_l3612_361204

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(-x > -y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l3612_361204


namespace NUMINAMATH_CALUDE_calculation_proof_l3612_361258

theorem calculation_proof : (4.5 - 1.23) * 2.1 = 6.867 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3612_361258


namespace NUMINAMATH_CALUDE_total_points_is_63_l3612_361282

/-- The total points scored by Zach and Ben in a football game -/
def total_points (zach_points ben_points : Float) : Float :=
  zach_points + ben_points

/-- Theorem stating that the total points scored by Zach and Ben is 63.0 -/
theorem total_points_is_63 :
  total_points 42.0 21.0 = 63.0 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_63_l3612_361282


namespace NUMINAMATH_CALUDE_simplify_expression_l3612_361247

theorem simplify_expression (b : ℝ) : (1:ℝ) * (3*b) * (5*b^2) * (7*b^3) * (9*b^4) = 945 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3612_361247


namespace NUMINAMATH_CALUDE_burrito_combinations_l3612_361254

def number_of_ways_to_make_burritos : ℕ :=
  let max_beef := 4
  let max_chicken := 3
  let total_wraps := 5
  (Nat.choose total_wraps 3) + (Nat.choose total_wraps 2) + (Nat.choose total_wraps 1)

theorem burrito_combinations : number_of_ways_to_make_burritos = 25 := by
  sorry

end NUMINAMATH_CALUDE_burrito_combinations_l3612_361254


namespace NUMINAMATH_CALUDE_root_of_inverse_point_l3612_361293

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
variable (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)

-- Assume f_inv(0) = 2
variable (h_f_inv_zero : f_inv 0 = 2)

-- Theorem: If f_inv(0) = 2, then f(2) = 0
theorem root_of_inverse_point (f f_inv : ℝ → ℝ) 
  (h_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x) 
  (h_f_inv_zero : f_inv 0 = 2) : 
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_root_of_inverse_point_l3612_361293


namespace NUMINAMATH_CALUDE_printer_X_time_proof_l3612_361207

/-- The time it takes for printer X to complete the job alone -/
def printerX_time : ℝ := 16

/-- The time it takes for printer Y to complete the job alone -/
def printerY_time : ℝ := 10

/-- The time it takes for printer Z to complete the job alone -/
def printerZ_time : ℝ := 20

/-- The ratio of printer X's time to the combined time of printers Y and Z -/
def time_ratio : ℝ := 2.4

theorem printer_X_time_proof :
  printerX_time = 16 ∧
  printerY_time = 10 ∧
  printerZ_time = 20 ∧
  time_ratio = 2.4 →
  printerX_time = time_ratio * (1 / (1 / printerY_time + 1 / printerZ_time)) :=
by
  sorry

end NUMINAMATH_CALUDE_printer_X_time_proof_l3612_361207


namespace NUMINAMATH_CALUDE_marble_probability_theorem_l3612_361205

/-- Represents a box containing marbles -/
structure Box where
  total : ℕ
  red : ℕ
  blue : ℕ
  hSum : red + blue = total

/-- The probability of drawing a red marble from a box -/
def redProb (b : Box) : ℚ :=
  b.red / b.total

/-- The probability of drawing a blue marble from a box -/
def blueProb (b : Box) : ℚ :=
  b.blue / b.total

/-- The main theorem -/
theorem marble_probability_theorem
  (box1 box2 : Box)
  (hTotal : box1.total + box2.total = 30)
  (hRedProb : redProb box1 * redProb box2 = 2/3) :
  blueProb box1 * blueProb box2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_theorem_l3612_361205


namespace NUMINAMATH_CALUDE_ideal_gas_entropy_change_l3612_361289

/-- Entropy change for an ideal gas under different conditions -/
theorem ideal_gas_entropy_change
  (m μ R Cp Cv : ℝ)
  (P V T P1 P2 V1 V2 T1 T2 : ℝ)
  (h_ideal_gas : P * V = (m / μ) * R * T)
  (h_m_pos : m > 0)
  (h_μ_pos : μ > 0)
  (h_R_pos : R > 0)
  (h_Cp_pos : Cp > 0)
  (h_Cv_pos : Cv > 0)
  (h_P_pos : P > 0)
  (h_V_pos : V > 0)
  (h_T_pos : T > 0)
  (h_P1_pos : P1 > 0)
  (h_P2_pos : P2 > 0)
  (h_V1_pos : V1 > 0)
  (h_V2_pos : V2 > 0)
  (h_T1_pos : T1 > 0)
  (h_T2_pos : T2 > 0) :
  (∃ ΔS : ℝ,
    (P = P1 ∧ P = P2 → ΔS = (m / μ) * Cp * Real.log (V2 / V1)) ∧
    (V = V1 ∧ V = V2 → ΔS = (m / μ) * Cv * Real.log (P2 / P1)) ∧
    (T = T1 ∧ T = T2 → ΔS = (m / μ) * R * Real.log (V2 / V1))) :=
by sorry

end NUMINAMATH_CALUDE_ideal_gas_entropy_change_l3612_361289


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l3612_361275

theorem absolute_value_equation_solution_count :
  ∃! (s : Finset ℤ), (∀ a ∈ s, |3*a+7| + |3*a-5| = 12) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l3612_361275


namespace NUMINAMATH_CALUDE_balance_equals_132_l3612_361219

/-- Calculates the account balance after two years given an initial deposit,
    annual interest rate, and additional annual deposit. -/
def account_balance_after_two_years (initial_deposit : ℝ) (interest_rate : ℝ) (annual_deposit : ℝ) : ℝ :=
  let balance_after_first_year := initial_deposit * (1 + interest_rate) + annual_deposit
  balance_after_first_year * (1 + interest_rate) + annual_deposit

/-- Theorem stating that given the specified conditions, the account balance
    after two years will be $132. -/
theorem balance_equals_132 :
  account_balance_after_two_years 100 0.1 10 = 132 := by
  sorry

#eval account_balance_after_two_years 100 0.1 10

end NUMINAMATH_CALUDE_balance_equals_132_l3612_361219


namespace NUMINAMATH_CALUDE_limits_zero_l3612_361215

open Real

theorem limits_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |n / (10 : ℝ)^n| < ε) ∧ 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |log n / n| < ε) := by
  sorry

end NUMINAMATH_CALUDE_limits_zero_l3612_361215


namespace NUMINAMATH_CALUDE_car_distance_proof_l3612_361248

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc h => acc + initialSpeed + h * speedIncrease) 0

/-- Proves that a car traveling 45 km in the first hour and increasing speed by 2 km/h
    each hour will travel 672 km in 12 hours. -/
theorem car_distance_proof :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end NUMINAMATH_CALUDE_car_distance_proof_l3612_361248


namespace NUMINAMATH_CALUDE_range_of_a_l3612_361271

def set_A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

def set_B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : ℝ) : set_A a ∩ set_B = ∅ → 2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3612_361271


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3612_361263

theorem quadratic_equation_roots (m : ℝ) :
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3612_361263


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l3612_361222

theorem sugar_solution_percentage (initial_sugar_percentage : ℝ) 
  (replaced_fraction : ℝ) (final_sugar_percentage : ℝ) :
  initial_sugar_percentage = 22 →
  replaced_fraction = 1/4 →
  final_sugar_percentage = 35 →
  let remaining_fraction := 1 - replaced_fraction
  let initial_sugar := initial_sugar_percentage * remaining_fraction
  let added_sugar := final_sugar_percentage - initial_sugar
  added_sugar / replaced_fraction = 74 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l3612_361222


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l3612_361264

theorem crazy_silly_school_series (num_books : ℕ) (movies_watched : ℕ) (books_read : ℕ) :
  num_books = 8 →
  movies_watched = 19 →
  books_read = 16 →
  movies_watched = books_read + 3 →
  ∃ (num_movies : ℕ), num_movies ≥ 19 :=
by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l3612_361264


namespace NUMINAMATH_CALUDE_cube_expansion_seven_plus_one_l3612_361252

theorem cube_expansion_seven_plus_one : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_expansion_seven_plus_one_l3612_361252


namespace NUMINAMATH_CALUDE_asymptote_sum_l3612_361277

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) where A, B, C are integers,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + A*x^2 + B*x + C)) →
  A + B + C = -9 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3612_361277


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3612_361234

/-- An arithmetic sequence with given third and eleventh terms -/
def ArithmeticSequence (a₃ a₁₁ : ℚ) :=
  ∃ (a₁ d : ℚ), a₃ = a₁ + 2 * d ∧ a₁₁ = a₁ + 10 * d

/-- Theorem stating the first term and common difference of the sequence -/
theorem arithmetic_sequence_solution :
  ∀ (a₃ a₁₁ : ℚ), a₃ = 3 ∧ a₁₁ = 15 →
  ArithmeticSequence a₃ a₁₁ →
  ∃ (a₁ d : ℚ), a₁ = 0 ∧ d = 3/2 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3612_361234


namespace NUMINAMATH_CALUDE_system_solution_l3612_361280

theorem system_solution : 
  let x : ℚ := 25 / 31
  let y : ℚ := -11 / 31
  (3 * x + 4 * y = 1) ∧ (7 * x - y = 6) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3612_361280


namespace NUMINAMATH_CALUDE_work_completion_time_l3612_361214

theorem work_completion_time
  (A_work : ℝ)
  (B_work : ℝ)
  (C_work : ℝ)
  (h1 : A_work = 1 / 3)
  (h2 : B_work + C_work = 1 / 3)
  (h3 : B_work = 1 / 6)
  : 1 / (A_work + C_work) = 2 :=
by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3612_361214


namespace NUMINAMATH_CALUDE_johns_former_wage_l3612_361297

/-- Represents John's work schedule and wage information -/
structure WorkInfo where
  hours_per_workday : ℕ
  days_between_workdays : ℕ
  monthly_pay : ℕ
  days_in_month : ℕ
  raise_percentage : ℚ

/-- Calculates the former hourly wage given the work information -/
def former_hourly_wage (info : WorkInfo) : ℚ :=
  let days_worked := info.days_in_month / (info.days_between_workdays + 1)
  let total_hours := days_worked * info.hours_per_workday
  let current_hourly_wage := info.monthly_pay / total_hours
  current_hourly_wage / (1 + info.raise_percentage)

/-- Theorem stating that John's former hourly wage was $20 -/
theorem johns_former_wage (info : WorkInfo) 
  (h1 : info.hours_per_workday = 12)
  (h2 : info.days_between_workdays = 1)
  (h3 : info.monthly_pay = 4680)
  (h4 : info.days_in_month = 30)
  (h5 : info.raise_percentage = 3/10) :
  former_hourly_wage info = 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_former_wage_l3612_361297


namespace NUMINAMATH_CALUDE_comic_books_triple_storybooks_l3612_361259

/-- The number of days after which the number of comic books is three times the number of storybooks -/
def days_until_triple_ratio : ℕ := 20

/-- The initial number of comic books -/
def initial_comic_books : ℕ := 140

/-- The initial number of storybooks -/
def initial_storybooks : ℕ := 100

/-- The number of books borrowed per day for each type -/
def daily_borrowing_rate : ℕ := 4

theorem comic_books_triple_storybooks :
  initial_comic_books - days_until_triple_ratio * daily_borrowing_rate =
  3 * (initial_storybooks - days_until_triple_ratio * daily_borrowing_rate) := by
  sorry

end NUMINAMATH_CALUDE_comic_books_triple_storybooks_l3612_361259


namespace NUMINAMATH_CALUDE_mabels_garden_petal_count_l3612_361294

/-- The number of petals remaining in Mabel's garden after a series of events -/
def final_petal_count (initial_daisies : ℕ) (initial_petals_per_daisy : ℕ) 
  (daisies_given_away : ℕ) (new_daisies : ℕ) (new_petals_per_daisy : ℕ) 
  (petals_lost_new_daisies : ℕ) (petals_lost_original_daisies : ℕ) : ℕ :=
  let initial_petals := initial_daisies * initial_petals_per_daisy
  let remaining_petals := initial_petals - (daisies_given_away * initial_petals_per_daisy)
  let new_petals := new_daisies * new_petals_per_daisy
  let total_petals := remaining_petals + new_petals
  total_petals - (petals_lost_new_daisies + petals_lost_original_daisies)

/-- Theorem stating that the final petal count in Mabel's garden is 39 -/
theorem mabels_garden_petal_count :
  final_petal_count 5 8 2 3 7 4 2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_mabels_garden_petal_count_l3612_361294


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l3612_361267

def last_two_digits (n : ℕ) : ℕ := n % 100

def power_pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_power (n : ℕ) :
  last_two_digits (7^n) = power_pattern n :=
sorry

theorem last_two_digits_of_7_2017 :
  last_two_digits (7^2017) = 07 :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_last_two_digits_of_7_2017_l3612_361267


namespace NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3612_361229

theorem dot_product_of_specific_vectors :
  let A : ℝ × ℝ := (Real.cos (110 * π / 180), Real.sin (110 * π / 180))
  let B : ℝ × ℝ := (Real.cos (50 * π / 180), Real.sin (50 * π / 180))
  let OA : ℝ × ℝ := A
  let OB : ℝ × ℝ := B
  (OA.1 * OB.1 + OA.2 * OB.2) = 1/2 := by
sorry


end NUMINAMATH_CALUDE_dot_product_of_specific_vectors_l3612_361229


namespace NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l3612_361295

/-- The perimeter of a rectangle composed of three squares with perimeter 24 each
    and three rectangles with perimeter 16 each is 52. -/
theorem large_rectangle_perimeter : ℝ → Prop :=
  fun (perimeter : ℝ) =>
    let square_perimeter := 24
    let small_rectangle_perimeter := 16
    let square_side := square_perimeter / 4
    let small_rectangle_width := (small_rectangle_perimeter / 2) - square_side
    let large_rectangle_height := square_side + small_rectangle_width
    let large_rectangle_width := 3 * square_side
    perimeter = 2 * (large_rectangle_height + large_rectangle_width) ∧
    perimeter = 52

/-- Proof of the theorem -/
theorem large_rectangle_perimeter_proof : large_rectangle_perimeter 52 := by
  sorry

#check large_rectangle_perimeter_proof

end NUMINAMATH_CALUDE_large_rectangle_perimeter_large_rectangle_perimeter_proof_l3612_361295


namespace NUMINAMATH_CALUDE_sum_of_ratios_bound_l3612_361238

theorem sum_of_ratios_bound (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y + z) + y / (z + x) + z / (x + y) ≥ (3 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_ratios_bound_l3612_361238


namespace NUMINAMATH_CALUDE_decimal_product_sum_l3612_361226

-- Define the structure for our decimal representation
structure DecimalPair :=
  (whole : Nat)
  (decimal : Nat)

-- Define the multiplication operation for DecimalPair
def multiply_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) * (y.whole + y.decimal / 10 : Rat)

-- Define the addition operation for DecimalPair
def add_decimal_pairs (x y : DecimalPair) : Rat :=
  (x.whole + x.decimal / 10 : Rat) + (y.whole + y.decimal / 10 : Rat)

-- The main theorem
theorem decimal_product_sum (a b c d : Nat) :
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) → (d ≠ 0) →
  (a ≤ 9) → (b ≤ 9) → (c ≤ 9) → (d ≤ 9) →
  multiply_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (56 : Rat) / 10 →
  add_decimal_pairs ⟨a, b⟩ ⟨c, d⟩ = (51 : Rat) / 10 := by
sorry

end NUMINAMATH_CALUDE_decimal_product_sum_l3612_361226


namespace NUMINAMATH_CALUDE_roger_donated_66_coins_l3612_361228

/-- Represents the number of coins Roger donated -/
def coins_donated (pennies nickels dimes coins_left : ℕ) : ℕ :=
  pennies + nickels + dimes - coins_left

/-- Proves that Roger donated 66 coins given the initial counts and remaining coins -/
theorem roger_donated_66_coins (h1 : coins_donated 42 36 15 27 = 66) : 
  coins_donated 42 36 15 27 = 66 := by
  sorry

end NUMINAMATH_CALUDE_roger_donated_66_coins_l3612_361228
