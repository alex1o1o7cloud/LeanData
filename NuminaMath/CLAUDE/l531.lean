import Mathlib

namespace NUMINAMATH_CALUDE_product_of_logs_l531_53194

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 1560) →
  (Real.log b / Real.log a = 3) →
  (a + b : ℕ) = 1740 := by sorry

end NUMINAMATH_CALUDE_product_of_logs_l531_53194


namespace NUMINAMATH_CALUDE_train_length_l531_53150

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), 
  (abs (length_m - (speed_kmh * 1000 / 3600 * time_s)) < 0.01) ∧ 
  (abs (length_m - 140) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_train_length_l531_53150


namespace NUMINAMATH_CALUDE_customers_per_table_l531_53102

theorem customers_per_table 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (num_tables : ℕ) 
  (h1 : initial_customers = 21)
  (h2 : left_customers = 12)
  (h3 : num_tables = 3)
  (h4 : num_tables > 0)
  : (initial_customers - left_customers) / num_tables = 3 := by
sorry

end NUMINAMATH_CALUDE_customers_per_table_l531_53102


namespace NUMINAMATH_CALUDE_system_solution_l531_53172

theorem system_solution (x y : ℝ) (h1 : x + 2*y = 8) (h2 : 2*x + y = 7) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l531_53172


namespace NUMINAMATH_CALUDE_log_inequality_l531_53176

theorem log_inequality (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x) / (x + 1) + 1 / x > (Real.log x) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l531_53176


namespace NUMINAMATH_CALUDE_keith_attended_four_games_l531_53178

/-- The number of football games Keith attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

theorem keith_attended_four_games :
  let total_games : ℕ := 8
  let missed_games : ℕ := 4
  games_attended total_games missed_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_attended_four_games_l531_53178


namespace NUMINAMATH_CALUDE_abc_zero_l531_53173

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) :
  a * b * c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l531_53173


namespace NUMINAMATH_CALUDE_prudence_weekend_sleep_l531_53112

/-- Represents Prudence's sleep schedule over 4 weeks -/
structure SleepSchedule where
  weekdayNightSleep : ℕ  -- Hours of sleep on weeknights (Sun-Thu)
  weekendNapHours : ℕ    -- Hours of nap on weekend days
  totalSleepHours : ℕ    -- Total hours of sleep in 4 weeks
  weekdayNights : ℕ      -- Number of weekday nights in 4 weeks
  weekendNights : ℕ      -- Number of weekend nights in 4 weeks
  weekendDays : ℕ        -- Number of weekend days in 4 weeks

/-- Calculates the hours of sleep per night on weekends given Prudence's sleep schedule -/
def weekendNightSleep (s : SleepSchedule) : ℚ :=
  let weekdaySleep := s.weekdayNightSleep * s.weekdayNights
  let weekendNapSleep := s.weekendNapHours * s.weekendDays
  let remainingSleep := s.totalSleepHours - weekdaySleep - weekendNapSleep
  remainingSleep / s.weekendNights

/-- Theorem stating that Prudence sleeps 9 hours per night on weekends -/
theorem prudence_weekend_sleep (s : SleepSchedule)
  (h1 : s.weekdayNightSleep = 6)
  (h2 : s.weekendNapHours = 1)
  (h3 : s.totalSleepHours = 200)
  (h4 : s.weekdayNights = 20)
  (h5 : s.weekendNights = 8)
  (h6 : s.weekendDays = 8) :
  weekendNightSleep s = 9 := by
  sorry

#eval weekendNightSleep {
  weekdayNightSleep := 6,
  weekendNapHours := 1,
  totalSleepHours := 200,
  weekdayNights := 20,
  weekendNights := 8,
  weekendDays := 8
}

end NUMINAMATH_CALUDE_prudence_weekend_sleep_l531_53112


namespace NUMINAMATH_CALUDE_oliver_stickers_l531_53154

theorem oliver_stickers (S : ℕ) : 
  (3/5 : ℚ) * (2/3 : ℚ) * S = 54 → S = 135 := by
sorry

end NUMINAMATH_CALUDE_oliver_stickers_l531_53154


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l531_53164

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Prime p → 
  Prime (x / (12 * p)) → 
  Even (x / (12 * p)) → 
  (∀ y : ℕ, y > 0 → Prime p → Prime (y / (12 * p)) → Even (y / (12 * p)) → x ≤ y) → 
  x = 48 :=
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l531_53164


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l531_53116

theorem diophantine_equation_solution (x y z t : ℤ) : 
  x^4 - 2*y^4 - 4*z^4 - 8*t^4 = 0 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l531_53116


namespace NUMINAMATH_CALUDE_count_multiples_of_three_l531_53105

/-- An arithmetic sequence with first term 9 and 8th term 12 -/
structure ArithmeticSequence where
  a₁ : ℕ
  a₈ : ℕ
  h₁ : a₁ = 9
  h₈ : a₈ = 12

/-- The number of terms among the first 2015 that are multiples of 3 -/
def multiples_of_three (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem -/
theorem count_multiples_of_three (seq : ArithmeticSequence) :
  multiples_of_three seq = 288 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_three_l531_53105


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l531_53160

theorem diophantine_equation_solutions (x y : ℕ) : 
  2^(2*x + 1) + 2^x + 1 = y^2 ↔ (x = 4 ∧ y = 23) ∨ (x = 0 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l531_53160


namespace NUMINAMATH_CALUDE_initial_average_age_l531_53199

/-- Given a group of people with an unknown initial average age, 
    prove that when a new person joins and changes the average, 
    we can determine the initial average age. -/
theorem initial_average_age 
  (n : ℕ) 
  (new_person_age : ℕ) 
  (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : new_person_age = 37)
  (h3 : new_average = 17) :
  ∃ (initial_average : ℝ),
    n * initial_average + new_person_age = (n + 1) * new_average ∧ 
    initial_average = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_age_l531_53199


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l531_53153

theorem cubic_equation_solution :
  ∀ x y : ℕ+, x^3 - y^3 = 999 ↔ (x = 12 ∧ y = 9) ∨ (x = 10 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l531_53153


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l531_53101

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 6 4 n = 206 ∧ n = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l531_53101


namespace NUMINAMATH_CALUDE_expand_polynomial_l531_53139

theorem expand_polynomial (x : ℝ) : (-2*x - 1) * (3*x - 2) = -6*x^2 + x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l531_53139


namespace NUMINAMATH_CALUDE_playground_count_l531_53184

/-- The total number of people on the playground after late arrivals -/
def total_people (initial_boys initial_girls teachers late_boys late_girls : ℕ) : ℕ :=
  initial_boys + initial_girls + teachers + late_boys + late_girls

/-- Theorem stating the total number of people on the playground after late arrivals -/
theorem playground_count : total_people 44 53 5 3 2 = 107 := by
  sorry

end NUMINAMATH_CALUDE_playground_count_l531_53184


namespace NUMINAMATH_CALUDE_max_value_theorem_l531_53109

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (2 * x + y)) + (y / (x + 2 * y)) ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l531_53109


namespace NUMINAMATH_CALUDE_min_value_on_circle_l531_53181

theorem min_value_on_circle (x y : ℝ) :
  (x - 2)^2 + (y - 3)^2 = 1 →
  ∃ (z : ℝ), z = 14 - 2 * Real.sqrt 13 ∧ ∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → x^2 + y^2 ≤ a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l531_53181


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l531_53159

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 120 ∧ 
  n % 8 = 5 ∧ 
  ∀ m : ℕ, m < 120 ∧ m % 8 = 5 → m ≤ n → 
  n = 117 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l531_53159


namespace NUMINAMATH_CALUDE_average_equation_l531_53100

theorem average_equation (x y : ℚ) : 
  x = 50 / 11399 ∧ y = -11275 / 151 →
  (List.sum (List.range 150) + x + y) / 152 = 75 * x + y := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l531_53100


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l531_53195

theorem integer_solutions_quadratic_equation :
  ∀ m n : ℤ, n^2 - 3*m*n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l531_53195


namespace NUMINAMATH_CALUDE_remainder_11_power_2023_mod_19_l531_53165

theorem remainder_11_power_2023_mod_19 : 11^2023 % 19 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_2023_mod_19_l531_53165


namespace NUMINAMATH_CALUDE_smallest_number_l531_53169

theorem smallest_number (a b c d : ℝ) (h1 : a = 2) (h2 : b = -2.5) (h3 : c = 0) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l531_53169


namespace NUMINAMATH_CALUDE_zongzi_sales_and_profit_l531_53146

/-- The daily sales volume function for zongzi -/
def sales_volume (x : ℝ) : ℝ := 800 * x + 400

/-- The maximum daily production of zongzi -/
def max_production : ℝ := 1100

/-- The initial profit per zongzi in yuan -/
def initial_profit : ℝ := 2

/-- The total profit function for zongzi sales -/
def total_profit (x : ℝ) : ℝ := (initial_profit - x) * sales_volume x

theorem zongzi_sales_and_profit :
  (sales_volume 0.2 = 560) ∧ 
  (total_profit 0.2 = 1008) ∧
  (∃ x : ℝ, total_profit x = 1200 ∧ x = 0.5 ∧ sales_volume x ≤ max_production) :=
by sorry

end NUMINAMATH_CALUDE_zongzi_sales_and_profit_l531_53146


namespace NUMINAMATH_CALUDE_regular_decagon_perimeter_l531_53162

/-- A regular decagon is a polygon with 10 sides of equal length -/
def RegularDecagon := Nat

/-- The side length of a regular decagon -/
def sideLength (d : RegularDecagon) : ℝ := 3

/-- The perimeter of a regular decagon -/
def perimeter (d : RegularDecagon) : ℝ := 10 * sideLength d

theorem regular_decagon_perimeter (d : RegularDecagon) : 
  perimeter d = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_decagon_perimeter_l531_53162


namespace NUMINAMATH_CALUDE_expression_evaluation_l531_53143

theorem expression_evaluation :
  let a : ℚ := 4/3
  (7 * a^2 - 15 * a + 2) * (3 * a - 4) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l531_53143


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l531_53156

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l531_53156


namespace NUMINAMATH_CALUDE_baker_productivity_l531_53179

/-- The number of ovens the baker has -/
def num_ovens : ℕ := 4

/-- The number of hours the baker bakes on weekdays -/
def weekday_hours : ℕ := 5

/-- The number of hours the baker bakes on weekend days -/
def weekend_hours : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of weeks the baker bakes -/
def num_weeks : ℕ := 3

/-- The total number of loaves baked in 3 weeks -/
def total_loaves : ℕ := 1740

/-- The number of loaves baked per hour in one oven -/
def loaves_per_hour : ℚ :=
  total_loaves / (num_ovens * (weekdays * weekday_hours + weekend_days * weekend_hours) * num_weeks)

theorem baker_productivity : loaves_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_baker_productivity_l531_53179


namespace NUMINAMATH_CALUDE_set_difference_equals_interval_l531_53190

def M : Set ℝ := {x | x^2 + x - 12 ≤ 0}

def N : Set ℝ := {y | ∃ x, y = 3^x ∧ x ≤ 1}

theorem set_difference_equals_interval :
  {x | x ∈ M ∧ x ∉ N} = Set.Ico (-4) 0 := by sorry

end NUMINAMATH_CALUDE_set_difference_equals_interval_l531_53190


namespace NUMINAMATH_CALUDE_marias_car_trip_l531_53198

theorem marias_car_trip (D : ℝ) : 
  (D / 2 / 4 / 3 + D / 2 / 4 * 2 / 3 + D / 2 * 3 / 4) = 630 → D = 840 := by
  sorry

end NUMINAMATH_CALUDE_marias_car_trip_l531_53198


namespace NUMINAMATH_CALUDE_problem_statement_l531_53157

theorem problem_statement (a b c d : ℝ) : 
  (Real.sqrt (a + b + c + d) + Real.sqrt (a^2 - 2*a + 3 - b) - Real.sqrt (b - c^2 + 4*c - 8) = 3) →
  (a - b + c - d = -7) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l531_53157


namespace NUMINAMATH_CALUDE_perfect_square_condition_l531_53121

theorem perfect_square_condition (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + k)^2) → (m = 6 ∨ m = -6) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l531_53121


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l531_53147

theorem geometric_progression_proof (x : ℝ) :
  (30 + x)^2 = (10 + x) * (90 + x) →
  x = 0 ∧ (30 + x) / (10 + x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l531_53147


namespace NUMINAMATH_CALUDE_stratified_sampling_c_l531_53151

/-- Represents the number of individuals in each sample -/
structure SampleSizes where
  A : ℕ
  B : ℕ
  C : ℕ

/-- The ratio of individuals in samples A, B, and C -/
def sample_ratio : SampleSizes := { A := 5, B := 3, C := 2 }

/-- The total sample size for stratified sampling -/
def total_sample_size : ℕ := 100

/-- Calculates the number of individuals to be drawn from a specific sample -/
def stratified_sample_size (ratio : ℕ) : ℕ :=
  (total_sample_size * ratio) / (sample_ratio.A + sample_ratio.B + sample_ratio.C)

theorem stratified_sampling_c :
  stratified_sample_size sample_ratio.C = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_c_l531_53151


namespace NUMINAMATH_CALUDE_function_equality_l531_53128

theorem function_equality (f : ℕ+ → ℕ+) 
  (h : ∀ m n : ℕ+, m^2 + f n^2 + (m - f n)^2 ≥ f m^2 + n^2) : 
  ∀ n : ℕ+, f n = n :=
sorry

end NUMINAMATH_CALUDE_function_equality_l531_53128


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l531_53114

/-- Represents the juice extraction information for a fruit type -/
structure FruitJuice where
  fruitCount : ℕ
  juiceAmount : ℕ

/-- Represents the blend composition -/
structure Blend where
  pearCount : ℕ
  orangeCount : ℕ

def calculateJuicePercentage (pearJuice : FruitJuice) (orangeJuice : FruitJuice) (blend : Blend) : ℚ :=
  let pearJuiceRate := pearJuice.juiceAmount / pearJuice.fruitCount
  let orangeJuiceRate := orangeJuice.juiceAmount / orangeJuice.fruitCount
  let totalPearJuice := pearJuiceRate * blend.pearCount
  let totalOrangeJuice := orangeJuiceRate * blend.orangeCount
  let totalJuice := totalPearJuice + totalOrangeJuice
  totalOrangeJuice / totalJuice

theorem orange_juice_percentage
  (pearJuice : FruitJuice)
  (orangeJuice : FruitJuice)
  (blend : Blend)
  (h1 : pearJuice.fruitCount = 5 ∧ pearJuice.juiceAmount = 10)
  (h2 : orangeJuice.fruitCount = 4 ∧ orangeJuice.juiceAmount = 12)
  (h3 : blend.pearCount = 9 ∧ blend.orangeCount = 6) :
  calculateJuicePercentage pearJuice orangeJuice blend = 1/2 := by
  sorry

#eval calculateJuicePercentage ⟨5, 10⟩ ⟨4, 12⟩ ⟨9, 6⟩

end NUMINAMATH_CALUDE_orange_juice_percentage_l531_53114


namespace NUMINAMATH_CALUDE_units_digit_problem_l531_53180

theorem units_digit_problem : ∃ n : ℕ, n < 10 ∧ 
  (72^129 + 36^93 + 57^73 - 45^105) % 10 = n ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l531_53180


namespace NUMINAMATH_CALUDE_balance_problem_l531_53196

/-- The problem of balancing weights on a scale --/
theorem balance_problem :
  let total_weight : ℝ := 4.5 -- in kg
  let num_weights : ℕ := 9
  let weight_per_item : ℝ := total_weight / num_weights -- in kg
  let pencil_case_weight : ℝ := 0.85 -- in kg
  let dictionary_weight : ℝ := 1.05 -- in kg
  let num_weights_on_scale : ℕ := 2
  let num_dictionaries : ℕ := 5
  ∃ (num_pencil_cases : ℕ),
    (num_weights_on_scale * weight_per_item + num_pencil_cases * pencil_case_weight) =
    (num_dictionaries * dictionary_weight) ∧
    num_pencil_cases = 5 :=
by sorry

end NUMINAMATH_CALUDE_balance_problem_l531_53196


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_inverse_trig_functions_l531_53197

theorem min_value_of_sum_of_inverse_trig_functions
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (m : ℝ), ∀ (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2),
    m ≤ a / (Real.sin θ)^3 + b / (Real.cos θ)^3 ∧
    m = (a^(2/5) + b^(2/5))^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_inverse_trig_functions_l531_53197


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l531_53183

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l531_53183


namespace NUMINAMATH_CALUDE_license_plate_theorem_l531_53182

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of possible digits -/
def digit_count : ℕ := 10

/-- The number of license plate combinations -/
def license_plate_count : ℕ := consonant_count * consonant_count * vowel_count * vowel_count * digit_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l531_53182


namespace NUMINAMATH_CALUDE_complex_number_properties_l531_53163

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l531_53163


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l531_53134

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ 
  (∀ (M : ℕ), M ≤ 9 → 6 ∣ (5678 * 10 + M) → M ≤ N) ∧
  (6 ∣ (5678 * 10 + N)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l531_53134


namespace NUMINAMATH_CALUDE_remainder_8927_mod_11_l531_53186

theorem remainder_8927_mod_11 : 8927 % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8927_mod_11_l531_53186


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l531_53113

theorem sum_of_squares_of_roots (a b c : ℂ) : 
  (2 * a^3 - a^2 + 4*a + 10 = 0) → 
  (2 * b^3 - b^2 + 4*b + 10 = 0) → 
  (2 * c^3 - c^2 + 4*c + 10 = 0) → 
  a^2 + b^2 + c^2 = -15/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l531_53113


namespace NUMINAMATH_CALUDE_no_real_solutions_l531_53171

theorem no_real_solutions (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / (x + 4) ≠ m / (x + 4)) ↔ m = -5 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l531_53171


namespace NUMINAMATH_CALUDE_bus_driver_overtime_pay_increase_l531_53132

/-- Calculates the percentage increase in overtime pay rate for a bus driver -/
theorem bus_driver_overtime_pay_increase 
  (regular_rate : ℝ) 
  (regular_hours : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) : 
  regular_rate = 16 →
  regular_hours = 40 →
  total_compensation = 920 →
  total_hours = 50 →
  ((total_compensation - regular_rate * regular_hours) / (total_hours - regular_hours) - regular_rate) / regular_rate * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_overtime_pay_increase_l531_53132


namespace NUMINAMATH_CALUDE_salary_spending_percentage_l531_53185

theorem salary_spending_percentage 
  (total_salary : ℝ) 
  (a_salary : ℝ) 
  (b_spending_rate : ℝ) 
  (h1 : total_salary = 5000)
  (h2 : a_salary = 3750)
  (h3 : b_spending_rate = 0.85)
  (h4 : a_salary * (1 - a_spending_rate) = (total_salary - a_salary) * (1 - b_spending_rate)) :
  a_spending_rate = 0.95 := by
sorry

end NUMINAMATH_CALUDE_salary_spending_percentage_l531_53185


namespace NUMINAMATH_CALUDE_last_digit_square_periodicity_and_symmetry_l531_53141

theorem last_digit_square_periodicity_and_symmetry :
  ∀ (n : ℕ), 
    (n^2 % 10 = ((n + 10)^2) % 10) ∧
    (∀ (k : ℕ), k ≤ 4 → (k^2 % 10 = ((10 - k)^2) % 10)) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_square_periodicity_and_symmetry_l531_53141


namespace NUMINAMATH_CALUDE_susie_q_investment_l531_53189

def pretty_penny_rate : ℝ := 0.03
def five_and_dime_rate : ℝ := 0.05
def total_investment : ℝ := 1000
def total_after_two_years : ℝ := 1090.02
def years : ℕ := 2

theorem susie_q_investment (x : ℝ) :
  x * (1 + pretty_penny_rate) ^ years + (total_investment - x) * (1 + five_and_dime_rate) ^ years = total_after_two_years →
  x = 300 := by
sorry

end NUMINAMATH_CALUDE_susie_q_investment_l531_53189


namespace NUMINAMATH_CALUDE_dog_count_l531_53123

theorem dog_count (dogs people : ℕ) : 
  (4 * dogs + 2 * people = 2 * (dogs + people) + 28) → dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_dog_count_l531_53123


namespace NUMINAMATH_CALUDE_total_tiles_to_replace_l531_53161

/-- Represents the layout of paths in the park -/
structure ParkPaths where
  horizontalLengths : List Nat
  verticalLengths : List Nat

/-- Calculates the total number of tiles needed for replacement -/
def totalTiles (paths : ParkPaths) : Nat :=
  let horizontalSum := paths.horizontalLengths.sum
  let verticalSum := paths.verticalLengths.sum
  let intersections := 16  -- This value is derived from the problem description
  horizontalSum + verticalSum - intersections

/-- The main theorem stating the total number of tiles to be replaced -/
theorem total_tiles_to_replace :
  ∃ (paths : ParkPaths),
    paths.horizontalLengths = [30, 50, 30, 20, 20, 50] ∧
    paths.verticalLengths = [20, 50, 20, 50, 50] ∧
    totalTiles paths = 374 :=
  sorry

end NUMINAMATH_CALUDE_total_tiles_to_replace_l531_53161


namespace NUMINAMATH_CALUDE_students_in_both_math_and_science_l531_53177

theorem students_in_both_math_and_science 
  (total : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total = 40) 
  (h2 : not_math = 10) 
  (h3 : not_science = 15) 
  (h4 : not_either = 2) : 
  total - not_math + total - not_science - (total - not_either) = 17 := by
sorry

end NUMINAMATH_CALUDE_students_in_both_math_and_science_l531_53177


namespace NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l531_53168

/-- A right triangle with specific leg relationship and area -/
structure SpecialRightTriangle where
  shorter_leg : ℝ
  longer_leg : ℝ
  hypotenuse : ℝ
  leg_relationship : longer_leg = 3 * shorter_leg - 3
  area_condition : (1 / 2) * shorter_leg * longer_leg = 108
  right_triangle : shorter_leg ^ 2 + longer_leg ^ 2 = hypotenuse ^ 2

/-- The hypotenuse of the special right triangle is √657 -/
theorem special_right_triangle_hypotenuse (t : SpecialRightTriangle) : t.hypotenuse = Real.sqrt 657 := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_hypotenuse_l531_53168


namespace NUMINAMATH_CALUDE_radiator_antifreeze_percentage_l531_53191

/-- The capacity of the radiator in liters -/
def radiator_capacity : ℝ := 6

/-- The volume of liquid replaced with pure antifreeze in liters -/
def replaced_volume : ℝ := 1

/-- The final percentage of antifreeze in the mixture -/
def final_percentage : ℝ := 0.5

/-- The initial percentage of antifreeze in the radiator -/
def initial_percentage : ℝ := 0.4

theorem radiator_antifreeze_percentage :
  let remaining_volume := radiator_capacity - replaced_volume
  let initial_antifreeze := initial_percentage * radiator_capacity
  let remaining_antifreeze := initial_antifreeze - initial_percentage * replaced_volume
  let final_antifreeze := remaining_antifreeze + replaced_volume
  final_antifreeze = final_percentage * radiator_capacity :=
by sorry

end NUMINAMATH_CALUDE_radiator_antifreeze_percentage_l531_53191


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l531_53167

theorem square_area_from_diagonal (x : ℝ) (h : x > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s * s = x * x / 2 :=
by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l531_53167


namespace NUMINAMATH_CALUDE_time_puzzle_l531_53133

theorem time_puzzle : 
  ∃ h : ℝ, h = (12 - h) + (2/5) * h ∧ h = 7.5 := by sorry

end NUMINAMATH_CALUDE_time_puzzle_l531_53133


namespace NUMINAMATH_CALUDE_max_product_range_l531_53118

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_range (h k : ℝ → ℝ) 
  (h_range : ∀ x, -3 ≤ h x ∧ h x ≤ 5)
  (k_range : ∀ x, 0 ≤ k x ∧ k x ≤ 4) :
  ∃ d, ∀ x, h x ^ 2 * k x ≤ d ∧ d = 100 :=
sorry

end NUMINAMATH_CALUDE_max_product_range_l531_53118


namespace NUMINAMATH_CALUDE_turnips_sum_l531_53117

/-- The number of turnips Keith grew -/
def keith_turnips : ℕ := 6

/-- The number of turnips Alyssa grew -/
def alyssa_turnips : ℕ := 9

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ := keith_turnips + alyssa_turnips

theorem turnips_sum :
  total_turnips = 15 := by sorry

end NUMINAMATH_CALUDE_turnips_sum_l531_53117


namespace NUMINAMATH_CALUDE_p_true_q_false_l531_53127

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by sorry

end NUMINAMATH_CALUDE_p_true_q_false_l531_53127


namespace NUMINAMATH_CALUDE_inequality_proof_l531_53175

/-- Given f(x) = e^x - x^2, prove that for all x > 0, (e^x + (2-e)x - 1) / x ≥ ln x + 1 -/
theorem inequality_proof (x : ℝ) (hx : x > 0) : (Real.exp x + (2 - Real.exp 1) * x - 1) / x ≥ Real.log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l531_53175


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l531_53125

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560/83521 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l531_53125


namespace NUMINAMATH_CALUDE_tourist_speeds_l531_53122

theorem tourist_speeds (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (20 / x + 2.5 = 20 / y) ∧ (20 / (x - 2) = 20 / (1.5 * y)) → x = 8 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_tourist_speeds_l531_53122


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l531_53124

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l531_53124


namespace NUMINAMATH_CALUDE_a_range_theorem_l531_53138

-- Define the type for real numbers greater than zero
def PositiveReal := {x : ℝ // x > 0}

-- Define the monotonically increasing property for a^x
def MonotonicallyIncreasing (a : PositiveReal) : Prop :=
  ∀ x y : ℝ, x < y → (a.val : ℝ) ^ x < (a.val : ℝ) ^ y

-- Define the property that x^2 - ax + 1 > 0 does not hold for all x
def NotAlwaysPositive (a : PositiveReal) : Prop :=
  ¬(∀ x : ℝ, x^2 - (a.val : ℝ) * x + 1 > 0)

-- State the theorem
theorem a_range_theorem (a : PositiveReal) 
  (h1 : MonotonicallyIncreasing a) 
  (h2 : NotAlwaysPositive a) : 
  (a.val : ℝ) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l531_53138


namespace NUMINAMATH_CALUDE_coat_price_proof_l531_53120

theorem coat_price_proof (reduction : ℝ) (percentage : ℝ) (original_price : ℝ) : 
  reduction = 400 →
  percentage = 0.8 →
  percentage * original_price = reduction →
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_coat_price_proof_l531_53120


namespace NUMINAMATH_CALUDE_ana_beto_game_l531_53119

def is_valid_sequence (seq : List Int) : Prop :=
  seq.length = 2016 ∧ (seq.count 1 = 1008) ∧ (seq.count (-1) = 1008)

def block_sum_squares (blocks : List (List Int)) : Int :=
  (blocks.map (λ block => (block.sum)^2)).sum

theorem ana_beto_game (N : Nat) :
  (∃ (seq : List Int) (blocks : List (List Int)),
    is_valid_sequence seq ∧
    seq = blocks.join ∧
    block_sum_squares blocks = N) ↔
  (N % 2 = 0 ∧ N ≤ 2016) :=
sorry

end NUMINAMATH_CALUDE_ana_beto_game_l531_53119


namespace NUMINAMATH_CALUDE_solution_set_l531_53187

theorem solution_set (x : ℝ) : 
  (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1/4 ∧ x - 2 > 0 → x > 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l531_53187


namespace NUMINAMATH_CALUDE_positive_real_inequality_l531_53140

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l531_53140


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l531_53192

theorem fraction_equation_solution (x : ℚ) :
  (1 / (x + 2) + 2 / (x + 2) + x / (x + 2) + 3 / (x + 2) = 4) → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l531_53192


namespace NUMINAMATH_CALUDE_class_grade_average_l531_53110

theorem class_grade_average (n : ℕ) (h : n > 0) :
  let first_quarter := n / 4
  let remaining := n - first_quarter
  let first_quarter_avg := 92
  let remaining_avg := 76
  let total_sum := first_quarter * first_quarter_avg + remaining * remaining_avg
  (total_sum : ℚ) / n = 80 := by
sorry

end NUMINAMATH_CALUDE_class_grade_average_l531_53110


namespace NUMINAMATH_CALUDE_range_of_x_l531_53152

open Set

def S : Set ℝ := {x | x ∈ Icc 2 5 ∨ x < 1 ∨ x > 4}

theorem range_of_x (h : ¬ ∀ x, x ∈ S) : 
  {x : ℝ | x ∈ Ico 1 2} = {x : ℝ | ¬ (x ∈ S)} := by sorry

end NUMINAMATH_CALUDE_range_of_x_l531_53152


namespace NUMINAMATH_CALUDE_root_ratio_theorem_l531_53145

theorem root_ratio_theorem (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + k*x₁ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₂^2 + k*x₂ + 3/4*k^2 - 3*k + 9/2 = 0 →
  x₁ ≠ x₂ →
  x₁^2020 / x₂^2021 = -2/3 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_theorem_l531_53145


namespace NUMINAMATH_CALUDE_book_purchase_solution_l531_53188

/-- Represents the cost and purchase details of two types of books -/
structure BookPurchase where
  costA : ℕ  -- Cost of book A
  costB : ℕ  -- Cost of book B
  totalBooks : ℕ  -- Total number of books to purchase
  maxCost : ℕ  -- Maximum total cost

/-- Defines the conditions of the book purchase problem -/
def validBookPurchase (bp : BookPurchase) : Prop :=
  bp.costB = bp.costA + 20 ∧  -- Condition 1
  540 / bp.costA = 780 / bp.costB ∧  -- Condition 2
  bp.totalBooks = 70 ∧  -- Condition 3
  bp.maxCost = 3550  -- Condition 4

/-- Theorem stating the solution to the book purchase problem -/
theorem book_purchase_solution (bp : BookPurchase) 
  (h : validBookPurchase bp) : 
  bp.costA = 45 ∧ bp.costB = 65 ∧ 
  (∀ m : ℕ, m * bp.costA + (bp.totalBooks - m) * bp.costB ≤ bp.maxCost → m ≥ 50) :=
sorry

end NUMINAMATH_CALUDE_book_purchase_solution_l531_53188


namespace NUMINAMATH_CALUDE_whack_a_mole_tickets_whack_a_mole_tickets_proof_l531_53130

theorem whack_a_mole_tickets : ℕ → Prop :=
  fun whack_tickets =>
    let skee_tickets : ℕ := 9
    let candy_cost : ℕ := 6
    let candies_bought : ℕ := 7
    whack_tickets + skee_tickets = candy_cost * candies_bought →
    whack_tickets = 33

-- The proof is omitted
theorem whack_a_mole_tickets_proof : whack_a_mole_tickets 33 := by
  sorry

end NUMINAMATH_CALUDE_whack_a_mole_tickets_whack_a_mole_tickets_proof_l531_53130


namespace NUMINAMATH_CALUDE_farm_animal_leg_difference_l531_53142

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 6

theorem farm_animal_leg_difference 
  (num_chickens : ℕ) 
  (total_legs : ℕ) 
  (h1 : total_legs = cow_legs * num_cows + chicken_legs * num_chickens)
  (h2 : total_legs > 2 * (num_cows + num_chickens)) :
  total_legs - 2 * (num_cows + num_chickens) = 12 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_leg_difference_l531_53142


namespace NUMINAMATH_CALUDE_congruence_problem_l531_53126

theorem congruence_problem (x : ℤ) : 
  (4 * x + 5) ≡ 3 [ZMOD 17] → (2 * x + 8) ≡ 7 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l531_53126


namespace NUMINAMATH_CALUDE_unique_digit_subtraction_l531_53108

theorem unique_digit_subtraction :
  ∃! (I K S : ℕ),
    I < 10 ∧ K < 10 ∧ S < 10 ∧
    100 * K + 10 * I + S ≥ 100 ∧
    100 * S + 10 * I + K ≥ 100 ∧
    100 * S + 10 * K + I ≥ 100 ∧
    (100 * K + 10 * I + S) - (100 * S + 10 * I + K) = 100 * S + 10 * K + I :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_subtraction_l531_53108


namespace NUMINAMATH_CALUDE_power_plus_one_not_divisible_by_power_minus_one_l531_53155

theorem power_plus_one_not_divisible_by_power_minus_one (x y : ℕ) (h : y > 2) :
  ¬ (2^y - 1 ∣ 2^x + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_plus_one_not_divisible_by_power_minus_one_l531_53155


namespace NUMINAMATH_CALUDE_circledTimes_calculation_l531_53107

-- Define the ⊗ operation
def circledTimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem circledTimes_calculation :
  circledTimes (circledTimes 5 7) (circledTimes 4 2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_circledTimes_calculation_l531_53107


namespace NUMINAMATH_CALUDE_crow_worm_consumption_l531_53137

/-- Given that 3 crows eat 30 worms in one hour, prove that 5 crows will eat 100 worms in 2 hours. -/
theorem crow_worm_consumption (crows_per_hour : ℕ → ℕ → ℕ) : 
  crows_per_hour 3 30 = 1  -- 3 crows eat 30 worms in 1 hour
  → crows_per_hour 5 100 = 2  -- 5 crows eat 100 worms in 2 hours
:= by sorry

end NUMINAMATH_CALUDE_crow_worm_consumption_l531_53137


namespace NUMINAMATH_CALUDE_odd_digits_346_base5_l531_53174

/-- Counts the number of odd digits in a base-5 number --/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 --/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_346_base5 : 
  countOddDigitsBase5 (toBase5 346) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_346_base5_l531_53174


namespace NUMINAMATH_CALUDE_biscuit_banana_cost_ratio_l531_53193

-- Define variables
variable (b : ℚ) -- Cost of one biscuit
variable (x : ℚ) -- Cost of one banana

-- Define Susie's and Daisy's expenditures
def susie_expenditure : ℚ := 6 * b + 4 * x
def daisy_expenditure : ℚ := 4 * b + 20 * x

-- State the theorem
theorem biscuit_banana_cost_ratio :
  (susie_expenditure b x = daisy_expenditure b x / 3) →
  (b / x = 4 / 7) := by
  sorry

end NUMINAMATH_CALUDE_biscuit_banana_cost_ratio_l531_53193


namespace NUMINAMATH_CALUDE_min_red_chips_l531_53170

theorem min_red_chips (w b r : ℕ) : 
  b ≥ (3 * w) / 4 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  r ≥ 132 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ (3 * w') / 4 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 132 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l531_53170


namespace NUMINAMATH_CALUDE_last_card_is_diamond_six_l531_53158

/-- Represents a playing card --/
inductive Card
| Joker : Bool → Card  -- True for Big Joker, False for Little Joker
| Number : Nat → Suit → Card
| Face : Face → Suit → Card

/-- Represents the suit of a card --/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- Represents face cards --/
inductive Face
| Jack | Queen | King

/-- Represents a deck of cards --/
def Deck := List Card

/-- Creates a standard deck of 54 cards in the specified order --/
def standardDeck : Deck := sorry

/-- Combines two decks --/
def combinedDeck (d1 d2 : Deck) : Deck := sorry

/-- Applies the discard-and-place rule to a deck --/
def applyRule (d : Deck) : Card := sorry

/-- Theorem: The last remaining card is the Diamond 6 --/
theorem last_card_is_diamond_six :
  let d1 := standardDeck
  let d2 := standardDeck
  let combined := combinedDeck d1 d2
  applyRule combined = Card.Number 6 Suit.Diamonds := by sorry

end NUMINAMATH_CALUDE_last_card_is_diamond_six_l531_53158


namespace NUMINAMATH_CALUDE_same_terminal_side_l531_53103

/-- Proves that 375° has the same terminal side as α = π/12 + 2kπ, where k is an integer -/
theorem same_terminal_side (k : ℤ) : ∃ (n : ℤ), 375 * π / 180 = π / 12 + 2 * k * π + 2 * n * π := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l531_53103


namespace NUMINAMATH_CALUDE_bus_express_speed_l531_53144

/-- Proves that the speed of a bus in express mode is 48 km/h given specific conditions -/
theorem bus_express_speed (route_length : ℝ) (time_reduction : ℝ) (speed_increase : ℝ)
  (h1 : route_length = 16)
  (h2 : time_reduction = 1 / 15)
  (h3 : speed_increase = 8)
  : ∃ x : ℝ, x = 48 ∧ 
    route_length / (x - speed_increase) - route_length / x = time_reduction :=
by sorry

end NUMINAMATH_CALUDE_bus_express_speed_l531_53144


namespace NUMINAMATH_CALUDE_spade_operation_l531_53135

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation : (5 : ℝ) * (spade 2 (spade 6 9)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_l531_53135


namespace NUMINAMATH_CALUDE_circular_track_circumference_l531_53131

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time_minutes : ℝ) 
  (h1 : speed1 = 20) 
  (h2 : speed2 = 16) 
  (h3 : meeting_time_minutes = 36) : 
  speed1 + speed2 * meeting_time_minutes / 60 = 21.6 := by
  sorry

#check circular_track_circumference

end NUMINAMATH_CALUDE_circular_track_circumference_l531_53131


namespace NUMINAMATH_CALUDE_reflected_light_is_two_thirds_l531_53136

/-- A mirror that reflects half the light shined on it back and passes the other half onward -/
structure FiftyPercentMirror :=
  (reflect : ℝ → ℝ)
  (pass : ℝ → ℝ)
  (reflect_half : ∀ x, reflect x = x / 2)
  (pass_half : ∀ x, pass x = x / 2)

/-- Two parallel fifty percent mirrors -/
structure TwoParallelMirrors :=
  (mirror1 : FiftyPercentMirror)
  (mirror2 : FiftyPercentMirror)

/-- The fraction of light reflected back to the left by two parallel fifty percent mirrors -/
def reflected_light (mirrors : TwoParallelMirrors) (initial_light : ℝ) : ℝ :=
  sorry

/-- Theorem: The total fraction of light reflected back to the left by two parallel "fifty percent mirrors" is 2/3 when light is shined from the left -/
theorem reflected_light_is_two_thirds (mirrors : TwoParallelMirrors) (initial_light : ℝ) :
  reflected_light mirrors initial_light = 2/3 * initial_light :=
sorry

end NUMINAMATH_CALUDE_reflected_light_is_two_thirds_l531_53136


namespace NUMINAMATH_CALUDE_log_division_simplification_l531_53166

theorem log_division_simplification : 
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l531_53166


namespace NUMINAMATH_CALUDE_elder_age_l531_53148

/-- The age difference between two people -/
def age_difference : ℕ := 20

/-- The number of years ago when the elder was 5 times as old as the younger -/
def years_ago : ℕ := 8

/-- The ratio of elder's age to younger's age in the past -/
def age_ratio : ℕ := 5

theorem elder_age (younger_age elder_age : ℕ) : 
  (elder_age = younger_age + age_difference) → 
  (elder_age - years_ago = age_ratio * (younger_age - years_ago)) →
  elder_age = 33 := by
sorry

end NUMINAMATH_CALUDE_elder_age_l531_53148


namespace NUMINAMATH_CALUDE_sum_of_digits_square_of_nine_twos_l531_53111

/-- The sum of digits of the square of a number consisting of n twos -/
def sum_of_digits_square_of_twos (n : ℕ) : ℕ := 2 * n^2

/-- The number of twos in our specific case -/
def num_twos : ℕ := 9

/-- Theorem: The sum of the digits of the square of a number consisting of 9 twos is 162 -/
theorem sum_of_digits_square_of_nine_twos :
  sum_of_digits_square_of_twos num_twos = 162 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_of_nine_twos_l531_53111


namespace NUMINAMATH_CALUDE_uncovered_side_length_l531_53104

/-- A rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of the field is 20 feet long -/
theorem uncovered_side_length (field : FencedField) 
  (h_area : field.area = 120)
  (h_fencing : field.fencing = 32)
  (h_rectangle : field.area = field.length * field.width)
  (h_fence_sides : field.fencing = field.length + 2 * field.width) :
  field.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l531_53104


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l531_53129

/-- A regular polygon with perimeter 180 cm and side length 15 cm has 12 sides and interior angles of 150°. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (perimeter side_length : ℝ) (interior_angle : ℝ),
    perimeter = 180 →
    side_length = 15 →
    n * side_length = perimeter →
    interior_angle = (n - 2) * 180 / n →
    n = 12 ∧ interior_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l531_53129


namespace NUMINAMATH_CALUDE_stratified_sampling_group_D_l531_53106

/-- Represents the number of districts in each group -/
structure GroupSizes :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Calculates the total number of districts -/
def total_districts (g : GroupSizes) : ℕ := g.A + g.B + g.C + g.D

/-- Calculates the number of districts to be selected from a group in stratified sampling -/
def stratified_sample (group_size : ℕ) (total : ℕ) (sample_size : ℕ) : ℚ :=
  (group_size : ℚ) / (total : ℚ) * (sample_size : ℚ)

theorem stratified_sampling_group_D :
  let groups : GroupSizes := ⟨4, 10, 16, 8⟩
  let total := total_districts groups
  let sample_size := 9
  stratified_sample groups.D total sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_D_l531_53106


namespace NUMINAMATH_CALUDE_bus_train_speed_ratio_l531_53149

/-- The fraction of the speed of a bus compared to the speed of a train -/
theorem bus_train_speed_ratio :
  -- The ratio between the speed of a train and a car
  ∀ (train_speed car_speed : ℝ),
  train_speed / car_speed = 16 / 15 →
  -- A bus covered 320 km in 5 hours
  ∀ (bus_speed : ℝ),
  bus_speed * 5 = 320 →
  -- The car will cover 525 km in 7 hours
  car_speed * 7 = 525 →
  -- The fraction of the speed of the bus compared to the speed of the train
  bus_speed / train_speed = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_bus_train_speed_ratio_l531_53149


namespace NUMINAMATH_CALUDE_grid_shading_l531_53115

/-- Given a 4 × 5 grid with 3 squares already shaded, 
    prove that 7 additional squares need to be shaded 
    to have half of all squares shaded. -/
theorem grid_shading (grid_width : Nat) (grid_height : Nat) 
  (total_squares : Nat) (already_shaded : Nat) (half_squares : Nat) 
  (additional_squares : Nat) : 
  grid_width = 4 → 
  grid_height = 5 → 
  total_squares = grid_width * grid_height →
  already_shaded = 3 →
  half_squares = total_squares / 2 →
  additional_squares = half_squares - already_shaded →
  additional_squares = 7 := by
sorry


end NUMINAMATH_CALUDE_grid_shading_l531_53115
