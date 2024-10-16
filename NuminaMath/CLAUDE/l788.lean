import Mathlib

namespace NUMINAMATH_CALUDE_oil_price_reduction_l788_78849

/-- Given a 25% reduction in oil price, prove the reduced price per kg is 30 Rs. --/
theorem oil_price_reduction (original_price : ℝ) (h1 : original_price > 0) : 
  let reduced_price := 0.75 * original_price
  (600 / reduced_price) = (600 / original_price) + 5 →
  reduced_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l788_78849


namespace NUMINAMATH_CALUDE_no_nontrivial_solution_x2_plus_y2_eq_3z2_l788_78823

theorem no_nontrivial_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_solution_x2_plus_y2_eq_3z2_l788_78823


namespace NUMINAMATH_CALUDE_angle_expression_value_l788_78806

theorem angle_expression_value (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin α = Real.sqrt 15 / 4) :  -- sin α = √15/4
  Real.sin (α + π/4) / (Real.sin (2*α) + Real.cos (2*α) + 1) = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l788_78806


namespace NUMINAMATH_CALUDE_odd_function_implies_a_value_f_is_increasing_f_range_l788_78861

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x / (3^x + 1) - a

theorem odd_function_implies_a_value (a : ℝ) :
  (∀ x, f x a = -f (-x) a) → a = 1/2 := by sorry

theorem f_is_increasing (a : ℝ) (h : a = 1/2) :
  Monotone (f · a) := by sorry

theorem f_range (a : ℝ) (h : a = 1/2) :
  Set.range (f · a) = Set.Ioo (-1/2 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_a_value_f_is_increasing_f_range_l788_78861


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l788_78836

theorem chess_tournament_participants (n : ℕ) : 
  n > 3 → 
  (n * (n - 1)) / 2 = 26 → 
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l788_78836


namespace NUMINAMATH_CALUDE_factorization_of_3m_squared_minus_12_l788_78816

theorem factorization_of_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_3m_squared_minus_12_l788_78816


namespace NUMINAMATH_CALUDE_weight_of_person_a_l788_78809

/-- Given the average weights of different groups and the relationship between individuals' weights,
    prove that the weight of person A is 80 kg. -/
theorem weight_of_person_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 8 →
  (b + c + d + e) / 4 = 79 →
  a = 80 := by
sorry

end NUMINAMATH_CALUDE_weight_of_person_a_l788_78809


namespace NUMINAMATH_CALUDE_kenny_must_do_at_least_three_on_thursday_l788_78818

/-- Represents the number of jumping jacks done on each day of the week -/
structure WeeklyJumpingJacks where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Calculates the total number of jumping jacks for a week -/
def weekTotal (w : WeeklyJumpingJacks) : ℕ :=
  w.sunday + w.monday + w.tuesday + w.wednesday + w.thursday + w.friday + w.saturday

theorem kenny_must_do_at_least_three_on_thursday 
  (lastWeek : ℕ) 
  (thisWeek : WeeklyJumpingJacks) 
  (someDay : ℕ) :
  lastWeek = 324 →
  thisWeek.sunday = 34 →
  thisWeek.monday = 20 →
  thisWeek.tuesday = 0 →
  thisWeek.wednesday = 123 →
  thisWeek.saturday = 61 →
  (thisWeek.thursday = someDay ∨ thisWeek.friday = someDay) →
  someDay = 23 →
  weekTotal thisWeek > lastWeek →
  thisWeek.thursday ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_kenny_must_do_at_least_three_on_thursday_l788_78818


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_of_three_l788_78815

theorem largest_of_three_consecutive_multiples_of_three (a b c : ℕ) : 
  (∃ n : ℕ, a = 3 * n ∧ b = 3 * (n + 1) ∧ c = 3 * (n + 2)) → 
  a + b + c = 72 → 
  max a (max b c) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_multiples_of_three_l788_78815


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l788_78827

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Definition of the first line -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0

/-- Definition of the second line -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∃ a : ℝ, (∀ x y : ℝ, line1 a x y ↔ line2 a x y) ∧ a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l788_78827


namespace NUMINAMATH_CALUDE_prob_last_is_one_l788_78839

/-- Represents the set of possible digits Andrea can write. -/
def Digits : Finset ℕ := {1, 2, 3, 4}

/-- Represents whether a number is prime. -/
def isPrime (n : ℕ) : Prop := sorry

/-- Represents the process of writing digits until the sum of the last two is prime. -/
def StoppingProcess : Type := sorry

/-- The probability of the last digit being 1 given the first digit. -/
def probLastIsOne (first : ℕ) : ℚ := sorry

/-- The probability of the last digit being 1 for the entire process. -/
def totalProbLastIsOne : ℚ := sorry

/-- Theorem stating the probability of the last digit being 1 is 17/44. -/
theorem prob_last_is_one :
  totalProbLastIsOne = 17 / 44 := by sorry

end NUMINAMATH_CALUDE_prob_last_is_one_l788_78839


namespace NUMINAMATH_CALUDE_warehouse_analysis_l788_78899

/-- Represents the daily changes in goods, where positive values indicate goods entering
    and negative values indicate goods leaving the warehouse -/
def daily_changes : List Int := [31, -31, -16, 34, -38, -20]

/-- The final amount of goods in the warehouse after 6 days -/
def final_amount : Int := 430

/-- The fee for loading or unloading one ton of goods -/
def fee_per_ton : Int := 5

theorem warehouse_analysis :
  let net_change := daily_changes.sum
  let initial_amount := final_amount - net_change
  let total_fees := (daily_changes.map abs).sum * fee_per_ton
  (net_change < 0) ∧
  (initial_amount = 470) ∧
  (total_fees = 850) := by sorry

end NUMINAMATH_CALUDE_warehouse_analysis_l788_78899


namespace NUMINAMATH_CALUDE_long_video_multiple_is_42_l788_78808

/-- Represents the video release schedule and durations for John's channel --/
structure VideoSchedule where
  short_videos_per_day : Nat
  long_videos_per_day : Nat
  short_video_duration : Nat
  days_per_week : Nat
  total_weekly_duration : Nat

/-- Calculates how many times longer the long video is compared to a short video --/
def long_video_multiple (schedule : VideoSchedule) : Nat :=
  let total_short_duration := schedule.short_videos_per_day * schedule.short_video_duration * schedule.days_per_week
  let long_video_duration := schedule.total_weekly_duration - total_short_duration
  long_video_duration / (schedule.long_videos_per_day * schedule.days_per_week * schedule.short_video_duration)

theorem long_video_multiple_is_42 (schedule : VideoSchedule) 
  (h1 : schedule.short_videos_per_day = 2)
  (h2 : schedule.long_videos_per_day = 1)
  (h3 : schedule.short_video_duration = 2)
  (h4 : schedule.days_per_week = 7)
  (h5 : schedule.total_weekly_duration = 112) :
  long_video_multiple schedule = 42 := by
  sorry

#eval long_video_multiple {
  short_videos_per_day := 2,
  long_videos_per_day := 1,
  short_video_duration := 2,
  days_per_week := 7,
  total_weekly_duration := 112
}

end NUMINAMATH_CALUDE_long_video_multiple_is_42_l788_78808


namespace NUMINAMATH_CALUDE_descending_order_l788_78855

-- Define the numbers
def a : ℝ := 0.8
def b : ℝ := 0.878
def c : ℝ := 0.877
def d : ℝ := 0.87

-- Theorem statement
theorem descending_order : b > c ∧ c > d ∧ d > a := by sorry

end NUMINAMATH_CALUDE_descending_order_l788_78855


namespace NUMINAMATH_CALUDE_remaining_days_temperature_l788_78864

/-- Calculates the total temperature of the remaining days in a week given specific temperature conditions. -/
theorem remaining_days_temperature
  (avg_temp : ℝ)
  (days_in_week : ℕ)
  (first_three_temp : ℝ)
  (thursday_friday_temp : ℝ)
  (h1 : avg_temp = 60)
  (h2 : days_in_week = 7)
  (h3 : first_three_temp = 40)
  (h4 : thursday_friday_temp = 80) :
  (days_in_week : ℝ) * avg_temp - (3 * first_three_temp + 2 * thursday_friday_temp) = 140 := by
  sorry

#check remaining_days_temperature

end NUMINAMATH_CALUDE_remaining_days_temperature_l788_78864


namespace NUMINAMATH_CALUDE_tangent_perpendicular_l788_78869

-- Define the curve f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 4 = 0

-- Theorem statement
theorem tangent_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent at (x₀, y₀) is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → (y - y₀) = -(1/4) * (x - x₀)) ∧
    -- The tangent line equation
    tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_l788_78869


namespace NUMINAMATH_CALUDE_different_color_probability_l788_78822

/-- The set of colors for shorts -/
def shorts_colors : Finset String := {"black", "gold", "silver"}

/-- The set of colors for jerseys -/
def jersey_colors : Finset String := {"black", "white", "gold"}

/-- The probability of selecting different colors for shorts and jerseys -/
theorem different_color_probability : 
  (shorts_colors.card * jersey_colors.card - (shorts_colors ∩ jersey_colors).card) / 
  (shorts_colors.card * jersey_colors.card : ℚ) = 7/9 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l788_78822


namespace NUMINAMATH_CALUDE_scaled_standard_deviation_l788_78889

def data := List ℝ

def variance (d : data) : ℝ := sorry

def standardDeviation (d : data) : ℝ := sorry

def scaleData (d : data) (k : ℝ) : data := sorry

theorem scaled_standard_deviation 
  (d : data) 
  (h : variance d = 2) : 
  standardDeviation (scaleData d 2) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_scaled_standard_deviation_l788_78889


namespace NUMINAMATH_CALUDE_acid_solution_mixing_l788_78819

theorem acid_solution_mixing (y z : ℝ) (hy : y > 25) :
  (y * y / 100 + z * 40 / 100) / (y + z) * 100 = y + 10 →
  z = 10 * y / (y - 30) := by
sorry

end NUMINAMATH_CALUDE_acid_solution_mixing_l788_78819


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l788_78871

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 + 3*x - 20 = 7*x + 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 3*x₁ - 20 = 7*x₁ + 8) ∧ 
                (x₂^2 + 3*x₂ - 20 = 7*x₂ + 8) ∧ 
                (x₁ + x₂ = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l788_78871


namespace NUMINAMATH_CALUDE_system_solution_exists_l788_78888

theorem system_solution_exists (a b : ℤ) (h1 : 5 * a ≥ 7 * b) (h2 : 7 * b ≥ 0) :
  ∃ (x y z u : ℕ), x + 2 * y + 3 * z + 7 * u = a ∧ y + 2 * z + 5 * u = b := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l788_78888


namespace NUMINAMATH_CALUDE_zero_count_in_circular_sequence_l788_78803

/-- Represents a circular sequence without repetitions -/
structure CircularSequence (α : Type) where
  elements : List α
  no_repetitions : elements.Nodup
  circular : elements ≠ []

/-- Counts the number of occurrences of an element in a list -/
def count (α : Type) [DecidableEq α] (l : List α) (x : α) : Nat :=
  l.filter (· = x) |>.length

/-- Theorem: The number of zeroes in a circular sequence without repetitions is 0, 1, 2, or 4 -/
theorem zero_count_in_circular_sequence (m : ℕ) (seq : CircularSequence ℕ) :
  let zero_count := count ℕ seq.elements 0
  zero_count = 0 ∨ zero_count = 1 ∨ zero_count = 2 ∨ zero_count = 4 :=
sorry

end NUMINAMATH_CALUDE_zero_count_in_circular_sequence_l788_78803


namespace NUMINAMATH_CALUDE_quadratic_roots_average_l788_78898

theorem quadratic_roots_average (d : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0) :
  (∃ x y : ℝ, x ≠ y ∧ 3 * x^2 - 9 * x + d = 0 ∧ 3 * y^2 - 9 * y + d = 0 ∧ (x + y) / 2 = 1.5) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_average_l788_78898


namespace NUMINAMATH_CALUDE_fraction_simplification_l788_78817

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : 
  (2 / (1 - x)) - ((2 * x) / (1 - x)) = 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l788_78817


namespace NUMINAMATH_CALUDE_number_fraction_problem_l788_78820

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l788_78820


namespace NUMINAMATH_CALUDE_rabbit_fraction_l788_78896

theorem rabbit_fraction (initial_cage : ℕ) (added : ℕ) (park : ℕ) : 
  initial_cage = 13 → added = 7 → park = 60 → 
  (initial_cage + added : ℚ) / park = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_fraction_l788_78896


namespace NUMINAMATH_CALUDE_system_has_solution_our_system_has_solution_l788_78841

/-- A system of two linear equations in two variables -/
structure LinearSystem (α : Type*) [Ring α] where
  eq1 : α → α → α
  eq2 : α → α → α

/-- A solution to a system of linear equations -/
structure Solution (α : Type*) where
  x : α
  y : α

/-- Theorem stating that the given system has the specified solution -/
theorem system_has_solution 
  (R : Type*) [Ring R] 
  (system : LinearSystem R) 
  (sol : Solution R) : Prop :=
  system.eq1 sol.x sol.y = 1 ∧ 
  system.eq2 sol.x sol.y = 3 ∧
  sol.x = 2 ∧
  sol.y = -1

/-- The specific system of equations -/
def our_system : LinearSystem ℤ := {
  eq1 := λ x y => x + y
  eq2 := λ x y => x - y
}

/-- The specific solution -/
def our_solution : Solution ℤ := {
  x := 2
  y := -1
}

/-- Theorem stating that our specific system has our specific solution -/
theorem our_system_has_solution : 
  system_has_solution ℤ our_system our_solution := by
  sorry


end NUMINAMATH_CALUDE_system_has_solution_our_system_has_solution_l788_78841


namespace NUMINAMATH_CALUDE_square_area_probability_l788_78826

/-- The probability of a randomly chosen point on a line segment of length 12
    forming a square with an area between 36 and 81 -/
theorem square_area_probability : ∀ (AB : ℝ) (lower upper : ℝ),
  AB = 12 →
  lower = 6 →
  upper = 9 →
  (upper - lower) / AB = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_square_area_probability_l788_78826


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l788_78830

def fibonacci_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120
  | 6 => 720
  | 7 => 5040
  | 8 => 40320
  | 9 => 362880
  | _ => 0  -- For n ≥ 10, we only care about the last two digits, which are 00

def modified_fibonacci_series : List ℕ :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (modified_fibonacci_series.map (λ x => last_two_digits (fibonacci_factorial x))).sum % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l788_78830


namespace NUMINAMATH_CALUDE_painting_cost_conversion_l788_78821

/-- Given exchange rates and the cost of a painting in Namibian dollars, 
    prove its cost in Euros -/
theorem painting_cost_conversion 
  (usd_to_nam : ℝ) 
  (usd_to_eur : ℝ) 
  (painting_cost_nam : ℝ) 
  (h1 : usd_to_nam = 7) 
  (h2 : usd_to_eur = 0.9) 
  (h3 : painting_cost_nam = 140) : 
  painting_cost_nam / usd_to_nam * usd_to_eur = 18 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_conversion_l788_78821


namespace NUMINAMATH_CALUDE_range_of_m_l788_78891

def f (x : ℝ) := -x^2 + 4*x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc m 4, f x ∈ Set.Icc 0 4) ∧
  (∀ y ∈ Set.Icc 0 4, ∃ x ∈ Set.Icc m 4, f x = y) →
  m ∈ Set.Icc 0 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l788_78891


namespace NUMINAMATH_CALUDE_at_least_one_zero_l788_78812

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_zero_l788_78812


namespace NUMINAMATH_CALUDE_sons_age_l788_78897

/-- Proves that given the conditions, the son's present age is 25 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l788_78897


namespace NUMINAMATH_CALUDE_chebyshev_birth_year_l788_78813

def is_valid_year (year : Nat) : Prop :=
  -- Year is in the 19th century
  1800 ≤ year ∧ year < 1900 ∧
  -- Sum of hundreds and thousands digits is 3 times sum of units and tens digits
  (year / 100 + (year / 1000) % 10) = 3 * ((year % 10) + (year / 10) % 10) ∧
  -- Tens digit is greater than units digit
  (year / 10) % 10 > year % 10 ∧
  -- Chebyshev lived for 73 years and died in the same century
  year + 73 < 1900

theorem chebyshev_birth_year :
  ∀ year : Nat, is_valid_year year ↔ year = 1821 := by sorry

end NUMINAMATH_CALUDE_chebyshev_birth_year_l788_78813


namespace NUMINAMATH_CALUDE_modulo_congruence_l788_78874

theorem modulo_congruence : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -4792 - 242 [ZMOD 8] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_l788_78874


namespace NUMINAMATH_CALUDE_quadratic_extrema_l788_78843

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 3)^2 - 1

-- Define the interval
def I : Set ℝ := Set.Icc 1 4

-- Theorem statement
theorem quadratic_extrema :
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≥ f x) ∧
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ -1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 3) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = -1) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l788_78843


namespace NUMINAMATH_CALUDE_square_of_integer_l788_78814

theorem square_of_integer (x y z : ℤ) (A : ℤ) 
  (h1 : A = x * y + y * z + z * x)
  (h2 : 4 * x + y + z = 0) : 
  ∃ (k : ℤ), (-1) * A = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l788_78814


namespace NUMINAMATH_CALUDE_equation_solution_l788_78805

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ∧ x = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l788_78805


namespace NUMINAMATH_CALUDE_students_not_both_count_l788_78844

/-- Given information about students taking chemistry and physics classes -/
structure ClassData where
  both : ℕ         -- Number of students taking both chemistry and physics
  chemistry : ℕ    -- Total number of students taking chemistry
  only_physics : ℕ -- Number of students taking only physics

/-- Calculate the number of students taking chemistry or physics but not both -/
def students_not_both (data : ClassData) : ℕ :=
  (data.chemistry - data.both) + data.only_physics

/-- Theorem stating the number of students taking chemistry or physics but not both -/
theorem students_not_both_count (data : ClassData) 
  (h1 : data.both = 12)
  (h2 : data.chemistry = 30)
  (h3 : data.only_physics = 18) :
  students_not_both data = 36 := by
  sorry

#eval students_not_both ⟨12, 30, 18⟩

end NUMINAMATH_CALUDE_students_not_both_count_l788_78844


namespace NUMINAMATH_CALUDE_sine_transformation_l788_78856

theorem sine_transformation (x : ℝ) : 
  Real.sin (2 * (x + π/4) + π/6) = Real.sin (2*x + 2*π/3) := by
  sorry

end NUMINAMATH_CALUDE_sine_transformation_l788_78856


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_value_l788_78862

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushao (f : ℝ → ℝ) (x : ℝ) : ℕ → ℝ
  | 0 => x + 3
  | 1 => qinJiushao f x 0 * x - 1
  | 2 => qinJiushao f x 1 * x
  | 3 => qinJiushao f x 2 * x + 2
  | 4 => qinJiushao f x 3 * x - 1
  | _ => 0

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem qin_jiushao_v3_value :
  qinJiushao f 2 2 = 18 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_value_l788_78862


namespace NUMINAMATH_CALUDE_factorization_of_5a_cubed_minus_125a_l788_78828

theorem factorization_of_5a_cubed_minus_125a (a : ℝ) :
  5 * a^3 - 125 * a = 5 * a * (a + 5) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5a_cubed_minus_125a_l788_78828


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_8000_l788_78829

theorem greatest_multiple_of_four_under_cube_root_8000 :
  (∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) →
  (∃ (x : ℕ), x = 16 ∧ x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_8000_l788_78829


namespace NUMINAMATH_CALUDE_baker_new_cakes_l788_78866

theorem baker_new_cakes (initial_cakes sold_cakes current_cakes : ℕ) 
  (h1 : initial_cakes = 121)
  (h2 : sold_cakes = 105)
  (h3 : current_cakes = 186) :
  current_cakes - (initial_cakes - sold_cakes) = 170 := by
  sorry

end NUMINAMATH_CALUDE_baker_new_cakes_l788_78866


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l788_78865

/-- Given vectors a and b in R^2, prove that their difference has magnitude 1 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.sin (15 * π / 180) ∧
  b.2 = Real.cos (15 * π / 180) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 := by
  sorry

#check vector_difference_magnitude

end NUMINAMATH_CALUDE_vector_difference_magnitude_l788_78865


namespace NUMINAMATH_CALUDE_decimal_point_problem_l788_78842

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 100000 * x = 5 * (1 / x)) : 
  x = Real.sqrt 2 / 200 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l788_78842


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_range_l788_78879

theorem sin_cos_equation_solution_range :
  let f : ℝ → ℝ → ℝ := λ x a => Real.sin x ^ 2 + 2 * Real.cos x + a
  ∀ a : ℝ, (∃ x : ℝ, f x a = 0) ↔ a ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_range_l788_78879


namespace NUMINAMATH_CALUDE_min_non_acute_angles_l788_78838

/-- A convex polygon with 1992 sides -/
structure ConvexPolygon1992 where
  sides : ℕ
  convex : Bool
  sides_eq : sides = 1992
  is_convex : convex = true

/-- The number of interior angles that are not acute in a polygon -/
def non_acute_angles (p : ConvexPolygon1992) : ℕ := sorry

/-- The theorem stating the minimum number of non-acute angles in a ConvexPolygon1992 -/
theorem min_non_acute_angles (p : ConvexPolygon1992) : 
  non_acute_angles p ≥ 1989 := by sorry

end NUMINAMATH_CALUDE_min_non_acute_angles_l788_78838


namespace NUMINAMATH_CALUDE_range_of_a_for_two_real_roots_l788_78852

theorem range_of_a_for_two_real_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2 * Real.sqrt a * x + 2 * a - 1 = 0 ∧ 
               y^2 - 2 * Real.sqrt a * y + 2 * a - 1 = 0) → 
  0 ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_real_roots_l788_78852


namespace NUMINAMATH_CALUDE_combine_like_terms_l788_78886

theorem combine_like_terms (a b : ℝ) :
  4 * (a - b)^2 - 6 * (a - b)^2 + 8 * (a - b)^2 = 6 * (a - b)^2 := by sorry

end NUMINAMATH_CALUDE_combine_like_terms_l788_78886


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l788_78857

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ  -- Hours worked on long days
  hours_short_day : ℕ -- Hours worked on short days
  long_days : ℕ       -- Number of long workdays per week
  short_days : ℕ      -- Number of short workdays per week
  weekly_earnings : ℕ -- Weekly earnings in dollars

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_long_day * schedule.long_days + 
                     schedule.hours_short_day * schedule.short_days
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  hours_short_day := 6,
  long_days := 3,
  short_days := 2,
  weekly_earnings := 468
}

/-- Theorem stating that Sheila's hourly wage is $13 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 13 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l788_78857


namespace NUMINAMATH_CALUDE_january_oil_bill_l788_78881

theorem january_oil_bill (feb_bill jan_bill : ℚ) : 
  (feb_bill / jan_bill = 3 / 2) → 
  ((feb_bill + 10) / jan_bill = 5 / 3) → 
  jan_bill = 60 := by
sorry

end NUMINAMATH_CALUDE_january_oil_bill_l788_78881


namespace NUMINAMATH_CALUDE_party_handshakes_l788_78832

-- Define the number of couples
def num_couples : ℕ := 13

-- Define the total number of people
def total_people : ℕ := 2 * num_couples

-- Define the number of handshakes between men
def men_handshakes : ℕ := num_couples.choose 2

-- Define the number of handshakes between men and women (excluding spouses)
def men_women_handshakes : ℕ := num_couples * (num_couples - 1)

-- Theorem statement
theorem party_handshakes :
  men_handshakes + men_women_handshakes = 234 :=
by sorry

end NUMINAMATH_CALUDE_party_handshakes_l788_78832


namespace NUMINAMATH_CALUDE_some_number_value_l788_78860

theorem some_number_value (x : ℝ) : (50 + x / 90) * 90 = 4520 → x = 4470 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l788_78860


namespace NUMINAMATH_CALUDE_sam_and_billy_total_money_l788_78892

/-- Given that Sam has $75 and Billy has $25 less than twice Sam's money, 
    prove that their total money is $200. -/
theorem sam_and_billy_total_money :
  ∀ (sam_money billy_money : ℕ),
    sam_money = 75 →
    billy_money = 2 * sam_money - 25 →
    sam_money + billy_money = 200 := by
sorry

end NUMINAMATH_CALUDE_sam_and_billy_total_money_l788_78892


namespace NUMINAMATH_CALUDE_library_visitors_average_l788_78853

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 30 - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

/-- Theorem stating the average number of visitors per day in the given scenario -/
theorem library_visitors_average :
  averageVisitors 630 240 = 292 := by
  sorry

#eval averageVisitors 630 240

end NUMINAMATH_CALUDE_library_visitors_average_l788_78853


namespace NUMINAMATH_CALUDE_combination_20_choose_6_l788_78824

theorem combination_20_choose_6 : Nat.choose 20 6 = 19380 := by
  sorry

end NUMINAMATH_CALUDE_combination_20_choose_6_l788_78824


namespace NUMINAMATH_CALUDE_emus_per_pen_l788_78863

/-- Proves that the number of emus in each pen is 6 -/
theorem emus_per_pen (num_pens : ℕ) (eggs_per_week : ℕ) (h1 : num_pens = 4) (h2 : eggs_per_week = 84) : 
  (eggs_per_week / 7 * 2) / num_pens = 6 := by
  sorry

#check emus_per_pen

end NUMINAMATH_CALUDE_emus_per_pen_l788_78863


namespace NUMINAMATH_CALUDE_rhombus_converse_and_inverse_false_l788_78845

-- Define what it means for a polygon to be a rhombus
def is_rhombus (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to have all sides of equal length
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define a polygon (we don't need to specify its properties here)
def Polygon : Type := sorry

theorem rhombus_converse_and_inverse_false :
  (∃ p : Polygon, has_equal_sides p ∧ ¬is_rhombus p) ∧
  (∃ p : Polygon, ¬is_rhombus p ∧ has_equal_sides p) :=
sorry

end NUMINAMATH_CALUDE_rhombus_converse_and_inverse_false_l788_78845


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l788_78840

/-- A natural number is prime if it's greater than 1 and its only positive divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The roots of the quadratic equation x^2 - 73x + k = 0 -/
def roots (k : ℕ) : Set ℝ :=
  {x : ℝ | x^2 - 73*x + k = 0}

/-- The statement that both roots of x^2 - 73x + k = 0 are prime numbers -/
def both_roots_prime (k : ℕ) : Prop :=
  ∀ x ∈ roots k, ∃ n : ℕ, (x : ℝ) = n ∧ is_prime n

/-- There is exactly one value of k such that both roots of x^2 - 73x + k = 0 are prime numbers -/
theorem unique_k_for_prime_roots : ∃! k : ℕ, both_roots_prime k :=
  sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l788_78840


namespace NUMINAMATH_CALUDE_prob_at_least_one_three_l788_78873

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := sides * sides

/-- The number of outcomes where neither die shows a 3 -/
def neither_three : ℕ := (sides - 1) * (sides - 1)

/-- The number of outcomes where at least one die shows a 3 -/
def at_least_one_three : ℕ := total_outcomes - neither_three

/-- The probability of getting at least one 3 when rolling two 8-sided dice -/
theorem prob_at_least_one_three : 
  (at_least_one_three : ℚ) / total_outcomes = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_three_l788_78873


namespace NUMINAMATH_CALUDE_equation_represents_point_l788_78859

/-- The equation represents a point in the xy-plane -/
theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2*a*x + 2*b*y + a^2 + b^2 = 0 ↔ (x = -a ∧ y = -b) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l788_78859


namespace NUMINAMATH_CALUDE_not_perfect_square_l788_78880

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n^2)) :
  ¬ ∃ (x : ℕ), (n : ℕ)^2 + (d : ℕ) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l788_78880


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l788_78868

/-- 
Given two lines represented by the equations 2x - 3y + 6 = 0 and bx - 3y - 4 = 0,
if these lines are perpendicular, then b = -9/2.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 2*x - 3*y + 6 = 0 → bx - 3*y - 4 = 0 → 
    (2 : ℝ)/3 * (b/3) = -1) → 
  b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l788_78868


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l788_78801

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Define the set S of functions f: N → R satisfying the given conditions
def S : Set (ℕ → ℝ) := {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

-- State the theorem
theorem smallest_upper_bound :
  ∃ M : ℕ, (∀ f ∈ S, ∀ n : ℕ, f n < M) ∧
  (∀ M' : ℕ, M' < M → ∃ f ∈ S, ∃ n : ℕ, f n ≥ M') :=
sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l788_78801


namespace NUMINAMATH_CALUDE_income_change_approx_23_86_percent_l788_78877

def job_a_initial_weekly : ℚ := 60
def job_a_final_weekly : ℚ := 78
def job_a_quarterly_bonus : ℚ := 50

def job_b_initial_weekly : ℚ := 100
def job_b_final_weekly : ℚ := 115
def job_b_initial_biannual_bonus : ℚ := 200
def job_b_bonus_increase_rate : ℚ := 0.1

def weekly_expenses : ℚ := 30
def weeks_per_quarter : ℕ := 13

def initial_quarterly_income : ℚ :=
  job_a_initial_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_initial_weekly * weeks_per_quarter + job_b_initial_biannual_bonus / 2

def final_quarterly_income : ℚ :=
  job_a_final_weekly * weeks_per_quarter + job_a_quarterly_bonus +
  job_b_final_weekly * weeks_per_quarter + 
  (job_b_initial_biannual_bonus * (1 + job_b_bonus_increase_rate)) / 2

def quarterly_expenses : ℚ := weekly_expenses * weeks_per_quarter

def initial_effective_income : ℚ := initial_quarterly_income - quarterly_expenses
def final_effective_income : ℚ := final_quarterly_income - quarterly_expenses

def income_change_percentage : ℚ :=
  (final_effective_income - initial_effective_income) / initial_effective_income * 100

theorem income_change_approx_23_86_percent : 
  ∃ ε > 0, abs (income_change_percentage - 23.86) < ε :=
sorry

end NUMINAMATH_CALUDE_income_change_approx_23_86_percent_l788_78877


namespace NUMINAMATH_CALUDE_circles_tangent_sum_l788_78878

-- Define the circles and line
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 3)^2 = 1
def circle_C2 (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the external tangency condition
def externally_tangent (a b : ℝ) : Prop := (a - 1)^2 + (b + 3)^2 > 4

-- Define the equal tangent length condition
def equal_tangent_length (a b : ℝ) : Prop := 
  ∃ m : ℝ, (4 + 2*a + 2*b)*m + 5 - a^2 - (1 + b)^2 = 0

-- State the theorem
theorem circles_tangent_sum (a b : ℝ) :
  externally_tangent a b →
  equal_tangent_length a b →
  a + b = -2 := by sorry

end NUMINAMATH_CALUDE_circles_tangent_sum_l788_78878


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l788_78800

theorem y_in_terms_of_x (x y : ℚ) : x - 2 = 4 * (y - 1) + 3 → y = (1/4) * x - (1/4) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l788_78800


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l788_78890

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := λ x => -x^2

theorem parabola_shift_theorem :
  vertical_shift original_parabola 2 = λ x => -x^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l788_78890


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l788_78884

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 35 / 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l788_78884


namespace NUMINAMATH_CALUDE_system_solutions_l788_78833

def is_solution (x y : ℤ) : Prop :=
  4 * x^2 = y^2 + 2*y + 1 + 3 ∧
  (2*x)^2 - (y + 1)^2 = 3 ∧
  (2*x - y - 1) * (2*x + y + 1) = 3

theorem system_solutions :
  ∀ x y : ℤ, is_solution x y ↔ (x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l788_78833


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l788_78834

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : a = 10) :
  let hypotenuse := a * Real.sqrt 2
  hypotenuse = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_hypotenuse_l788_78834


namespace NUMINAMATH_CALUDE_round_table_seats_l788_78835

/-- A round table with equally spaced seats numbered clockwise. -/
structure RoundTable where
  num_seats : ℕ
  seat_numbers : Fin num_seats → ℕ
  seat_numbers_clockwise : ∀ (i j : Fin num_seats), i < j → seat_numbers i < seat_numbers j

/-- Two seats are opposite if they are half the total number of seats apart. -/
def are_opposite (t : RoundTable) (s1 s2 : Fin t.num_seats) : Prop :=
  (s2.val + t.num_seats / 2) % t.num_seats = s1.val

theorem round_table_seats (t : RoundTable) (s1 s2 : Fin t.num_seats) :
  t.seat_numbers s1 = 10 →
  t.seat_numbers s2 = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seats_l788_78835


namespace NUMINAMATH_CALUDE_factorial_last_nonzero_digit_not_periodic_l788_78850

/-- The last nonzero digit of n! -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of last nonzero digits of factorials is not eventually periodic -/
theorem factorial_last_nonzero_digit_not_periodic :
  ¬ ∃ (p d : ℕ), p > 0 ∧ d > 0 ∧ 
  ∀ n ≥ d, lastNonzeroDigit n = lastNonzeroDigit (n + p) :=
sorry

end NUMINAMATH_CALUDE_factorial_last_nonzero_digit_not_periodic_l788_78850


namespace NUMINAMATH_CALUDE_negation_of_not_all_zero_l788_78872

theorem negation_of_not_all_zero (a b c : ℝ) :
  ¬(¬(a = 0 ∧ b = 0 ∧ c = 0)) ↔ (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_not_all_zero_l788_78872


namespace NUMINAMATH_CALUDE_carlos_laundry_time_l788_78875

/-- The time it takes for Carlos to do his laundry -/
def laundry_time (num_loads : ℕ) (wash_time_per_load : ℕ) (dry_time : ℕ) : ℕ :=
  num_loads * wash_time_per_load + dry_time

/-- Theorem: Carlos's laundry takes 165 minutes -/
theorem carlos_laundry_time :
  laundry_time 2 45 75 = 165 := by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_time_l788_78875


namespace NUMINAMATH_CALUDE_sallys_out_of_pocket_cost_l788_78825

/-- The amount of money Sally needs to pay out of pocket to buy a reading book for each student -/
theorem sallys_out_of_pocket_cost 
  (budget : ℕ) 
  (book_cost : ℕ) 
  (num_students : ℕ) 
  (h1 : budget = 320)
  (h2 : book_cost = 12)
  (h3 : num_students = 30) :
  (book_cost * num_students - budget : ℕ) = 40 := by
  sorry

#check sallys_out_of_pocket_cost

end NUMINAMATH_CALUDE_sallys_out_of_pocket_cost_l788_78825


namespace NUMINAMATH_CALUDE_max_min_difference_r_l788_78893

theorem max_min_difference_r (p q r : ℝ) 
  (sum_condition : p + q + r = 5)
  (sum_squares_condition : p^2 + q^2 + r^2 = 27) :
  ∃ (r_max r_min : ℝ),
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≤ r_max) ∧
    (∀ r' : ℝ, p + q + r' = 5 ∧ p^2 + q^2 + r'^2 = 27 → r' ≥ r_min) ∧
    r_max - r_min = 8 * Real.sqrt 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_r_l788_78893


namespace NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l788_78885

/-- The slope of a line parallel to the line containing the points (2, -3) and (-4, 5) is -4/3 -/
theorem parallel_line_slope : ℝ → ℝ → Prop :=
  fun x y =>
    let point1 : ℝ × ℝ := (2, -3)
    let point2 : ℝ × ℝ := (-4, 5)
    let slope : ℝ := (point2.2 - point1.2) / (point2.1 - point1.1)
    slope = -4/3

/-- The theorem statement -/
theorem parallel_line_slope_is_negative_four_thirds :
  ∃ (x y : ℝ), parallel_line_slope x y :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_slope_is_negative_four_thirds_l788_78885


namespace NUMINAMATH_CALUDE_order_of_abc_l788_78847

theorem order_of_abc (a b c : ℝ) : 
  a = 5^(1/5) → b = Real.log 3 / Real.log π → c = Real.log 0.2 / Real.log 5 → a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l788_78847


namespace NUMINAMATH_CALUDE_cone_volume_lateral_area_l788_78858

/-- The volume of a cone in terms of its lateral surface area and the distance from the center of the base to the slant height. -/
theorem cone_volume_lateral_area (S r : ℝ) (h1 : S > 0) (h2 : r > 0) : ∃ V : ℝ, V = (1/3) * S * r ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_lateral_area_l788_78858


namespace NUMINAMATH_CALUDE_total_flooring_cost_l788_78894

/-- Represents the dimensions and costs associated with a room's flooring replacement. -/
structure Room where
  length : ℝ
  width : ℝ
  removal_cost : ℝ
  new_flooring_cost_per_sqft : ℝ

/-- Calculates the total cost of replacing flooring in a room. -/
def room_cost (r : Room) : ℝ :=
  r.removal_cost + r.length * r.width * r.new_flooring_cost_per_sqft

/-- Theorem stating that the total cost of replacing flooring in all rooms is $264. -/
theorem total_flooring_cost (living_room bedroom kitchen : Room)
    (h1 : living_room = { length := 8, width := 7, removal_cost := 50, new_flooring_cost_per_sqft := 1.25 })
    (h2 : bedroom = { length := 6, width := 6, removal_cost := 35, new_flooring_cost_per_sqft := 1.50 })
    (h3 : kitchen = { length := 5, width := 4, removal_cost := 20, new_flooring_cost_per_sqft := 1.75 }) :
    room_cost living_room + room_cost bedroom + room_cost kitchen = 264 := by
  sorry

end NUMINAMATH_CALUDE_total_flooring_cost_l788_78894


namespace NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l788_78883

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EFext : ℝ
  FGext : ℝ
  GHext : ℝ
  HEext : ℝ
  area : ℝ

/-- The area of quadrilateral E'F'G'H' is 57 -/
theorem area_of_extended_quadrilateral (q : ExtendedQuadrilateral) 
  (h1 : q.EF = 5)
  (h2 : q.EFext = 5)
  (h3 : q.FG = 6)
  (h4 : q.FGext = 6)
  (h5 : q.GH = 7)
  (h6 : q.GHext = 7)
  (h7 : q.HE = 10)
  (h8 : q.HEext = 10)
  (h9 : q.area = 15)
  (h10 : q.EF = q.EFext) -- Isosceles triangle condition
  : (q.area + 2 * q.area + 12 : ℝ) = 57 := by
  sorry

end NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l788_78883


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l788_78867

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (∃ d : ℝ, k = -1 - d ∧ b = -1 + d) →
  k * 1 + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l788_78867


namespace NUMINAMATH_CALUDE_product_of_squares_l788_78802

theorem product_of_squares (x : ℝ) :
  (2024 - x)^2 + (2022 - x)^2 = 4038 →
  (2024 - x) * (2022 - x) = 2017 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l788_78802


namespace NUMINAMATH_CALUDE_frames_per_page_l788_78811

theorem frames_per_page (total_frames : ℕ) (num_pages : ℕ) (h1 : total_frames = 143) (h2 : num_pages = 13) :
  total_frames / num_pages = 11 := by
  sorry

end NUMINAMATH_CALUDE_frames_per_page_l788_78811


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l788_78876

-- Define the valid range for hours and minutes
def valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23
def valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Theorem statement
theorem max_sum_of_digits_24hour_watch : 
  ∀ h m, valid_hour h → valid_minute m →
  sum_of_digits h + sum_of_digits m ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l788_78876


namespace NUMINAMATH_CALUDE_tripled_base_and_exponent_l788_78804

/-- Given c and d are real numbers with d ≠ 0, and s and y are defined such that
    s = (3c)^(3d) and s = c^d * y^(3d), prove that y = 3c. -/
theorem tripled_base_and_exponent (c d : ℝ) (s y : ℝ) (h1 : d ≠ 0) 
    (h2 : s = (3 * c) ^ (3 * d)) (h3 : s = c^d * y^(3*d)) : y = 3 * c := by
  sorry

end NUMINAMATH_CALUDE_tripled_base_and_exponent_l788_78804


namespace NUMINAMATH_CALUDE_trinomial_square_l788_78882

theorem trinomial_square (c : ℚ) : 
  (∃ b y : ℚ, ∀ x : ℚ, 9*x^2 - 21*x + c = (3*x + b + y)^2) → c = 49/4 := by
sorry

end NUMINAMATH_CALUDE_trinomial_square_l788_78882


namespace NUMINAMATH_CALUDE_product_equals_one_l788_78848

theorem product_equals_one (x y : ℝ) 
  (h : x * y - x / (y^2) - y / (x^2) + x^2 / (y^3) = 4) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l788_78848


namespace NUMINAMATH_CALUDE_aitana_jayda_spending_l788_78810

theorem aitana_jayda_spending (jayda_spent : ℚ) (total_spent : ℚ) 
  (h1 : jayda_spent = 400)
  (h2 : total_spent = 960) : 
  (total_spent - jayda_spent) / jayda_spent = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_aitana_jayda_spending_l788_78810


namespace NUMINAMATH_CALUDE_no_nonneg_integer_solution_l788_78851

theorem no_nonneg_integer_solution (a b : ℕ) (ha : a ≠ b) :
  let d := Nat.gcd a b
  let a' := a / d
  let b' := b / d
  ∀ n : ℕ, (∀ x y : ℕ, a * x + b * y ≠ n) ↔ n = d * (a' * b' - a' - b') := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_integer_solution_l788_78851


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l788_78807

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) : A = 64 * Real.pi → d = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l788_78807


namespace NUMINAMATH_CALUDE_baseball_card_difference_l788_78854

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) 
  (h1 : marcus_cards = 210) 
  (h2 : carter_cards = 152) : 
  marcus_cards - carter_cards = 58 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_difference_l788_78854


namespace NUMINAMATH_CALUDE_total_bathing_suits_l788_78831

/-- The total number of bathing suits is the sum of men's and women's bathing suits. -/
theorem total_bathing_suits
  (men_suits : ℕ)
  (women_suits : ℕ)
  (h1 : men_suits = 14797)
  (h2 : women_suits = 4969) :
  men_suits + women_suits = 19766 :=
by sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l788_78831


namespace NUMINAMATH_CALUDE_sock_pairs_theorem_l788_78837

theorem sock_pairs_theorem (n : ℕ) : 
  (2 * n * (2 * n - 1)) / 2 = 42 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_theorem_l788_78837


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l788_78846

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3.5) * (y - 1.5) = x * y ∧ 
  (x - 3.5) * (y + 2) = x * y ∧ 
  x * y = 294 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l788_78846


namespace NUMINAMATH_CALUDE_earnings_calculation_l788_78895

/-- Calculates the discounted price for a given quantity and unit price with a discount rate and minimum quantity for discount --/
def discountedPrice (quantity : ℕ) (unitPrice : ℚ) (discountRate : ℚ) (minQuantity : ℕ) : ℚ :=
  if quantity ≥ minQuantity then
    (1 - discountRate) * (quantity : ℚ) * unitPrice
  else
    (quantity : ℚ) * unitPrice

/-- Calculates the total earnings after all discounts --/
def totalEarnings (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) : ℚ :=
  let smallPrice := discountedPrice smallQuantity (30 : ℚ) (1/10 : ℚ) 4
  let mediumPrice := discountedPrice mediumQuantity (45 : ℚ) (3/20 : ℚ) 3
  let largePrice := discountedPrice largeQuantity (60 : ℚ) (1/20 : ℚ) 6
  let extraLargePrice := discountedPrice extraLargeQuantity (85 : ℚ) (2/25 : ℚ) 2
  let subtotal := smallPrice + mediumPrice + largePrice + extraLargePrice
  if smallQuantity + mediumQuantity ≥ 10 then
    (97/100 : ℚ) * subtotal
  else
    subtotal

theorem earnings_calculation (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) :
  smallQuantity = 8 ∧ mediumQuantity = 11 ∧ largeQuantity = 4 ∧ extraLargeQuantity = 3 →
  totalEarnings smallQuantity mediumQuantity largeQuantity extraLargeQuantity = (1078.01 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_earnings_calculation_l788_78895


namespace NUMINAMATH_CALUDE_square_area_increase_l788_78870

theorem square_area_increase (x y : ℝ) : 
  (∀ s : ℝ, s = 3 → (s + x)^2 - s^2 = y) → 
  y = x^2 + 6*x := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l788_78870


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l788_78887

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (100000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000000) →
  Nat.gcd a b < 1000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l788_78887
