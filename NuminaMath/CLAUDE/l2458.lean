import Mathlib

namespace NUMINAMATH_CALUDE_circle_ratio_l2458_245895

theorem circle_ratio (r R : ℝ) (hr : r > 0) (hR : R > 0) 
  (h : π * R^2 - π * r^2 = 4 * (π * r^2)) : r / R = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2458_245895


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l2458_245808

theorem bake_sale_group_composition (p : ℕ) : 
  (p : ℚ) > 0 →
  (p / 2 : ℚ) / p = 1 / 2 →
  ((p / 2 - 3) : ℚ) / p = 2 / 5 →
  p / 2 = 15 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_group_composition_l2458_245808


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2458_245802

theorem fractional_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) 
  ↔ k ≠ -3 ∧ k ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2458_245802


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2458_245872

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 1%
    results in Rs. 78 more interest, then P = 2600. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 78) :
  P = 2600 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2458_245872


namespace NUMINAMATH_CALUDE_symmetric_lines_ellipse_intersection_l2458_245816

/-- Given two lines symmetric about y = x + 1 intersecting an ellipse, 
    prove properties about their slopes and intersection points. -/
theorem symmetric_lines_ellipse_intersection 
  (k : ℝ) 
  (h_k_pos : k > 0) 
  (h_k_neq_one : k ≠ 1) 
  (k₁ : ℝ) 
  (h_symmetric : ∀ x y, y = k * x + 1 ↔ y = k₁ * x + 1) 
  (E : Set (ℝ × ℝ)) 
  (h_E : E = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}) 
  (A M N : ℝ × ℝ) 
  (h_A : A ∈ E ∧ A.2 = k * A.1 + 1 ∧ A.2 = k₁ * A.1 + 1) 
  (h_M : M ∈ E ∧ M.2 = k * M.1 + 1) 
  (h_N : N ∈ E ∧ N.2 = k₁ * N.1 + 1) : 
  k * k₁ = 1 ∧ 
  ∃ t : ℝ, (1 - t) * M.1 + t * N.1 = 0 ∧ (1 - t) * M.2 + t * N.2 = -5/3 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_lines_ellipse_intersection_l2458_245816


namespace NUMINAMATH_CALUDE_area_outside_overlapping_squares_l2458_245806

/-- The area of the region outside two overlapping squares within a larger square -/
theorem area_outside_overlapping_squares (large_side : ℝ) (small_side : ℝ) 
  (h_large : large_side = 9) 
  (h_small : small_side = 4) : 
  large_side^2 - 2 * small_side^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_overlapping_squares_l2458_245806


namespace NUMINAMATH_CALUDE_tournament_divisibility_l2458_245809

theorem tournament_divisibility (n : ℕ) 
  (h1 : ∃ (m : ℕ), (n * (n - 1) / 2 + 2 * n^2 - m) = 5 / 4 * (2 * n * (2 * n - 1) + m)) : 
  9 ∣ (3 * n) := by
sorry

end NUMINAMATH_CALUDE_tournament_divisibility_l2458_245809


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2458_245880

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) * (x^2 - x + 1) > 0 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2458_245880


namespace NUMINAMATH_CALUDE_aunt_may_milk_problem_l2458_245890

/-- Aunt May's milk problem -/
theorem aunt_may_milk_problem (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by sorry

end NUMINAMATH_CALUDE_aunt_may_milk_problem_l2458_245890


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2458_245894

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 → 
  (∃ (k : ℝ), (1 + a * i) / (2 - i) = k * i) → 
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2458_245894


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2458_245876

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 30 + (5 : ℚ) / 300 + (7 : ℚ) / 3000 = 0.119 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2458_245876


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l2458_245847

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l2458_245847


namespace NUMINAMATH_CALUDE_fish_comparison_l2458_245805

theorem fish_comparison (x g s r : ℕ) : 
  x > 0 ∧ 
  x = g + s + r ∧ 
  x - g = (2 * x) / 3 - 1 ∧ 
  x - r = (2 * x) / 3 + 4 → 
  s = g + 2 := by
sorry

end NUMINAMATH_CALUDE_fish_comparison_l2458_245805


namespace NUMINAMATH_CALUDE_fraction_inequality_l2458_245827

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / ((a - c)^2) > e / ((b - d)^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2458_245827


namespace NUMINAMATH_CALUDE_det_of_matrix_l2458_245866

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; -3, 5]

theorem det_of_matrix : Matrix.det matrix = 29 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_l2458_245866


namespace NUMINAMATH_CALUDE_area_of_region_R_l2458_245823

/-- A square with side length 3 -/
structure Square :=
  (side_length : ℝ)
  (is_three : side_length = 3)

/-- The region R within the square -/
def region_R (s : Square) := {p : ℝ × ℝ | 
  p.1 ≥ 0 ∧ p.1 ≤ s.side_length ∧ 
  p.2 ≥ 0 ∧ p.2 ≤ s.side_length ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + p.2^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < p.1^2 + (p.2 - s.side_length)^2 ∧
  (p.1 - s.side_length)^2 + p.2^2 < (p.1 - s.side_length)^2 + (p.2 - s.side_length)^2
}

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of region R in a square with side length 3 is 9/4 -/
theorem area_of_region_R (s : Square) : area (region_R s) = 9/4 := by sorry

end NUMINAMATH_CALUDE_area_of_region_R_l2458_245823


namespace NUMINAMATH_CALUDE_boy_travel_time_l2458_245824

/-- Proves that given the conditions of the problem, the boy arrives 8 minutes early on the second day -/
theorem boy_travel_time (distance : ℝ) (speed_day1 speed_day2 : ℝ) (late_time : ℝ) : 
  distance = 2.5 →
  speed_day1 = 5 →
  speed_day2 = 10 →
  late_time = 7 / 60 →
  let time_day1 : ℝ := distance / speed_day1
  let on_time : ℝ := time_day1 - late_time
  let time_day2 : ℝ := distance / speed_day2
  (on_time - time_day2) * 60 = 8 := by sorry

end NUMINAMATH_CALUDE_boy_travel_time_l2458_245824


namespace NUMINAMATH_CALUDE_larger_sample_more_accurate_l2458_245874

-- Define a sampling survey
structure SamplingSurvey where
  population : Set ℝ
  sample : Set ℝ
  sample_size : ℕ

-- Define estimation accuracy
def estimation_accuracy (survey : SamplingSurvey) : ℝ := sorry

-- Theorem stating that larger sample size leads to more accurate estimation
theorem larger_sample_more_accurate (survey1 survey2 : SamplingSurvey) 
  (h : survey1.population = survey2.population) 
  (h_size : survey1.sample_size < survey2.sample_size) : 
  estimation_accuracy survey1 < estimation_accuracy survey2 := by
  sorry

end NUMINAMATH_CALUDE_larger_sample_more_accurate_l2458_245874


namespace NUMINAMATH_CALUDE_product_equals_negative_nine_tenths_l2458_245869

theorem product_equals_negative_nine_tenths :
  12 * (-0.5) * (3/4 : ℚ) * 0.20 = -9/10 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_negative_nine_tenths_l2458_245869


namespace NUMINAMATH_CALUDE_f_divides_characterization_l2458_245846

def f (x : ℕ) : ℕ := x^2 + x + 1

def is_valid (n : ℕ) : Prop :=
  n = 1 ∨ 
  (Nat.Prime n ∧ n % 3 = 1) ∨ 
  (∃ p, Nat.Prime p ∧ p ≠ 3 ∧ n = p^2)

theorem f_divides_characterization (n : ℕ) :
  (∀ k : ℕ, k > 0 → k ∣ n → f k ∣ f n) ↔ is_valid n :=
sorry

end NUMINAMATH_CALUDE_f_divides_characterization_l2458_245846


namespace NUMINAMATH_CALUDE_shelby_rainy_driving_time_l2458_245897

/-- Represents the driving scenario of Shelby --/
structure DrivingScenario where
  sunny_speed : ℝ  -- Speed in sunny conditions (mph)
  rainy_speed : ℝ  -- Speed in rainy conditions (mph)
  total_distance : ℝ  -- Total distance covered (miles)
  total_time : ℝ  -- Total time of travel (minutes)

/-- Calculates the time spent driving in rainy conditions --/
def rainy_time (scenario : DrivingScenario) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specific conditions, the rainy driving time is 40 minutes --/
theorem shelby_rainy_driving_time :
  let scenario : DrivingScenario := {
    sunny_speed := 35,
    rainy_speed := 25,
    total_distance := 22.5,
    total_time := 50
  }
  rainy_time scenario = 40 := by
  sorry

end NUMINAMATH_CALUDE_shelby_rainy_driving_time_l2458_245897


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2458_245819

theorem hyperbola_eccentricity (a : ℝ) : 
  a > 0 →
  (∃ x y : ℝ, x^2/a^2 - y^2/3^2 = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = (a^2 + 3^2)/a^2) →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2458_245819


namespace NUMINAMATH_CALUDE_shortest_handspan_l2458_245842

def sangwon_handspan : ℝ := 19 + 0.8
def doyoon_handspan : ℝ := 18.9
def changhyeok_handspan : ℝ := 19.3

theorem shortest_handspan :
  doyoon_handspan < sangwon_handspan ∧ doyoon_handspan < changhyeok_handspan :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_handspan_l2458_245842


namespace NUMINAMATH_CALUDE_tom_net_calories_consumed_l2458_245867

/-- Calculates the net calories consumed from candy bars in a week -/
def net_calories_from_candy_bars (calories_per_bar : ℕ) (bars_per_week : ℕ) (calories_burned : ℕ) : ℤ :=
  (calories_per_bar * bars_per_week : ℤ) - calories_burned

/-- Proves that given the conditions, Tom consumes 1082 net calories from candy bars in a week -/
theorem tom_net_calories_consumed : 
  net_calories_from_candy_bars 347 6 1000 = 1082 := by
  sorry

#eval net_calories_from_candy_bars 347 6 1000

end NUMINAMATH_CALUDE_tom_net_calories_consumed_l2458_245867


namespace NUMINAMATH_CALUDE_birthday_cake_candles_l2458_245848

theorem birthday_cake_candles (total : ℕ) (yellow : ℕ) (red : ℕ) (blue : ℕ) : 
  total = 79 →
  yellow = 27 →
  red = 14 →
  blue = total - yellow - red →
  blue = 38 := by
sorry

end NUMINAMATH_CALUDE_birthday_cake_candles_l2458_245848


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2458_245864

theorem inequality_solution_set (x : ℝ) :
  (3 - x) / (2 * x - 4) < 1 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2458_245864


namespace NUMINAMATH_CALUDE_problem_2023_squared_minus_2024_times_2022_l2458_245814

theorem problem_2023_squared_minus_2024_times_2022 : 2023^2 - 2024 * 2022 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_squared_minus_2024_times_2022_l2458_245814


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2458_245812

theorem negation_of_proposition (p : Prop) :
  (¬ (∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2458_245812


namespace NUMINAMATH_CALUDE_machine_fill_time_l2458_245815

theorem machine_fill_time (time_A time_AB : ℝ) (time_A_pos : time_A > 0) (time_AB_pos : time_AB > 0) :
  time_A = 20 → time_AB = 12 → ∃ time_B : ℝ, time_B > 0 ∧ 1 / time_A + 1 / time_B = 1 / time_AB ∧ time_B = 30 :=
by sorry

end NUMINAMATH_CALUDE_machine_fill_time_l2458_245815


namespace NUMINAMATH_CALUDE_max_distance_is_217_12_l2458_245836

-- Define the constants
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def total_gallons : ℝ := 23

-- Define the percentages for regular and peak traffic
def regular_highway_percent : ℝ := 0.4
def regular_city_percent : ℝ := 0.6
def peak_highway_percent : ℝ := 0.25
def peak_city_percent : ℝ := 0.75

-- Calculate distances for regular and peak traffic
def regular_distance : ℝ := 
  (regular_highway_percent * total_gallons * highway_mpg) + 
  (regular_city_percent * total_gallons * city_mpg)

def peak_distance : ℝ := 
  (peak_highway_percent * total_gallons * highway_mpg) + 
  (peak_city_percent * total_gallons * city_mpg)

-- Theorem to prove
theorem max_distance_is_217_12 : 
  max regular_distance peak_distance = 217.12 := by sorry

end NUMINAMATH_CALUDE_max_distance_is_217_12_l2458_245836


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l2458_245875

theorem repeating_decimal_division (A B C D : Nat) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (100 * A + 10 * B + C) / (1000 * B + 100 * B + 10 * B + B) = 
    (1000 * B + 100 * C + 10 * D + B) / 9999 →
  A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l2458_245875


namespace NUMINAMATH_CALUDE_smallest_number_l2458_245889

theorem smallest_number (S : Set ℚ) (h : S = {-3, -1, 0, 1}) : 
  ∃ (m : ℚ), m ∈ S ∧ ∀ (x : ℚ), x ∈ S → m ≤ x ∧ m = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2458_245889


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l2458_245818

theorem sum_of_squares_problem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 52)
  (sum_of_products : a*b + b*c + c*a = 28) :
  a + b + c = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l2458_245818


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2458_245877

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 100 ∧ n % sum_of_digits n = 0

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid n) ∧ S.card = 24 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2458_245877


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l2458_245841

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i^11 + i^111 + i^222 = -2*i - 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l2458_245841


namespace NUMINAMATH_CALUDE_equation_solution_l2458_245886

theorem equation_solution :
  ∀ x : ℚ, x ≠ 4 → ((7 * x + 2) / (x - 4) = -6 / (x - 4) ↔ x = -8 / 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2458_245886


namespace NUMINAMATH_CALUDE_notebook_pen_cost_l2458_245865

/-- The cost of notebooks and pens -/
theorem notebook_pen_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.40)
  (h2 : 2 * x + 5 * y = 9.75) :
  x + 3 * y = 5.53 := by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_l2458_245865


namespace NUMINAMATH_CALUDE_f_range_implies_a_value_l2458_245857

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    -x + 3
  else if 2 ≤ x ∧ x ≤ 8 then
    1 + Real.log (2 * x) / Real.log (a^2 - 1)
  else
    0  -- undefined for other x values

theorem f_range_implies_a_value (a : ℝ) :
  (∀ y ∈ Set.range (f a), 2 ≤ y ∧ y ≤ 5) →
  (a = Real.sqrt 3 ∨ a = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_a_value_l2458_245857


namespace NUMINAMATH_CALUDE_complex_simplification_l2458_245868

theorem complex_simplification : 
  ((-2 + Complex.I * Real.sqrt 7) / 3) ^ 4 + 
  ((-2 - Complex.I * Real.sqrt 7) / 3) ^ 4 = 242 / 81 := by
sorry

end NUMINAMATH_CALUDE_complex_simplification_l2458_245868


namespace NUMINAMATH_CALUDE_constant_expression_l2458_245825

theorem constant_expression (x y z : ℝ) 
  (h1 : x * y + y * z + z * x = 4) 
  (h2 : x * y * z = 6) : 
  (x*y - 3/2*(x+y)) * (y*z - 3/2*(y+z)) * (z*x - 3/2*(z+x)) = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_l2458_245825


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2458_245899

theorem min_value_fraction_sum (x y a b : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) 
  (h_sum : x + y = 1) : 
  (a / x + b / y) ≥ (Real.sqrt a + Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2458_245899


namespace NUMINAMATH_CALUDE_area_code_digits_l2458_245838

/-- The number of valid area codes for n digits -/
def validCodes (n : ℕ) : ℕ := 3^n - 1

theorem area_code_digits : 
  ∃ n : ℕ, n > 0 ∧ validCodes n = 26 := by sorry

end NUMINAMATH_CALUDE_area_code_digits_l2458_245838


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2458_245856

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, m < n →
    ¬((m ≥ 10000 ∧ m < 100000) ∧
      (∃ x : ℕ, m = x^2) ∧
      (∃ y : ℕ, m = y^3))) ∧
  n = 15625 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2458_245856


namespace NUMINAMATH_CALUDE_soda_volume_difference_is_14_l2458_245834

/-- Calculates the difference in soda volume between Julio and Mateo -/
def soda_volume_difference : ℕ :=
  let julio_orange := 4
  let julio_grape := 7
  let mateo_orange := 1
  let mateo_grape := 3
  let liters_per_bottle := 2
  let julio_total := (julio_orange + julio_grape) * liters_per_bottle
  let mateo_total := (mateo_orange + mateo_grape) * liters_per_bottle
  julio_total - mateo_total

theorem soda_volume_difference_is_14 : soda_volume_difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_soda_volume_difference_is_14_l2458_245834


namespace NUMINAMATH_CALUDE_roses_kept_l2458_245860

/-- Given that Ian had 20 roses and gave away specific numbers to different people,
    prove that he kept exactly 1 rose. -/
theorem roses_kept (total : ℕ) (mother grandmother sister : ℕ)
    (h1 : total = 20)
    (h2 : mother = 6)
    (h3 : grandmother = 9)
    (h4 : sister = 4) :
    total - (mother + grandmother + sister) = 1 := by
  sorry

end NUMINAMATH_CALUDE_roses_kept_l2458_245860


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2458_245822

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 5 ∧ 2 - x ≤ 1) ↔ (1 ≤ x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2458_245822


namespace NUMINAMATH_CALUDE_point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l2458_245879

/-- Defines the plane region x + y - 1 ≤ 0 -/
def in_plane_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

/-- The point (-1,3) is not in the plane region -/
theorem point_c_not_in_region : ¬ in_plane_region (-1) 3 := by sorry

/-- Point A (0,0) is in the plane region -/
theorem point_a_in_region : in_plane_region 0 0 := by sorry

/-- Point B (-1,1) is in the plane region -/
theorem point_b_in_region : in_plane_region (-1) 1 := by sorry

/-- Point D (2,-3) is in the plane region -/
theorem point_d_in_region : in_plane_region 2 (-3) := by sorry

/-- The main theorem combining all results -/
theorem main_result : 
  ¬ in_plane_region (-1) 3 ∧ 
  in_plane_region 0 0 ∧ 
  in_plane_region (-1) 1 ∧ 
  in_plane_region 2 (-3) := by sorry

end NUMINAMATH_CALUDE_point_c_not_in_region_point_a_in_region_point_b_in_region_point_d_in_region_main_result_l2458_245879


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l2458_245840

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l2458_245840


namespace NUMINAMATH_CALUDE_simplify_expression_l2458_245850

theorem simplify_expression : (27 * (10 ^ 12)) / (9 * (10 ^ 5)) = 30000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2458_245850


namespace NUMINAMATH_CALUDE_problem_solution_l2458_245801

theorem problem_solution :
  (∀ x : ℝ, x^2 = 0 → x = 0) ∧
  (∀ x : ℝ, x^2 < 2*x → x > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2458_245801


namespace NUMINAMATH_CALUDE_line_tangent_to_circumcircle_l2458_245810

/-- Represents a line in the form x = my + n -/
structure Line where
  m : ℝ
  n : ℝ
  h : n > 0

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  x = l.m * y + l.n

/-- Represents the feasible region with its circumcircle -/
structure FeasibleRegion where
  diameter : ℝ

/-- Main theorem -/
theorem line_tangent_to_circumcircle (l : Line) (fr : FeasibleRegion) :
  l.passesThrough 4 4 → fr.diameter = 8 → l.n = 4 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circumcircle_l2458_245810


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l2458_245830

/-- The initial amount of money given the lending conditions --/
theorem initial_amount_theorem (amount_to_B : ℝ) 
  (h1 : amount_to_B = 4000.0000000000005)
  (h2 : ∃ amount_to_A : ℝ, 
    amount_to_A * 0.15 * 2 = amount_to_B * 0.18 * 2 + 360) :
  ∃ initial_amount : ℝ, initial_amount = 10000.000000000002 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l2458_245830


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2458_245893

/-- A quadratic function f(x) = x² - mx + c -/
def f (m c x : ℝ) : ℝ := x^2 - m*x + c

/-- The derivative of f with respect to x -/
def f' (m : ℝ) (x : ℝ) : ℝ := 2*x - m

theorem quadratic_decreasing_condition (m c : ℝ) :
  (∀ x < 1, (f' m x) < 0) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l2458_245893


namespace NUMINAMATH_CALUDE_johns_earnings_l2458_245839

def tax_calculation (earnings : ℕ) : Prop :=
  let deductions : ℕ := 30000
  let taxable_income : ℕ := earnings - deductions
  let first_bracket : ℕ := 20000
  let first_rate : ℚ := 1/10
  let second_rate : ℚ := 1/5
  let total_tax : ℕ := 12000
  (min taxable_income first_bracket) * first_rate +
  (max (taxable_income - first_bracket) 0) * second_rate = total_tax

theorem johns_earnings : ∃ (earnings : ℕ), tax_calculation earnings ∧ earnings = 100000 :=
sorry

end NUMINAMATH_CALUDE_johns_earnings_l2458_245839


namespace NUMINAMATH_CALUDE_number_division_problem_l2458_245873

theorem number_division_problem (x : ℝ) : x / 5 = 100 + x / 6 → x = 3000 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2458_245873


namespace NUMINAMATH_CALUDE_rainy_days_count_l2458_245817

theorem rainy_days_count (n : ℤ) (R : ℕ) (NR : ℕ) : 
  n * R + 3 * NR = 26 →  -- Total cups equation
  3 * NR - n * R = 10 →  -- Difference in cups equation
  R + NR = 7 →           -- Total days equation
  R = 1 :=                -- Conclusion: 1 rainy day
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l2458_245817


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2458_245870

/-- Given a geometric sequence {a_n} where a_3 = 9 and a_6 = 243, 
    prove that the general term formula is a_n = 3^(n-1) -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a3 : a 3 = 9) 
  (h_a6 : a 6 = 243) : 
  ∀ n : ℕ, a n = 3^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2458_245870


namespace NUMINAMATH_CALUDE_gcd_90_252_l2458_245820

theorem gcd_90_252 : Nat.gcd 90 252 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_252_l2458_245820


namespace NUMINAMATH_CALUDE_number_line_position_l2458_245878

/-- Given a number line where the distance from 0 to 25 is divided into 5 equal steps,
    the position after 4 steps from 0 is 20. -/
theorem number_line_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) :
  total_distance = 25 ∧ total_steps = 5 ∧ steps_taken = 4 →
  (total_distance / total_steps) * steps_taken = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_line_position_l2458_245878


namespace NUMINAMATH_CALUDE_decision_box_distinguishes_l2458_245862

-- Define the components of control structures
inductive ControlComponent
  | ProcessingBox
  | DecisionBox
  | StartEndBox
  | InputOutputBox

-- Define the types of control structures
structure ControlStructure where
  components : List ControlComponent

-- Define a selection structure
def SelectionStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.DecisionBox, 
                 ControlComponent.StartEndBox, ControlComponent.InputOutputBox]
}

-- Define a sequential structure
def SequentialStructure : ControlStructure := {
  components := [ControlComponent.ProcessingBox, ControlComponent.StartEndBox, 
                 ControlComponent.InputOutputBox]
}

-- Define the distinguishing feature
def isDistinguishingFeature (component : ControlComponent) 
                            (struct1 struct2 : ControlStructure) : Prop :=
  (component ∈ struct1.components) ∧ (component ∉ struct2.components)

-- Theorem stating that the decision box is the distinguishing feature
theorem decision_box_distinguishes :
  isDistinguishingFeature ControlComponent.DecisionBox SelectionStructure SequentialStructure :=
by
  sorry


end NUMINAMATH_CALUDE_decision_box_distinguishes_l2458_245862


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2458_245887

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≤ 1) ↔ (∃ x : ℝ, x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2458_245887


namespace NUMINAMATH_CALUDE_xyz_sign_sum_l2458_245831

theorem xyz_sign_sum (x y z : ℝ) (h : x * y * z / |x * y * z| = 1) :
  |x| / x + y / |y| + |z| / z = -1 ∨ |x| / x + y / |y| + |z| / z = 3 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sign_sum_l2458_245831


namespace NUMINAMATH_CALUDE_max_ab_value_l2458_245881

theorem max_ab_value (a b : ℝ) : 
  (∃! x, x^2 - 2*a*x - b^2 + 12 ≤ 0) → 
  ∀ c, a*b ≤ c → c ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2458_245881


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l2458_245851

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Condition for three consecutive terms of a sequence to form a geometric sequence -/
def IsGeometric (a : Sequence) (n : ℕ) : Prop :=
  ∃ r : ℝ, a (n + 1) = a n * r ∧ a (n + 2) = a (n + 1) * r

/-- The condition a_{n+1}^2 = a_n * a_{n+2} -/
def SquareMiddleCondition (a : Sequence) (n : ℕ) : Prop :=
  a (n + 1) ^ 2 = a n * a (n + 2)

theorem geometric_sequence_condition (a : Sequence) :
  (∀ n : ℕ, IsGeometric a n → SquareMiddleCondition a n) ∧
  ¬(∀ n : ℕ, SquareMiddleCondition a n → IsGeometric a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l2458_245851


namespace NUMINAMATH_CALUDE_average_math_chemistry_l2458_245813

-- Define the marks for each subject
variable (M P C : ℕ)

-- Define the conditions
axiom total_math_physics : M + P = 70
axiom chemistry_score : C = P + 20

-- Define the theorem to prove
theorem average_math_chemistry : (M + C) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_math_chemistry_l2458_245813


namespace NUMINAMATH_CALUDE_min_circular_arrangement_with_shared_digit_l2458_245858

/-- A function that checks if two natural numbers share a common digit in their decimal representation -/
def share_digit (a b : ℕ) : Prop := sorry

/-- A function that represents a circular arrangement of numbers from 1 to n -/
def circular_arrangement (n : ℕ) : (ℕ → ℕ) := sorry

/-- The main theorem stating that 29 is the smallest number satisfying the conditions -/
theorem min_circular_arrangement_with_shared_digit :
  ∀ n : ℕ, n ≥ 2 →
  (∃ arr : ℕ → ℕ, 
    (∀ i : ℕ, arr i ≤ n) ∧ 
    (∀ i : ℕ, share_digit (arr i) (arr (i + 1))) ∧
    (∀ k : ℕ, k ≤ n → ∃ i : ℕ, arr i = k)) →
  n ≥ 29 :=
sorry

end NUMINAMATH_CALUDE_min_circular_arrangement_with_shared_digit_l2458_245858


namespace NUMINAMATH_CALUDE_elises_initial_money_l2458_245828

/-- Proves that Elise's initial amount of money was $8 --/
theorem elises_initial_money :
  ∀ (initial savings comic_cost puzzle_cost final : ℕ),
  savings = 13 →
  comic_cost = 2 →
  puzzle_cost = 18 →
  final = 1 →
  initial + savings - comic_cost - puzzle_cost = final →
  initial = 8 := by
sorry

end NUMINAMATH_CALUDE_elises_initial_money_l2458_245828


namespace NUMINAMATH_CALUDE_functional_equation_not_surjective_l2458_245807

/-- A function from reals to natural numbers satisfying a specific functional equation -/
def FunctionalEquation (f : ℝ → ℕ) : Prop :=
  ∀ x y : ℝ, f (x + 1 / (f y : ℝ)) = f (y + 1 / (f x : ℝ))

/-- Theorem stating that a function satisfying the functional equation cannot map onto all natural numbers -/
theorem functional_equation_not_surjective (f : ℝ → ℕ) (h : FunctionalEquation f) : 
  ¬(Set.range f = Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_not_surjective_l2458_245807


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2458_245804

theorem min_value_reciprocal_sum (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_sum : 2*m + n = 1) : 
  (1/m + 2/n) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2458_245804


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2458_245859

/-- A geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi) : 
  Real.tan (a 2 * a 12) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2458_245859


namespace NUMINAMATH_CALUDE_possible_values_of_2a_plus_b_l2458_245863

theorem possible_values_of_2a_plus_b (a b x y z : ℕ) :
  a^x = b^y ∧ 
  a^x = 1994^z ∧ 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / z →
  2*a + b = 1001 ∨ 2*a + b = 1996 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_2a_plus_b_l2458_245863


namespace NUMINAMATH_CALUDE_sales_discount_effect_l2458_245832

theorem sales_discount_effect (P N : ℝ) (h_positive : P > 0 ∧ N > 0) :
  let D : ℝ := 10  -- Discount percentage
  let new_price : ℝ := P * (1 - D / 100)
  let new_quantity : ℝ := N * 1.20
  let original_income : ℝ := P * N
  let new_income : ℝ := new_price * new_quantity
  (new_quantity = N * 1.20) ∧ (new_income = original_income * 1.08) :=
by sorry

end NUMINAMATH_CALUDE_sales_discount_effect_l2458_245832


namespace NUMINAMATH_CALUDE_salt_solution_mixture_l2458_245833

theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Ensure total volume is positive
  0.60 * x = 0.10 * ((1 : ℝ) + x) → 
  x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_mixture_l2458_245833


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2458_245871

/-- Proves that the rate of interest is 7% given the problem conditions -/
theorem interest_rate_calculation (loan_amount interest_paid : ℚ) : 
  loan_amount = 1500 →
  interest_paid = 735 →
  ∃ (rate : ℚ), 
    (interest_paid = loan_amount * rate * rate / 100) ∧
    (rate = 7) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2458_245871


namespace NUMINAMATH_CALUDE_complement_of_A_l2458_245892

-- Define the set A
def A : Set ℝ := {x : ℝ | x ≤ 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2458_245892


namespace NUMINAMATH_CALUDE_series_solution_l2458_245884

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The given series as a function of k -/
noncomputable def givenSeries (k : ℝ) : ℝ :=
  4 + geometricSum ((4 + k) / 5) (1 / 5)

theorem series_solution :
  ∃ k : ℝ, givenSeries k = 10 ∧ k = 16 := by sorry

end NUMINAMATH_CALUDE_series_solution_l2458_245884


namespace NUMINAMATH_CALUDE_maize_donated_amount_l2458_245854

/-- The amount of maize donated to Alfred -/
def maize_donated (
  stored_per_month : ℕ)  -- Amount of maize stored per month
  (months : ℕ)           -- Number of months
  (stolen : ℕ)           -- Amount of maize stolen
  (final_amount : ℕ)     -- Final amount of maize after 2 years
  : ℕ :=
  final_amount - (stored_per_month * months - stolen)

/-- Theorem stating the amount of maize donated to Alfred -/
theorem maize_donated_amount :
  maize_donated 1 24 5 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maize_donated_amount_l2458_245854


namespace NUMINAMATH_CALUDE_number_division_addition_l2458_245821

theorem number_division_addition : ∃ x : ℝ, 7500 + x / 50 = 7525 := by
  sorry

end NUMINAMATH_CALUDE_number_division_addition_l2458_245821


namespace NUMINAMATH_CALUDE_contractor_payment_example_l2458_245845

/-- Calculates the total amount a contractor receives given the contract terms and absent days. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℚ) * payment_per_day - (absent_days : ℚ) * fine_per_day

/-- Theorem stating that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment_example : 
  contractor_payment 30 25 7.5 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_example_l2458_245845


namespace NUMINAMATH_CALUDE_monday_temperature_l2458_245888

theorem monday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + 36) / 4 = 46)
  : mon = 44 := by
  sorry

end NUMINAMATH_CALUDE_monday_temperature_l2458_245888


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2458_245891

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 = 0}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + p.2 = 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(0, 0)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2458_245891


namespace NUMINAMATH_CALUDE_linear_function_intersection_l2458_245852

theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ abs x = 4) → k = 1/2 ∨ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l2458_245852


namespace NUMINAMATH_CALUDE_integral_evaluation_l2458_245855

open Set
open MeasureTheory
open Interval

theorem integral_evaluation :
  ∫ x in (-2)..2, (x^2 * Real.sin x + Real.sqrt (4 - x^2)) = 2 * Real.pi :=
by
  have h1 : ∫ x in (-2)..2, x^2 * Real.sin x = 0 := sorry
  have h2 : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := sorry
  sorry

end NUMINAMATH_CALUDE_integral_evaluation_l2458_245855


namespace NUMINAMATH_CALUDE_jogger_difference_l2458_245861

def jogger_problem (tyson martha alexander christopher natasha : ℕ) : Prop :=
  martha = max 0 (tyson - 15) ∧
  alexander = tyson + 22 ∧
  christopher = 20 * tyson ∧
  natasha = 2 * (martha + alexander) ∧
  christopher = 80

theorem jogger_difference (tyson martha alexander christopher natasha : ℕ) 
  (h : jogger_problem tyson martha alexander christopher natasha) : 
  christopher - natasha = 28 := by
sorry

end NUMINAMATH_CALUDE_jogger_difference_l2458_245861


namespace NUMINAMATH_CALUDE_cow_value_increase_l2458_245843

/-- Calculates the increase in value of a cow after weight gain -/
theorem cow_value_increase (initial_weight : ℝ) (weight_factor : ℝ) (price_per_pound : ℝ)
  (h1 : initial_weight = 400)
  (h2 : weight_factor = 1.5)
  (h3 : price_per_pound = 3) :
  (initial_weight * weight_factor - initial_weight) * price_per_pound = 600 := by
  sorry

#check cow_value_increase

end NUMINAMATH_CALUDE_cow_value_increase_l2458_245843


namespace NUMINAMATH_CALUDE_swimming_problem_l2458_245853

/-- Proves that Jamir swims 20 more meters per day than Sarah given the conditions of the swimming problem. -/
theorem swimming_problem (julien sarah jamir : ℕ) : 
  julien = 50 →  -- Julien swims 50 meters per day
  sarah = 2 * julien →  -- Sarah swims twice the distance Julien swims
  jamir > sarah →  -- Jamir swims some more meters per day than Sarah
  7 * (julien + sarah + jamir) = 1890 →  -- Combined distance for the week
  jamir - sarah = 20 := by  -- Jamir swims 20 more meters per day than Sarah
sorry

end NUMINAMATH_CALUDE_swimming_problem_l2458_245853


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2458_245896

theorem semicircle_perimeter (r : ℝ) (h : r = 2.1) : 
  let perimeter := π * r + 2 * r
  perimeter = π * 2.1 + 4.2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2458_245896


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l2458_245898

/-- In a convex hexagon ABCDEF, prove that the measure of angle D is 145 degrees
    given the following conditions:
    - Angles A, B, and C are congruent
    - Angles D, E, and F are congruent
    - Angle A is 50 degrees less than angle D -/
theorem hexagon_angle_measure (A B C D E F : ℝ) : 
  A = B ∧ B = C ∧  -- Angles A, B, and C are congruent
  D = E ∧ E = F ∧  -- Angles D, E, and F are congruent
  A + 50 = D ∧     -- Angle A is 50 degrees less than angle D
  A + B + C + D + E + F = 720  -- Sum of angles in a hexagon
  → D = 145 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l2458_245898


namespace NUMINAMATH_CALUDE_minimum_third_term_l2458_245883

def SallySequence (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 3, a n = a (n - 1) + a (n - 2)) ∧
  (a 8 = 400)

theorem minimum_third_term (a : ℕ → ℕ) (h : SallySequence a) :
  ∃ (m : ℕ), (∀ (b : ℕ → ℕ), SallySequence b → a 3 ≤ b 3) ∧ (a 3 = m) ∧ (m = 35) := by
  sorry

end NUMINAMATH_CALUDE_minimum_third_term_l2458_245883


namespace NUMINAMATH_CALUDE_marathon_training_duration_l2458_245837

theorem marathon_training_duration (d : ℕ) : 
  (5 * d + 10 * d + 20 * d = 1050) → d = 30 := by
  sorry

end NUMINAMATH_CALUDE_marathon_training_duration_l2458_245837


namespace NUMINAMATH_CALUDE_max_m_value_eight_is_achievable_max_m_is_eight_l2458_245829

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating the maximum value of m -/
theorem max_m_value (t : ℝ) (m : ℝ) (h : ∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) :
  m ≤ 8 :=
sorry

/-- The theorem stating that 8 is achievable -/
theorem eight_is_achievable :
  ∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x :=
sorry

/-- The main theorem combining the above results -/
theorem max_m_is_eight :
  (∃ m : ℝ, ∃ t : ℝ, (∀ x ∈ Set.Icc 1 m, f (x + t) ≤ 3*x) ∧
    (∀ m' > m, ¬∃ t' : ℝ, ∀ x ∈ Set.Icc 1 m', f (x + t') ≤ 3*x)) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 8, f (x + t) ≤ 3*x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_eight_is_achievable_max_m_is_eight_l2458_245829


namespace NUMINAMATH_CALUDE_f_has_unique_zero_l2458_245835

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x + (a / 2) * x^2

theorem f_has_unique_zero (a : ℝ) (h : a ∈ Set.Icc (-Real.exp 1) 0) :
  ∃! x, f a x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_l2458_245835


namespace NUMINAMATH_CALUDE_odd_numbers_sum_greater_than_20000_l2458_245826

/-- The count of odd numbers between 200 and 405 whose sum is greater than 20000 -/
def count_odd_numbers_with_large_sum : ℕ :=
  let first_odd := 201
  let last_odd := 403
  let count := (last_odd - first_odd) / 2 + 1
  count

theorem odd_numbers_sum_greater_than_20000 :
  count_odd_numbers_with_large_sum = 102 :=
sorry


end NUMINAMATH_CALUDE_odd_numbers_sum_greater_than_20000_l2458_245826


namespace NUMINAMATH_CALUDE_unique_year_l2458_245803

def is_valid_year (year : ℕ) : Prop :=
  1000 ≤ year ∧ year < 10000 ∧
  (year / 1000 = 1) ∧
  (year % 1000 * 10 + 1 = 5 * year - 4)

theorem unique_year : ∃! year, is_valid_year year :=
  sorry

end NUMINAMATH_CALUDE_unique_year_l2458_245803


namespace NUMINAMATH_CALUDE_soda_cost_l2458_245800

/-- The cost of items in cents -/
structure ItemCosts where
  burger : ℕ
  soda : ℕ
  fries : ℕ

/-- Represents the purchase combinations -/
inductive Purchase
  | uri1 : Purchase
  | gen1 : Purchase
  | uri2 : Purchase
  | gen2 : Purchase

/-- The cost of each purchase in cents -/
def purchaseCost (p : Purchase) (costs : ItemCosts) : ℕ :=
  match p with
  | .uri1 => 3 * costs.burger + costs.soda
  | .gen1 => 2 * costs.burger + 3 * costs.soda
  | .uri2 => costs.burger + 2 * costs.fries
  | .gen2 => costs.soda + 3 * costs.fries

theorem soda_cost (costs : ItemCosts) 
  (h1 : purchaseCost .uri1 costs = 390)
  (h2 : purchaseCost .gen1 costs = 440)
  (h3 : purchaseCost .uri2 costs = 230)
  (h4 : purchaseCost .gen2 costs = 270) :
  costs.soda = 234 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l2458_245800


namespace NUMINAMATH_CALUDE_decorations_used_l2458_245811

theorem decorations_used (boxes : Nat) (decorations_per_box : Nat) (given_away : Nat) : 
  boxes = 4 → decorations_per_box = 15 → given_away = 25 →
  boxes * decorations_per_box - given_away = 35 := by
  sorry

end NUMINAMATH_CALUDE_decorations_used_l2458_245811


namespace NUMINAMATH_CALUDE_largest_b_value_l2458_245882

theorem largest_b_value (b : ℚ) (h : (3*b + 7) * (b - 2) = 9*b) : b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l2458_245882


namespace NUMINAMATH_CALUDE_f_has_maximum_l2458_245885

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

theorem f_has_maximum : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 28/3 := by
  sorry

end NUMINAMATH_CALUDE_f_has_maximum_l2458_245885


namespace NUMINAMATH_CALUDE_combination_problem_l2458_245849

theorem combination_problem (m : ℕ) : 
  (1 : ℚ) / (Nat.choose 5 m) - (1 : ℚ) / (Nat.choose 6 m) = (7 : ℚ) / (10 * Nat.choose 7 m) →
  Nat.choose 21 m = 210 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l2458_245849


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2458_245844

/-- A triangle with marked points on two sides -/
structure MarkedTriangle where
  -- The number of points marked on side BC
  pointsOnBC : ℕ
  -- The number of points marked on side AB
  pointsOnAB : ℕ
  -- Ensure the points are distinct from vertices
  distinctPoints : pointsOnBC > 0 ∧ pointsOnAB > 0

/-- The number of intersection points formed by connecting marked points -/
def intersectionPoints (t : MarkedTriangle) : ℕ := t.pointsOnBC * t.pointsOnAB

/-- Theorem: The number of intersection points in a triangle with 60 points on BC and 50 points on AB is 3000 -/
theorem intersection_points_theorem (t : MarkedTriangle) 
  (h1 : t.pointsOnBC = 60) (h2 : t.pointsOnAB = 50) : 
  intersectionPoints t = 3000 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2458_245844
