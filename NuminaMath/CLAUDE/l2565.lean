import Mathlib

namespace NUMINAMATH_CALUDE_sequence_sum_l2565_256565

theorem sequence_sum (a : ℕ → ℕ) (h : ∀ k : ℕ, k > 0 → a k + a (k + 1) = 2 * k + 1) :
  a 1 + a 100 = 101 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2565_256565


namespace NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2565_256545

theorem complex_power_one_minus_i_six :
  (1 - Complex.I : ℂ)^6 = 8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_power_one_minus_i_six_l2565_256545


namespace NUMINAMATH_CALUDE_mixed_fraction_power_product_l2565_256548

theorem mixed_fraction_power_product (n : ℕ) (m : ℕ) :
  (-(3 : ℚ) / 2) ^ (2021 : ℕ) * (2 : ℚ) / 3 ^ (2023 : ℕ) = -(4 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_power_product_l2565_256548


namespace NUMINAMATH_CALUDE_log_8_512_l2565_256523

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_8_512 : log 8 512 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_8_512_l2565_256523


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2565_256588

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, -1), then m + b = -9 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The point (x, y) is on the line y = mx + b
    y = m * x + b ∧ 
    -- The point (x, y) is equidistant from (2, 3) and (10, -1)
    (x - 2)^2 + (y - 3)^2 = (x - 10)^2 + (y + 1)^2 ∧
    -- The line through (2, 3) and (10, -1) is perpendicular to y = mx + b
    m * ((10 - 2) / ((-1) - 3)) = -1) →
  m + b = -9 := by sorry


end NUMINAMATH_CALUDE_reflection_line_sum_l2565_256588


namespace NUMINAMATH_CALUDE_fourth_guard_theorem_l2565_256557

/-- Represents a rectangular perimeter with guards at each corner -/
structure GuardedRectangle where
  perimeter : ℝ
  three_guard_distance : ℝ

/-- Calculates the distance run by the fourth guard -/
def fourth_guard_distance (rect : GuardedRectangle) : ℝ :=
  rect.perimeter - rect.three_guard_distance

/-- Theorem stating that for a rectangle with perimeter 1000 meters,
    if three guards run 850 meters, the fourth guard runs 150 meters -/
theorem fourth_guard_theorem (rect : GuardedRectangle)
  (h1 : rect.perimeter = 1000)
  (h2 : rect.three_guard_distance = 850) :
  fourth_guard_distance rect = 150 := by
  sorry

end NUMINAMATH_CALUDE_fourth_guard_theorem_l2565_256557


namespace NUMINAMATH_CALUDE_sum_first_50_digits_1_101_l2565_256524

/-- The decimal representation of 1/101 -/
def decimal_rep_1_101 : ℕ → ℕ
| 0 => 0  -- First digit after decimal point
| 1 => 0  -- Second digit
| 2 => 9  -- Third digit
| 3 => 9  -- Fourth digit
| n + 4 => decimal_rep_1_101 n  -- Repeating pattern

/-- Sum of the first n digits after the decimal point in 1/101 -/
def sum_digits (n : ℕ) : ℕ :=
  (List.range n).map decimal_rep_1_101 |>.sum

/-- The sum of the first 50 digits after the decimal point in the decimal representation of 1/101 is 216 -/
theorem sum_first_50_digits_1_101 : sum_digits 50 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_50_digits_1_101_l2565_256524


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2565_256550

theorem gcd_lcm_product (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  (∃ s : Finset ℕ+, s.card = 4 ∧ ∀ x : ℕ+, x ∈ s ↔ ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 252) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2565_256550


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l2565_256582

theorem a_greater_than_b_squared {a b : ℝ} (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l2565_256582


namespace NUMINAMATH_CALUDE_range_of_m_l2565_256502

theorem range_of_m (p q : Prop) (m : ℝ) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : p ↔ m < 0) 
  (h4 : q ↔ m < 2) : 
  0 ≤ m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2565_256502


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l2565_256553

theorem traditionalist_fraction (num_provinces : ℕ) (num_traditionalists_per_province : ℚ) (num_progressives : ℚ) :
  num_provinces = 5 →
  num_traditionalists_per_province = num_progressives / 15 →
  (num_provinces * num_traditionalists_per_province) / (num_progressives + num_provinces * num_traditionalists_per_province) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l2565_256553


namespace NUMINAMATH_CALUDE_simplify_expression_l2565_256593

theorem simplify_expression (n : ℕ) : (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2565_256593


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l2565_256590

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |a * x^3 - Real.log x| ≥ 1) → a ≥ Real.exp 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l2565_256590


namespace NUMINAMATH_CALUDE_solution_to_system_l2565_256519

theorem solution_to_system (x y : ℝ) 
  (h1 : 9 * x^2 - 25 * y^2 = 0) 
  (h2 : x^2 + y^2 = 10) : 
  (x = 5 * Real.sqrt (45/17) / 3 ∨ x = -5 * Real.sqrt (45/17) / 3) ∧
  (y = Real.sqrt (45/17) ∨ y = -Real.sqrt (45/17)) := by
sorry


end NUMINAMATH_CALUDE_solution_to_system_l2565_256519


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2565_256541

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2565_256541


namespace NUMINAMATH_CALUDE_add_25_to_number_l2565_256578

theorem add_25_to_number (x : ℤ) : 43 + x = 81 → x + 25 = 63 := by
  sorry

end NUMINAMATH_CALUDE_add_25_to_number_l2565_256578


namespace NUMINAMATH_CALUDE_cherry_tomato_yield_theorem_l2565_256538

-- Define the relationship between number of plants and yield
def yield_function (x : ℕ) : ℝ := -0.5 * (x : ℝ) + 5

-- Define the total yield function
def total_yield (x : ℕ) : ℝ := (x : ℝ) * yield_function x

-- Theorem statement
theorem cherry_tomato_yield_theorem (x : ℕ) 
  (h1 : 2 ≤ x ∧ x ≤ 8) 
  (h2 : yield_function 2 = 4) 
  (h3 : ∀ n : ℕ, 2 ≤ n ∧ n < 8 → yield_function (n + 1) = yield_function n - 0.5) :
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → yield_function n = -0.5 * (n : ℝ) + 5) ∧
  (∃ max_yield : ℝ, max_yield = 12.5 ∧ 
    ∀ n : ℕ, 2 ≤ n ∧ n ≤ 8 → total_yield n ≤ max_yield) ∧
  (total_yield 5 = 12.5) :=
by sorry

end NUMINAMATH_CALUDE_cherry_tomato_yield_theorem_l2565_256538


namespace NUMINAMATH_CALUDE_nate_search_speed_l2565_256503

/-- The number of rows in Section G of the parking lot -/
def section_g_rows : ℕ := 15

/-- The number of cars per row in Section G -/
def section_g_cars_per_row : ℕ := 10

/-- The number of rows in Section H of the parking lot -/
def section_h_rows : ℕ := 20

/-- The number of cars per row in Section H -/
def section_h_cars_per_row : ℕ := 9

/-- The time Nate spent searching in minutes -/
def search_time : ℕ := 30

/-- The number of cars Nate can walk past per minute -/
def cars_per_minute : ℕ := 11

theorem nate_search_speed :
  (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / search_time = cars_per_minute := by
  sorry

end NUMINAMATH_CALUDE_nate_search_speed_l2565_256503


namespace NUMINAMATH_CALUDE_hcf_problem_l2565_256577

theorem hcf_problem (A B : ℕ) (H : ℕ) : 
  A = 900 → 
  A > B → 
  B > 0 →
  Nat.lcm A B = H * 11 * 15 →
  Nat.gcd A B = H →
  Nat.gcd A B = 165 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l2565_256577


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l2565_256521

/-- Given a piece of work that can be completed by three workers a, b, and c, 
    this theorem proves the time taken by worker b to complete the work alone. -/
theorem worker_b_completion_time 
  (total_time : ℝ) 
  (time_a : ℝ) 
  (time_c : ℝ) 
  (h1 : total_time = 4) 
  (h2 : time_a = 36) 
  (h3 : time_c = 6) : 
  ∃ (time_b : ℝ), time_b = 18 ∧ 
  1 / total_time = 1 / time_a + 1 / time_b + 1 / time_c :=
by sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l2565_256521


namespace NUMINAMATH_CALUDE_division_remainder_l2565_256558

theorem division_remainder (N : ℕ) : 
  (∃ r : ℕ, N = 5 * 5 + r ∧ r < 5) ∧ 
  (∃ q : ℕ, N = 11 * q + 3) → 
  N % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_division_remainder_l2565_256558


namespace NUMINAMATH_CALUDE_candy_probability_l2565_256526

theorem candy_probability : 
  let total_candies : ℕ := 12
  let green_candies : ℕ := 5
  let yellow_candies : ℕ := 3
  let orange_candies : ℕ := 4
  let picked_candies : ℕ := 4
  let target_green : ℕ := 3

  total_candies = green_candies + yellow_candies + orange_candies →
  (Nat.choose green_candies target_green * Nat.choose (yellow_candies + orange_candies) (picked_candies - target_green)) / 
  Nat.choose total_candies picked_candies = 14 / 99 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_probability_l2565_256526


namespace NUMINAMATH_CALUDE_road_trip_duration_l2565_256562

theorem road_trip_duration 
  (initial_duration : ℕ) 
  (stretch_interval : ℕ) 
  (food_stops : ℕ) 
  (gas_stops : ℕ) 
  (stop_duration : ℕ) 
  (h1 : initial_duration = 14)
  (h2 : stretch_interval = 2)
  (h3 : food_stops = 2)
  (h4 : gas_stops = 3)
  (h5 : stop_duration = 20) :
  initial_duration + 
  (initial_duration / stretch_interval + food_stops + gas_stops) * stop_duration / 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_duration_l2565_256562


namespace NUMINAMATH_CALUDE_completing_square_sum_l2565_256513

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x : ℝ, 64 * x^2 + 48 * x - 36 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 56 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2565_256513


namespace NUMINAMATH_CALUDE_trajectory_is_circle_l2565_256580

/-- A trajectory in 2D space -/
structure Trajectory where
  path : Set (ℝ × ℝ)

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Definition of a circular trajectory -/
def is_circular_trajectory (t : Trajectory) (c : Circle) : Prop :=
  ∀ p ∈ t.path, distance p c.center = c.radius

/-- The given trajectory -/
def given_trajectory : Trajectory := sorry

/-- Point A -/
def point_A : Point := sorry

/-- Theorem stating that the given trajectory is a circle -/
theorem trajectory_is_circle :
  ∃ (c : Circle), c.center = point_A ∧ c.radius = 3 ∧ is_circular_trajectory given_trajectory c := by sorry

end NUMINAMATH_CALUDE_trajectory_is_circle_l2565_256580


namespace NUMINAMATH_CALUDE_total_groom_time_in_minutes_l2565_256510

/-- The time in hours it takes to groom a dog -/
def dog_groom_time : ℝ := 2.5

/-- The time in hours it takes to groom a cat -/
def cat_groom_time : ℝ := 0.5

/-- The number of dogs to be groomed -/
def num_dogs : ℕ := 5

/-- The number of cats to be groomed -/
def num_cats : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating that the total time to groom 5 dogs and 3 cats is 840 minutes -/
theorem total_groom_time_in_minutes : 
  (dog_groom_time * num_dogs + cat_groom_time * num_cats) * minutes_per_hour = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_in_minutes_l2565_256510


namespace NUMINAMATH_CALUDE_prob_select_one_from_2_7_l2565_256595

/-- The decimal representation of 2/7 -/
def decimal_rep_2_7 : List Nat := [2, 8, 5, 7, 1, 4]

/-- The probability of selecting a specific digit from the decimal representation of 2/7 -/
def prob_select_digit (d : Nat) : Rat :=
  (decimal_rep_2_7.count d) / (decimal_rep_2_7.length)

theorem prob_select_one_from_2_7 :
  prob_select_digit 1 = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_select_one_from_2_7_l2565_256595


namespace NUMINAMATH_CALUDE_range_of_m_l2565_256544

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  ¬(x ≤ 1 + m → |x - 4| ≤ 6)) → 
  m ∈ Set.Ici 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2565_256544


namespace NUMINAMATH_CALUDE_max_integers_greater_than_15_l2565_256581

theorem max_integers_greater_than_15 (a b c d e f : ℤ) : 
  a + b + c + d + e + f = 20 →
  (∃ (count : ℕ), count ≤ 6 ∧ 
    (∃ (subset : Finset ℤ), subset.card = count ∧ 
      (∀ x ∈ subset, x > 15) ∧ 
      subset ⊆ {a, b, c, d, e, f})) →
  (∀ (count : ℕ), count ≤ 6 → 
    (∃ (subset : Finset ℤ), subset.card = count ∧ 
      (∀ x ∈ subset, x > 15) ∧ 
      subset ⊆ {a, b, c, d, e, f}) → 
    count ≤ 5) :=
by sorry


end NUMINAMATH_CALUDE_max_integers_greater_than_15_l2565_256581


namespace NUMINAMATH_CALUDE_max_digit_sum_l2565_256516

theorem max_digit_sum (d e f z : ℕ) : 
  d ≤ 9 → e ≤ 9 → f ≤ 9 →
  (d * 100 + e * 10 + f : ℚ) / 1000 = 1 / z →
  0 < z → z ≤ 9 →
  d + e + f ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_digit_sum_l2565_256516


namespace NUMINAMATH_CALUDE_sample_size_determination_l2565_256529

theorem sample_size_determination (total_population : Nat) (n : Nat) : 
  total_population = 36 →
  n > 0 →
  total_population % n = 0 →
  (total_population / n) % 6 = 0 →
  35 % (n + 1) = 0 →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_determination_l2565_256529


namespace NUMINAMATH_CALUDE_third_largest_number_l2565_256517

/-- Given five numbers in a specific ratio with a known product, 
    this theorem proves the value of the third largest number. -/
theorem third_largest_number 
  (a b c d e : ℝ) 
  (ratio : a / 2.3 = b / 3.7 ∧ a / 2.3 = c / 5.5 ∧ a / 2.3 = d / 7.1 ∧ a / 2.3 = e / 8.9) 
  (product : a * b * c * d * e = 900000) : 
  ∃ (ε : ℝ), abs (c - 14.85) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_third_largest_number_l2565_256517


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l2565_256567

-- Define the sets
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | (x - 5) / x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 3 * a ≤ x ∧ x ≤ 2 * a + 1}
def D (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1/2}

-- Part 1
theorem range_of_a : 
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ (a ∈ Set.Ioo 0 (1/2) ∪ Set.Ioi 1) :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ (A ∪ B)) ∧ 
           (∃ y : ℝ, y ∈ (A ∪ B) ∧ y ∉ D m) ↔
           m ∈ Set.Icc (-2) (9/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l2565_256567


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2565_256511

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 < 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2565_256511


namespace NUMINAMATH_CALUDE_inequality_proof_l2565_256515

theorem inequality_proof (m n : ℕ) (h : m < Real.sqrt 2 * n) :
  (m : ℝ) / n < Real.sqrt 2 * (1 - 1 / (4 * n^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2565_256515


namespace NUMINAMATH_CALUDE_max_profit_at_two_l2565_256552

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then -x + 7
  else 0

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (sales_volume x) * (x - 1)

-- Main theorem
theorem max_profit_at_two :
  ∀ x, 1 < x ∧ x ≤ 5 → profit x ≤ profit 2 :=
by sorry

end

end NUMINAMATH_CALUDE_max_profit_at_two_l2565_256552


namespace NUMINAMATH_CALUDE_sum_range_l2565_256508

theorem sum_range (x y z : ℝ) 
  (eq1 : x + 2*y + 3*z = 1) 
  (eq2 : y*z + z*x + x*y = -1) : 
  (3 - 3*Real.sqrt 3) / 4 ≤ x + y + z ∧ x + y + z ≤ (3 + 3*Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_CALUDE_sum_range_l2565_256508


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2565_256528

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2565_256528


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_max_m_for_f_geq_quadratic_l2565_256543

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part (I)
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (II)
theorem max_m_for_f_geq_quadratic :
  ∃ (m : ℝ), m = 2 ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ -x^2 + 2*x + m) ∧
  (∀ m' > m, ∃ x ∈ Set.Icc 0 2, f x < -x^2 + 2*x + m') := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_max_m_for_f_geq_quadratic_l2565_256543


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2565_256537

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2565_256537


namespace NUMINAMATH_CALUDE_range_of_a_for_union_complement_l2565_256539

-- Define the sets M, N, and A
def M : Set ℝ := {x | 1 < x ∧ x < 4}
def N : Set ℝ := {x | 3 < x ∧ x < 5}
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}

-- Define the theorem
theorem range_of_a_for_union_complement (a : ℝ) : 
  (A a ∪ (Set.univ \ N) = Set.univ) ↔ (2 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_union_complement_l2565_256539


namespace NUMINAMATH_CALUDE_consecutive_powers_divisibility_l2565_256540

theorem consecutive_powers_divisibility (a : ℝ) (n : ℕ) :
  ∃ k : ℤ, a^n + a^(n+1) = k * a * (a + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_powers_divisibility_l2565_256540


namespace NUMINAMATH_CALUDE_kombucha_bottle_cost_l2565_256592

/-- Represents the cost of a bottle of kombucha -/
def bottle_cost : ℝ := sorry

/-- Represents the number of bottles Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- Represents the cash refund per bottle in dollars -/
def refund_per_bottle : ℝ := 0.1

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of bottles that can be bought with the yearly refund -/
def bottles_bought_with_refund : ℕ := 6

theorem kombucha_bottle_cost :
  bottle_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kombucha_bottle_cost_l2565_256592


namespace NUMINAMATH_CALUDE_yarn_cost_calculation_l2565_256500

/-- Proves that the cost of each ball of yarn is $6 given the conditions of Chantal's sweater business -/
theorem yarn_cost_calculation (num_sweaters : ℕ) (yarn_per_sweater : ℕ) (price_per_sweater : ℚ) (total_profit : ℚ) : 
  num_sweaters = 28 →
  yarn_per_sweater = 4 →
  price_per_sweater = 35 →
  total_profit = 308 →
  (num_sweaters * price_per_sweater - total_profit) / (num_sweaters * yarn_per_sweater) = 6 := by
sorry

end NUMINAMATH_CALUDE_yarn_cost_calculation_l2565_256500


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2565_256509

theorem constant_term_binomial_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 1120 ∧ 
   ∃ f : ℝ → ℝ, 
   (∀ y, f y = (y - 2/y)^8) ∧
   (∃ g : ℝ → ℝ, (∀ y, f y = g y + c + y * (g y)))) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2565_256509


namespace NUMINAMATH_CALUDE_least_N_congruence_l2565_256551

/-- Sum of digits in base 3 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_congruence : N ≡ 862 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_least_N_congruence_l2565_256551


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l2565_256531

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (2 - (2 - (2 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l2565_256531


namespace NUMINAMATH_CALUDE_ratio_abc_l2565_256556

theorem ratio_abc (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) :
  ∃ k : ℝ, a = k ∧ b = 2*k ∧ c = 3*k :=
sorry

end NUMINAMATH_CALUDE_ratio_abc_l2565_256556


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_ratio_three_implies_m_values_l2565_256561

/-- The quadratic equation x^2 - 4x - m(m+4) = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*x - m*(m+4) = 0

theorem quadratic_always_real_roots :
  ∀ m : ℝ, ∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ x₁ ≠ x₂ :=
sorry

theorem roots_ratio_three_implies_m_values :
  ∀ m x₁ x₂ : ℝ, quadratic_equation x₁ m → quadratic_equation x₂ m → x₂ = 3*x₁ →
  m = -1 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_roots_ratio_three_implies_m_values_l2565_256561


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2565_256594

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ (α β : ℝ), (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2565_256594


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l2565_256568

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_terms
  (a₁ : ℚ) (a₂ : ℚ) (h₁ : a₁ = 4) (h₂ : a₂ = 16/3) :
  let r := a₂ / a₁
  (geometric_sequence a₁ r 10 = 1048576/19683) ∧
  (geometric_sequence a₁ r 5 = 1024/81) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l2565_256568


namespace NUMINAMATH_CALUDE_spring_math_camp_attendance_l2565_256572

theorem spring_math_camp_attendance : ∃ (total boys girls : ℕ),
  total = boys + girls ∧
  50 ≤ total ∧ total ≤ 70 ∧
  3 * boys + 9 * girls = 8 * boys + 2 * girls ∧
  total = 60 := by
  sorry

end NUMINAMATH_CALUDE_spring_math_camp_attendance_l2565_256572


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2565_256589

def p (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2565_256589


namespace NUMINAMATH_CALUDE_probability_reach_4_2_after_6_moves_l2565_256527

/-- The probability of a particle reaching a specific point after a given number of moves -/
def reach_probability (total_moves : ℕ) (right_moves : ℕ) (up_moves : ℕ) : ℚ :=
  (Nat.choose total_moves right_moves : ℚ) * (1 / 2) ^ total_moves

/-- Theorem: The probability of reaching point (4,2) after 6 moves -/
theorem probability_reach_4_2_after_6_moves :
  reach_probability 6 4 2 = (Nat.choose 6 2 : ℚ) * (1 / 2) ^ 6 := by
  sorry

#eval reach_probability 6 4 2

end NUMINAMATH_CALUDE_probability_reach_4_2_after_6_moves_l2565_256527


namespace NUMINAMATH_CALUDE_tyrones_pennies_l2565_256591

/-- The number of pennies Tyrone found -/
def pennies : ℕ := sorry

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The total value of Tyrone's money in dollars -/
def total_value : ℚ := 13

/-- The value of Tyrone's money excluding pennies -/
def value_without_pennies : ℚ :=
  2 * 1 + -- two $1 bills
  1 * 5 + -- one $5 bill
  13 * (1 / 4) + -- 13 quarters
  20 * (1 / 10) + -- 20 dimes
  8 * (1 / 20) -- 8 nickels

theorem tyrones_pennies :
  pennies * penny_value = total_value - value_without_pennies ∧
  pennies = 35 := by sorry

end NUMINAMATH_CALUDE_tyrones_pennies_l2565_256591


namespace NUMINAMATH_CALUDE_function_value_l2565_256507

theorem function_value (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) : f 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l2565_256507


namespace NUMINAMATH_CALUDE_root_in_interval_l2565_256564

def f (x : ℝ) := x^3 - 3*x + 1

theorem root_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2565_256564


namespace NUMINAMATH_CALUDE_stratified_sampling_l2565_256504

theorem stratified_sampling (total : ℕ) (sample_size : ℕ) (group_size : ℕ) 
  (h1 : total = 700) 
  (h2 : sample_size = 14) 
  (h3 : group_size = 300) :
  (group_size * sample_size) / total = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2565_256504


namespace NUMINAMATH_CALUDE_journey_time_increase_l2565_256546

theorem journey_time_increase (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_journey_time_increase_l2565_256546


namespace NUMINAMATH_CALUDE_solution_set_max_t_value_l2565_256583

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 3|

-- Theorem for the solution of f(x) < 5
theorem solution_set (x : ℝ) : f x < 5 ↔ -7/4 < x ∧ x < 3/4 := by sorry

-- Theorem for the maximum value of t
theorem max_t_value : ∃ (a : ℝ), a = 4 ∧ ∀ (t : ℝ), (∀ (x : ℝ), f x - t ≥ 0) ↔ t ≤ a := by sorry

end NUMINAMATH_CALUDE_solution_set_max_t_value_l2565_256583


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l2565_256501

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 →
  side_length = 7 →
  exterior_angle = 90 →
  (360 : ℝ) / n = exterior_angle →
  n * side_length = 28 := by
  sorry

#check regular_polygon_perimeter

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l2565_256501


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2565_256559

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ - 1⌋ = ⌊x + 3⌋) ↔ (5/3 ≤ x ∧ x < 3 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2565_256559


namespace NUMINAMATH_CALUDE_min_value_with_constraint_l2565_256520

theorem min_value_with_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 3) :
  x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 ≥ 162 ∧ 
  (x^2 + 4*x*y + 12*y^2 + 8*y*z + 3*z^2 = 162 ↔ x = 3 ∧ y = 1/2 ∧ z = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_with_constraint_l2565_256520


namespace NUMINAMATH_CALUDE_trig_inequality_l2565_256554

theorem trig_inequality : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l2565_256554


namespace NUMINAMATH_CALUDE_count_grid_paths_l2565_256574

/-- The number of paths from (0,0) to (m, n) on a grid, moving only right or up by one unit at a time -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that the number of paths from (0,0) to (m, n) on a grid,
    moving only right or up by one unit at a time, is equal to (m+n choose m) -/
theorem count_grid_paths (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_count_grid_paths_l2565_256574


namespace NUMINAMATH_CALUDE_max_ways_to_schedule_single_game_l2565_256506

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : Nat
  total_games : Nat
  games_per_day : Nat → Nat

/-- The specific tournament configuration -/
def tournament : ChessTournament :=
  { team_size := 15,
    total_games := 15 * 15,
    games_per_day := fun d => if d = 1 then 15 else 1 }

/-- The number of ways to schedule a single game -/
def ways_to_schedule_single_game (t : ChessTournament) : Nat :=
  t.total_games - t.team_size

theorem max_ways_to_schedule_single_game :
  ways_to_schedule_single_game tournament ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_max_ways_to_schedule_single_game_l2565_256506


namespace NUMINAMATH_CALUDE_exactly_two_ultra_squarish_l2565_256560

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that extracts the first three digits of a seven-digit base 9 number -/
def first_three_digits (n : ℕ) : ℕ := n / (9^4)

/-- A function that extracts the middle three digits of a seven-digit base 9 number -/
def middle_three_digits (n : ℕ) : ℕ := (n / 9^2) % (9^3)

/-- A function that extracts the last three digits of a seven-digit base 9 number -/
def last_three_digits (n : ℕ) : ℕ := n % (9^3)

/-- A function that checks if a number is ultra-squarish -/
def is_ultra_squarish (n : ℕ) : Prop :=
  n ≥ 9^6 ∧ n < 9^7 ∧  -- seven-digit number in base 9
  (∀ d, d ∈ (List.range 7).map (fun i => (n / (9^i)) % 9) → d ≠ 0) ∧  -- no digit is zero
  is_perfect_square n ∧
  is_perfect_square (first_three_digits n) ∧
  is_perfect_square (middle_three_digits n) ∧
  is_perfect_square (last_three_digits n)

/-- The theorem stating that there are exactly 2 ultra-squarish numbers -/
theorem exactly_two_ultra_squarish : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_ultra_squarish n) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_ultra_squarish_l2565_256560


namespace NUMINAMATH_CALUDE_square_area_is_9_l2565_256522

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square ABCD -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The square is entirely below the x-axis -/
def below_x_axis (s : Square) : Prop :=
  s.A.2 ≤ 0 ∧ s.B.2 ≤ 0 ∧ s.C.2 ≤ 0 ∧ s.D.2 ≤ 0

/-- The square is inscribed within the region bounded by the parabola and the x-axis -/
def inscribed_in_parabola (s : Square) : Prop :=
  s.A.2 = 0 ∧ s.B.2 = 0 ∧ s.C.2 = f s.C.1 ∧ s.D.2 = f s.D.1

/-- The top vertex A lies at (2, 0) -/
def top_vertex_at_2_0 (s : Square) : Prop :=
  s.A = (2, 0)

/-- The theorem stating that the area of the square is 9 -/
theorem square_area_is_9 (s : Square) 
    (h1 : below_x_axis s)
    (h2 : inscribed_in_parabola s)
    (h3 : top_vertex_at_2_0 s) : 
  (s.B.1 - s.A.1)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_9_l2565_256522


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l2565_256555

/-- The equation (x+y)^2 = x^2 + y^2 + 1 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ) (k : ℝ), k ≠ 0 ∧
  (∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 1 ↔ (x * y = k ∧ (x / a)^2 - (y / b)^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l2565_256555


namespace NUMINAMATH_CALUDE_gold_quarter_value_ratio_l2565_256575

theorem gold_quarter_value_ratio : 
  let melted_value_per_ounce : ℚ := 100
  let quarter_weight : ℚ := 1 / 5
  let spent_value : ℚ := 1 / 4
  (melted_value_per_ounce * quarter_weight) / spent_value = 80 := by
  sorry

end NUMINAMATH_CALUDE_gold_quarter_value_ratio_l2565_256575


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2565_256518

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  1/x + 2/y ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2565_256518


namespace NUMINAMATH_CALUDE_percent_greater_l2565_256505

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
sorry

end NUMINAMATH_CALUDE_percent_greater_l2565_256505


namespace NUMINAMATH_CALUDE_find_set_N_l2565_256535

def U : Set ℕ := {1, 2, 3, 4, 5}

theorem find_set_N (M N : Set ℕ) 
  (h1 : U = M ∪ N) 
  (h2 : M ∩ (U \ N) = {2, 4}) : 
  N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_find_set_N_l2565_256535


namespace NUMINAMATH_CALUDE_t_range_for_strictly_decreasing_function_l2565_256598

theorem t_range_for_strictly_decreasing_function 
  (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y < f x) :
  ∀ t : ℝ, f (t^2) - f t < 0 → t < 0 ∨ t > 1 :=
by sorry

end NUMINAMATH_CALUDE_t_range_for_strictly_decreasing_function_l2565_256598


namespace NUMINAMATH_CALUDE_tower_levels_l2565_256542

theorem tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) :
  steps_per_level = 8 →
  blocks_per_step = 3 →
  total_blocks = 96 →
  total_blocks / (steps_per_level * blocks_per_step) = 4 :=
by sorry

end NUMINAMATH_CALUDE_tower_levels_l2565_256542


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2565_256599

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (P Q : ℝ × ℝ) (l : Line) : Prop := sorry

/-- A point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem circle_symmetry_line (C : Circle) (P Q : ℝ × ℝ) (m : ℝ) :
  C.center = (-1, 3) →
  C.radius = 3 →
  on_circle P C →
  on_circle Q C →
  symmetric_points P Q (Line.mk 1 m 4) →
  m = -1 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l2565_256599


namespace NUMINAMATH_CALUDE_bucket3_most_efficient_bucket3_count_verification_l2565_256533

-- Define the tank capacities
def tank1_capacity : ℕ := 20000
def tank2_capacity : ℕ := 25000
def tank3_capacity : ℕ := 30000

-- Define the bucket capacities
def bucket1_capacity : ℕ := 13
def bucket2_capacity : ℕ := 28
def bucket3_capacity : ℕ := 36

-- Function to calculate the number of buckets needed
def buckets_needed (tank_capacity bucket_capacity : ℕ) : ℕ :=
  (tank_capacity + bucket_capacity - 1) / bucket_capacity

-- Theorem stating that the 36-litre bucket is most efficient for all tanks
theorem bucket3_most_efficient :
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket1_capacity) ∧
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket2_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket1_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket2_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket1_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket2_capacity) :=
by sorry

-- Verify the exact number of 36-litre buckets needed for each tank
theorem bucket3_count_verification :
  (buckets_needed tank1_capacity bucket3_capacity = 556) ∧
  (buckets_needed tank2_capacity bucket3_capacity = 695) ∧
  (buckets_needed tank3_capacity bucket3_capacity = 834) :=
by sorry

end NUMINAMATH_CALUDE_bucket3_most_efficient_bucket3_count_verification_l2565_256533


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_5_l2565_256563

theorem units_digit_of_7_to_5 : 7^5 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_5_l2565_256563


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2565_256512

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → (n.choose k) ≤ (n.choose 4)) →
  (∃ k : ℕ, (8 : ℝ) - (4 * k) / 3 = 0) →
  (∃ c : ℝ, c = (n.choose 6) * (1/2)^2 * (-1)^6) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2565_256512


namespace NUMINAMATH_CALUDE_expected_reflections_l2565_256584

open Real

/-- The expected number of reflections for a billiard ball on a rectangular table -/
theorem expected_reflections (table_length table_width ball_travel : ℝ) :
  table_length = 3 →
  table_width = 1 →
  ball_travel = 2 →
  ∃ (expected_reflections : ℝ),
    expected_reflections = (2 / π) * (3 * arccos (1/4) - arcsin (3/4) + arccos (3/4)) := by
  sorry

end NUMINAMATH_CALUDE_expected_reflections_l2565_256584


namespace NUMINAMATH_CALUDE_age_problem_l2565_256571

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 17 →
  b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2565_256571


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2565_256547

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.im z = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2565_256547


namespace NUMINAMATH_CALUDE_smallest_floor_sum_l2565_256530

theorem smallest_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y + z) / x⌋ + ⌊(x + y + z) / y⌋ + ⌊(x + y + z) / z⌋ ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_floor_sum_l2565_256530


namespace NUMINAMATH_CALUDE_trig_identity_l2565_256534

theorem trig_identity : 
  Real.sin (40 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (220 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l2565_256534


namespace NUMINAMATH_CALUDE_fraction_equality_l2565_256596

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (2 * a) / (2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2565_256596


namespace NUMINAMATH_CALUDE_sum_m_n_equals_three_l2565_256573

theorem sum_m_n_equals_three (m n : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_three_l2565_256573


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2565_256536

theorem angle_sum_is_pi_over_two (α β : Real) : 
  (0 < α ∧ α < π / 2) →  -- α is acute
  (0 < β ∧ β < π / 2) →  -- β is acute
  3 * (Real.sin α) ^ 2 + 2 * (Real.sin β) ^ 2 = 1 →
  3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0 →
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l2565_256536


namespace NUMINAMATH_CALUDE_item_list_price_l2565_256514

theorem item_list_price (list_price : ℝ) : 
  (0.10 * (list_price - 15) = 0.15 * (list_price - 20)) → list_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_item_list_price_l2565_256514


namespace NUMINAMATH_CALUDE_third_chest_silver_excess_l2565_256549

/-- Represents the number of coins in each chest -/
structure ChestContents where
  gold : ℕ
  silver : ℕ

/-- Problem setup -/
def coin_problem (chest1 chest2 chest3 : ChestContents) : Prop :=
  let total_gold := chest1.gold + chest2.gold + chest3.gold
  let total_silver := chest1.silver + chest2.silver + chest3.silver
  total_gold = 40 ∧
  total_silver = 40 ∧
  chest1.gold = chest1.silver + 7 ∧
  chest2.gold = chest2.silver + 15

/-- Theorem statement -/
theorem third_chest_silver_excess 
  (chest1 chest2 chest3 : ChestContents) 
  (h : coin_problem chest1 chest2 chest3) : 
  chest3.silver = chest3.gold + 22 := by
  sorry

#check third_chest_silver_excess

end NUMINAMATH_CALUDE_third_chest_silver_excess_l2565_256549


namespace NUMINAMATH_CALUDE_phi_value_l2565_256566

open Real

noncomputable def f (x φ : ℝ) : ℝ := sin (Real.sqrt 3 * x + φ)

noncomputable def f_deriv (x φ : ℝ) : ℝ := Real.sqrt 3 * cos (Real.sqrt 3 * x + φ)

theorem phi_value (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, f x φ + f_deriv x φ = -(f (-x) φ + f_deriv (-x) φ)) : 
  φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l2565_256566


namespace NUMINAMATH_CALUDE_target_cube_volume_l2565_256525

-- Define the volume of the reference cube
def reference_volume : ℝ := 8

-- Define the surface area of the target cube in terms of the reference cube
def target_surface_area (reference_side : ℝ) : ℝ := 2 * (6 * reference_side^2)

-- Define the volume of a cube given its side length
def cube_volume (side : ℝ) : ℝ := side^3

-- Define the surface area of a cube given its side length
def cube_surface_area (side : ℝ) : ℝ := 6 * side^2

-- Theorem statement
theorem target_cube_volume :
  ∃ (target_side : ℝ),
    cube_surface_area target_side = target_surface_area (reference_volume^(1/3)) ∧
    cube_volume target_side = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_target_cube_volume_l2565_256525


namespace NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l2565_256532

/-- The perimeter of a figure formed by cutting a square into two equal rectangles and placing them side by side -/
theorem perimeter_of_cut_square (side_length : ℝ) : 
  side_length > 0 → 
  (3 * side_length + 4 * (side_length / 2)) = 5 * side_length := by
  sorry

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles and placing them side by side is 500 -/
theorem perimeter_of_specific_cut_square : 
  (3 * 100 + 4 * (100 / 2)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l2565_256532


namespace NUMINAMATH_CALUDE_complex_cube_plus_one_in_first_quadrant_l2565_256597

theorem complex_cube_plus_one_in_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_plus_one_in_first_quadrant_l2565_256597


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2565_256585

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = -52) : x + 1/x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2565_256585


namespace NUMINAMATH_CALUDE_linear_function_decreasing_implies_negative_slope_l2565_256586

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Property that y decreases as x increases -/
def decreasing (f : LinearFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f.m * x₁ + f.b > f.m * x₂ + f.b

theorem linear_function_decreasing_implies_negative_slope (f : LinearFunction) 
    (h : f.b = 5) (dec : decreasing f) : f.m < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_implies_negative_slope_l2565_256586


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2379_l2565_256570

theorem smallest_prime_factor_of_2379 : Nat.minFac 2379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2379_l2565_256570


namespace NUMINAMATH_CALUDE_cosine_intersection_theorem_l2565_256587

theorem cosine_intersection_theorem (f : ℝ → ℝ) (θ : ℝ) : 
  (∀ x ≥ 0, f x = |Real.cos x|) →
  (∃ l : ℝ → ℝ, l 0 = 0 ∧ (∃ a b c d : ℝ, 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d = θ ∧
    f a = l a ∧ f b = l b ∧ f c = l c ∧ f d = l d ∧
    ∀ x : ℝ, x ≥ 0 → x ≠ a → x ≠ b → x ≠ c → x ≠ d → f x ≠ l x)) →
  ((1 + θ^2) * Real.sin (2*θ)) / θ = -2 := by
sorry

end NUMINAMATH_CALUDE_cosine_intersection_theorem_l2565_256587


namespace NUMINAMATH_CALUDE_tangent_length_to_circle_l2565_256569

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_to_circle (x y : ℝ) : 
  let p : ℝ × ℝ := (2, 3)
  let center : ℝ × ℝ := (1, 1)
  let radius : ℝ := 1
  let dist_squared : ℝ := (p.1 - center.1)^2 + (p.2 - center.2)^2
  (x - 1)^2 + (y - 1)^2 = 1 →  -- Circle equation
  dist_squared > radius^2 →    -- P is outside the circle
  Real.sqrt (dist_squared - radius^2) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_length_to_circle_l2565_256569


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l2565_256576

theorem max_value_expression (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x ≤ Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

theorem max_value_achievable : 
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
  3 * x * y * Real.sqrt 5 + 6 * y * z * Real.sqrt 3 + 9 * z * x = Real.sqrt 5 + 3 * Real.sqrt 3 + 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l2565_256576


namespace NUMINAMATH_CALUDE_helmet_store_theorem_l2565_256579

/-- Represents the sales data for a single day -/
structure DailySales where
  helmetA : ℕ
  helmetB : ℕ
  totalAmount : ℕ

/-- Represents the helmet store problem -/
structure HelmetStore where
  day1 : DailySales
  day2 : DailySales
  costPriceA : ℕ
  costPriceB : ℕ
  totalHelmets : ℕ
  budget : ℕ
  profitGoal : ℕ

/-- The main theorem for the helmet store problem -/
theorem helmet_store_theorem (store : HelmetStore)
  (h1 : store.day1 = ⟨10, 15, 1150⟩)
  (h2 : store.day2 = ⟨6, 12, 810⟩)
  (h3 : store.costPriceA = 40)
  (h4 : store.costPriceB = 30)
  (h5 : store.totalHelmets = 100)
  (h6 : store.budget = 3400)
  (h7 : store.profitGoal = 1300) :
  ∃ (priceA priceB maxA : ℕ),
    priceA = 55 ∧
    priceB = 40 ∧
    maxA = 40 ∧
    ¬∃ (numA : ℕ), numA ≤ maxA ∧ 
      (priceA - store.costPriceA) * numA + 
      (priceB - store.costPriceB) * (store.totalHelmets - numA) ≥ store.profitGoal :=
sorry

end NUMINAMATH_CALUDE_helmet_store_theorem_l2565_256579
