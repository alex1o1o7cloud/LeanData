import Mathlib

namespace NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_three_to_one_l611_61136

/-- The number of episodes watched on a weekday -/
def weekday_episodes : ℕ := 8

/-- The total number of episodes watched in a week -/
def total_episodes : ℕ := 88

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The ratio of episodes watched on a weekend day to episodes watched on a weekday -/
def weekend_to_weekday_ratio : ℚ :=
  (total_episodes - weekday_episodes * weekdays) / (weekend_days * weekday_episodes)

theorem weekend_to_weekday_ratio_is_three_to_one :
  weekend_to_weekday_ratio = 3 := by sorry

end NUMINAMATH_CALUDE_weekend_to_weekday_ratio_is_three_to_one_l611_61136


namespace NUMINAMATH_CALUDE_admission_score_calculation_l611_61174

theorem admission_score_calculation (total_applicants : ℕ) 
  (admitted_ratio : ℚ) 
  (admitted_avg_diff : ℝ) 
  (not_admitted_avg_diff : ℝ) 
  (total_avg_score : ℝ) 
  (h1 : admitted_ratio = 1 / 4)
  (h2 : admitted_avg_diff = 10)
  (h3 : not_admitted_avg_diff = -26)
  (h4 : total_avg_score = 70) :
  ∃ (admission_score : ℝ),
    admission_score = 87 ∧
    (admitted_ratio * (admission_score + admitted_avg_diff) + 
     (1 - admitted_ratio) * (admission_score + not_admitted_avg_diff) = total_avg_score) := by
  sorry

end NUMINAMATH_CALUDE_admission_score_calculation_l611_61174


namespace NUMINAMATH_CALUDE_floor_product_eq_sum_iff_in_solution_set_l611_61104

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The solution set for the equation [x] · [y] = x + y -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 = 2) ∨ (2 ≤ p.1 ∧ p.1 < 4 ∧ p.1 ≠ 3 ∧ p.2 = 6 - p.1)}

theorem floor_product_eq_sum_iff_in_solution_set (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (floor x) * (floor y) = x + y ↔ (x, y) ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_floor_product_eq_sum_iff_in_solution_set_l611_61104


namespace NUMINAMATH_CALUDE_sams_age_five_years_ago_l611_61187

/-- Proves Sam's age 5 years ago given the conditions about John, Sam, and Ted's ages --/
theorem sams_age_five_years_ago (sam_current_age : ℕ) : 
  -- John is 3 times as old as Sam
  (3 * sam_current_age = 3 * sam_current_age) →
  -- In 15 years, John will be twice as old as Sam
  (3 * sam_current_age + 15 = 2 * (sam_current_age + 15)) →
  -- Ted is 5 years younger than Sam
  (sam_current_age - 5 = sam_current_age - 5) →
  -- In 15 years, Ted will be three-fourths the age of Sam
  ((sam_current_age - 5 + 15) * 4 = (sam_current_age + 15) * 3) →
  -- Sam's age 5 years ago was 10
  sam_current_age - 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sams_age_five_years_ago_l611_61187


namespace NUMINAMATH_CALUDE_tangent_line_at_two_range_of_m_for_three_roots_l611_61161

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_two :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 := by sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + m = 0 ∧ f y + m = 0 ∧ f z + m = 0) ↔ 
  -3 < m ∧ m < -2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_two_range_of_m_for_three_roots_l611_61161


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l611_61101

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_condition : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l611_61101


namespace NUMINAMATH_CALUDE_double_symmetry_quadratic_l611_61117

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    this function returns the quadratic function 
    that results from applying y-axis symmetry 
    followed by x-axis symmetry -/
def double_symmetry (a b c : ℝ) : ℝ → ℝ := 
  fun x => -a * x^2 + b * x - c

/-- Theorem stating that the double symmetry operation 
    on a quadratic function results in the expected 
    transformed function -/
theorem double_symmetry_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  ∀ x, double_symmetry a b c x = -(a * x^2 + b * x + c) :=
by
  sorry

#check double_symmetry_quadratic

end NUMINAMATH_CALUDE_double_symmetry_quadratic_l611_61117


namespace NUMINAMATH_CALUDE_ordinary_time_rate_l611_61198

/-- Calculates the ordinary time rate given total hours, overtime hours, overtime rate, and total earnings -/
theorem ordinary_time_rate 
  (total_hours : ℕ) 
  (overtime_hours : ℕ) 
  (overtime_rate : ℚ) 
  (total_earnings : ℚ) 
  (h1 : total_hours = 50)
  (h2 : overtime_hours = 8)
  (h3 : overtime_rate = 9/10)
  (h4 : total_earnings = 1620/50)
  (h5 : overtime_hours ≤ total_hours) :
  let ordinary_hours := total_hours - overtime_hours
  let ordinary_rate := (total_earnings - overtime_rate * overtime_hours) / ordinary_hours
  ordinary_rate = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ordinary_time_rate_l611_61198


namespace NUMINAMATH_CALUDE_lana_morning_muffins_l611_61126

/-- Proves that Lana sold 12 muffins in the morning given the conditions of the bake sale -/
theorem lana_morning_muffins (total_goal : ℕ) (afternoon_sales : ℕ) (remaining : ℕ) 
  (h1 : total_goal = 20)
  (h2 : afternoon_sales = 4)
  (h3 : remaining = 4) :
  total_goal = afternoon_sales + remaining + 12 := by
  sorry

end NUMINAMATH_CALUDE_lana_morning_muffins_l611_61126


namespace NUMINAMATH_CALUDE_sum_even_coefficients_l611_61181

theorem sum_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₂ + a₄ + a₆ + a₈ = -24 := by
sorry

end NUMINAMATH_CALUDE_sum_even_coefficients_l611_61181


namespace NUMINAMATH_CALUDE_system_solution_l611_61124

theorem system_solution : ∃ (x y : ℝ), 
  x = 2 ∧ y = -1 ∧ 2*x + y = 3 ∧ -x - y = -1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l611_61124


namespace NUMINAMATH_CALUDE_veena_bill_fraction_l611_61148

theorem veena_bill_fraction :
  ∀ (L V A : ℚ),
  V = (1/2) * L →
  A = (3/4) * V →
  V / (L + V + A) = 4/15 := by
sorry

end NUMINAMATH_CALUDE_veena_bill_fraction_l611_61148


namespace NUMINAMATH_CALUDE_smallest_valid_arrangement_l611_61141

def is_valid_arrangement (n : ℕ) : Prop :=
  (12 ∣ n) ∧ 
  (Finset.card (Finset.filter (λ d => d ∣ n) (Finset.range (n + 1))) = 13) ∧
  (∀ k : ℕ, 1 ≤ k → k ≤ 13 → ∃ m : ℕ, k ≤ m ∧ m < n ∧ m ∣ n)

theorem smallest_valid_arrangement :
  ∃ n : ℕ, is_valid_arrangement n ∧ ∀ m : ℕ, m < n → ¬is_valid_arrangement m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_arrangement_l611_61141


namespace NUMINAMATH_CALUDE_family_age_average_l611_61180

/-- Given the ages of four family members with specific relationships, 
    prove that their average age is 31.5 years. -/
theorem family_age_average (devin_age eden_age mom_age grandfather_age : ℕ) :
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  grandfather_age = (devin_age + eden_age + mom_age) / 2 →
  (devin_age + eden_age + mom_age + grandfather_age : ℚ) / 4 = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_family_age_average_l611_61180


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l611_61134

/-- Given a point P with coordinates (-1, 2), its symmetric point about the x-axis has coordinates (-1, -2) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (-1, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l611_61134


namespace NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l611_61163

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in the grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Represents the minimum number of white cell pairs in the grid -/
def min_white_pairs (g : Grid) : Nat :=
  total_pairs g - (g.black_cells + 40)

/-- Theorem stating the minimum number of white cell pairs in an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  ∀ (g : Grid), g.size = 8 → g.black_cells = 20 → min_white_pairs g = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l611_61163


namespace NUMINAMATH_CALUDE_johnson_carter_tie_l611_61189

/-- Represents the months of the baseball season --/
inductive Month
| March
| April
| May
| June
| July
| August

/-- Represents a player's home run data --/
structure PlayerData where
  monthly_hrs : Month → ℕ

def johnson_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 2
    | Month.April => 12
    | Month.May => 18
    | Month.June => 0
    | Month.July => 0
    | Month.August => 12⟩

def carter_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 0
    | Month.April => 4
    | Month.May => 8
    | Month.June => 22
    | Month.July => 10
    | Month.August => 0⟩

def total_hrs (player : PlayerData) : ℕ :=
  (player.monthly_hrs Month.March) +
  (player.monthly_hrs Month.April) +
  (player.monthly_hrs Month.May) +
  (player.monthly_hrs Month.June) +
  (player.monthly_hrs Month.July) +
  (player.monthly_hrs Month.August)

theorem johnson_carter_tie :
  total_hrs johnson_data = total_hrs carter_data :=
by sorry

end NUMINAMATH_CALUDE_johnson_carter_tie_l611_61189


namespace NUMINAMATH_CALUDE_central_cell_value_l611_61109

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_products : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_products : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_products : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l611_61109


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l611_61115

-- Define the equation of motion
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l611_61115


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l611_61125

/-- Given a triangle and a trapezoid with the same height, prove that the median of the trapezoid is 18 inches when the triangle's base is 36 inches and their areas are equal. -/
theorem trapezoid_median_length (h : ℝ) (h_pos : h > 0) : 
  let triangle_base : ℝ := 36
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let trapezoid_median : ℝ := triangle_area / h
  trapezoid_median = 18 := by sorry

end NUMINAMATH_CALUDE_trapezoid_median_length_l611_61125


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l611_61185

theorem similar_triangles_leg_sum (area_small area_large hyp_small : ℝ) 
  (h1 : area_small = 10)
  (h2 : area_large = 250)
  (h3 : hyp_small = 13)
  (h4 : area_small > 0)
  (h5 : area_large > 0)
  (h6 : hyp_small > 0) :
  ∃ (leg1_small leg2_small leg1_large leg2_large : ℝ),
    leg1_small^2 + leg2_small^2 = hyp_small^2 ∧
    leg1_small * leg2_small / 2 = area_small ∧
    leg1_large^2 + leg2_large^2 = (hyp_small * (area_large / area_small).sqrt)^2 ∧
    leg1_large * leg2_large / 2 = area_large ∧
    leg1_large + leg2_large = 35 := by
sorry


end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l611_61185


namespace NUMINAMATH_CALUDE_digit_sum_problem_l611_61133

/-- Given six unique digits from 2 to 7, prove that if their sums along specific lines total 66, then B must be 4. -/
theorem digit_sum_problem (A B C D E F : ℕ) : 
  A ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  B ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  C ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  D ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  E ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  F ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  (A + B + C) + (A + B + E + F) + (C + D + E) + (B + D + F) + (C + F) = 66 →
  B = 4 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l611_61133


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l611_61175

theorem arithmetic_calculation : 72 / (6 / 2) * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l611_61175


namespace NUMINAMATH_CALUDE_workers_combined_rate_l611_61176

/-- The fraction of a job two workers can complete together in one day, 
    given their individual completion times. -/
def combined_work_rate (time_a time_b : ℚ) : ℚ :=
  1 / time_a + 1 / time_b

/-- Theorem: Two workers, where one takes 18 days and the other takes half that time,
    can complete 1/6 of the job in one day when working together. -/
theorem workers_combined_rate : 
  combined_work_rate 18 9 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_workers_combined_rate_l611_61176


namespace NUMINAMATH_CALUDE_distinct_permutations_l611_61177

def word1 := "NONNA"
def word2 := "MATHEMATICS"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· == c) |>.length

theorem distinct_permutations :
  (Nat.factorial 5 / Nat.factorial (count_letter word1 'N')) = 20 ∧
  (Nat.factorial 10 / (Nat.factorial (count_letter word2 'M') *
                       Nat.factorial (count_letter word2 'A') *
                       Nat.factorial (count_letter word2 'T'))) = 151200 := by
  sorry

#check distinct_permutations

end NUMINAMATH_CALUDE_distinct_permutations_l611_61177


namespace NUMINAMATH_CALUDE_min_value_of_M_l611_61129

theorem min_value_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  M ≥ 8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧
    (1/a₀ - 1) * (1/b₀ - 1) * (1/c₀ - 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l611_61129


namespace NUMINAMATH_CALUDE_even_function_sum_l611_61152

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_l611_61152


namespace NUMINAMATH_CALUDE_wire_length_ratio_l611_61102

theorem wire_length_ratio (edge_length : ℕ) (wire_pieces : ℕ) (wire_length : ℕ) : 
  edge_length = wire_length ∧ wire_pieces = 12 →
  (wire_pieces * wire_length) / (edge_length^3 * 12) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l611_61102


namespace NUMINAMATH_CALUDE_ash_cloud_ratio_l611_61158

/-- Given a volcanic eruption where ashes are shot into the sky, this theorem proves
    the ratio of the ash cloud's diameter to the eruption height. -/
theorem ash_cloud_ratio (eruption_height : ℝ) (cloud_radius : ℝ) 
    (h1 : eruption_height = 300)
    (h2 : cloud_radius = 2700) : 
    (2 * cloud_radius) / eruption_height = 18 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_ratio_l611_61158


namespace NUMINAMATH_CALUDE_expected_value_unfair_coin_l611_61140

/-- The expected value of an unfair coin flip -/
theorem expected_value_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := -9
  p_heads * gain_heads + p_tails * loss_tails = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_unfair_coin_l611_61140


namespace NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l611_61138

/-- Two lines in the form y = kx + l are parallel if and only if 
    they have the same slope but different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept 
  (k₁ k₂ l₁ l₂ : ℝ) : 
  (∀ x y : ℝ, y = k₁ * x + l₁ ↔ y = k₂ * x + l₂) ↔ 
  (k₁ = k₂ ∧ l₁ ≠ l₂) :=
by sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_same_slope_diff_intercept_l611_61138


namespace NUMINAMATH_CALUDE_S_is_closed_closed_set_contains_zero_l611_61119

-- Define a closed set
def ClosedSet (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Define the set S
def S : Set ℝ := {x | ∃ a b : ℤ, x = a + b * Real.sqrt 3}

-- Theorem 1: S is a closed set
theorem S_is_closed : ClosedSet S := sorry

-- Theorem 2: Any closed set contains 0
theorem closed_set_contains_zero (T : Set ℝ) (h : ClosedSet T) : (0 : ℝ) ∈ T := sorry

end NUMINAMATH_CALUDE_S_is_closed_closed_set_contains_zero_l611_61119


namespace NUMINAMATH_CALUDE_range_of_a_l611_61143

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (a ≤ 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l611_61143


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l611_61131

/-- An arithmetic sequence of integers -/
def arithmeticSeq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- An increasing sequence -/
def increasingSeq (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_problem (b : ℕ → ℤ) 
    (h_arith : arithmeticSeq b)
    (h_incr : increasingSeq b)
    (h_prod : b 4 * b 5 = 30) : 
  b 3 * b 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l611_61131


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_equal_orthocenter_quadrilateral_l611_61156

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Definition of an inscribed quadrilateral -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Definition of an orthocenter of a triangle -/
def isOrthocenter (H : Point) (A B C : Point) : Prop :=
  sorry

/-- Definition of equality between quadrilaterals -/
def quadrilateralEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_equal_orthocenter_quadrilateral 
  (A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ : Point) :
  isInscribed (Quadrilateral.mk A₁ A₂ A₃ A₄) →
  isOrthocenter H₁ A₂ A₃ A₄ →
  isOrthocenter H₂ A₁ A₃ A₄ →
  isOrthocenter H₃ A₁ A₂ A₄ →
  isOrthocenter H₄ A₁ A₂ A₃ →
  quadrilateralEqual 
    (Quadrilateral.mk A₁ A₂ A₃ A₄) 
    (Quadrilateral.mk H₁ H₂ H₃ H₄) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_equal_orthocenter_quadrilateral_l611_61156


namespace NUMINAMATH_CALUDE_sum_of_four_solution_values_l611_61188

-- Define the polynomial function f
noncomputable def f (x : ℝ) : ℝ := 
  (x - 5) * (x - 3) * (x - 1) * (x + 1) * (x + 3) * (x + 5) / 315 - 3.4

-- Define the property of having exactly 4 solutions
def has_four_solutions (c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (f x₁ = c ∧ f x₂ = c ∧ f x₃ = c ∧ f x₄ = c) ∧
    (∀ x, f x = c → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Theorem statement
theorem sum_of_four_solution_values :
  ∃ (a b : ℤ), has_four_solutions (a : ℝ) ∧ has_four_solutions (b : ℝ) ∧ a + b = -7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_solution_values_l611_61188


namespace NUMINAMATH_CALUDE_goldfish_problem_l611_61199

theorem goldfish_problem (initial : ℕ) (final : ℕ) (birds : ℕ) (disease : ℕ) 
  (h1 : initial = 240)
  (h2 : final = 45)
  (h3 : birds = 15)
  (h4 : disease = 30) :
  let vanished := initial - final
  let heat := (vanished * 20) / 100
  let eaten := vanished - heat - disease - birds
  let raccoons := eaten / 3
  let cats := raccoons * 2
  cats = 64 ∧ raccoons = 32 ∧ heat = 39 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_problem_l611_61199


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l611_61182

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l611_61182


namespace NUMINAMATH_CALUDE_min_squared_distance_to_origin_l611_61171

/-- The minimum squared distance from a point on the line 2x + y + 5 = 0 to the origin is 5 -/
theorem min_squared_distance_to_origin : 
  ∀ x y : ℝ, 2 * x + y + 5 = 0 → x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_squared_distance_to_origin_l611_61171


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l611_61137

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem
theorem perpendicular_line_equation (A B C : ℝ) (P₀ : Point2D) :
  let L₁ : Line := { a := A, b := B, c := C }
  let L₂ : Line := { a := B, b := -A, c := -B * P₀.x + A * P₀.y }
  (∀ (x y : ℝ), A * x + B * y + C = 0 → L₁.a * x + L₁.b * y + L₁.c = 0) →
  (L₂.a * P₀.x + L₂.b * P₀.y + L₂.c = 0) →
  (∀ (x y : ℝ), B * x - A * y - B * P₀.x + A * P₀.y = 0 ↔ L₂.a * x + L₂.b * y + L₂.c = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l611_61137


namespace NUMINAMATH_CALUDE_divisibility_by_3_and_2_l611_61151

theorem divisibility_by_3_and_2 (n : ℕ) : 
  (3 ∣ n) → (2 ∣ n) → (6 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_3_and_2_l611_61151


namespace NUMINAMATH_CALUDE_sam_seashells_l611_61118

def seashells_problem (yesterday_found : ℕ) (given_to_joan : ℕ) (today_found : ℕ) (given_to_tom : ℕ) : ℕ :=
  yesterday_found - given_to_joan + today_found - given_to_tom

theorem sam_seashells : seashells_problem 35 18 20 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l611_61118


namespace NUMINAMATH_CALUDE_scientific_notation_of_71300000_l611_61130

theorem scientific_notation_of_71300000 :
  ∃ (a : ℝ) (n : ℤ), 71300000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7.13 ∧ n = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_71300000_l611_61130


namespace NUMINAMATH_CALUDE_trajectory_of_point_on_moving_segment_l611_61184

/-- The trajectory of a point M on a moving line segment AB -/
theorem trajectory_of_point_on_moving_segment (A B M : ℝ × ℝ) 
  (h_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
  (h_A_on_x : A.2 = 0)
  (h_B_on_y : B.1 = 0)
  (h_M_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B)
  (h_ratio : ∃ k : ℝ, k > 0 ∧ 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = k^2 * ((B.1 - M.1)^2 + (B.2 - M.2)^2) ∧
    k = 1/2) :
  9 * M.1^2 + 36 * M.2^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_on_moving_segment_l611_61184


namespace NUMINAMATH_CALUDE_inequality_solution_set_l611_61128

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) → 
  ((3 * a) / (3 - 1) = 1) → 
  a - b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l611_61128


namespace NUMINAMATH_CALUDE_cube_volume_from_lateral_surface_area_l611_61154

theorem cube_volume_from_lateral_surface_area :
  ∀ (lateral_surface_area : ℝ) (volume : ℝ),
  lateral_surface_area = 100 →
  volume = (lateral_surface_area / 4) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_lateral_surface_area_l611_61154


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l611_61196

theorem smaller_solution_quadratic_equation :
  ∃ (x y : ℝ), x < y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧
  ∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y ∧
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l611_61196


namespace NUMINAMATH_CALUDE_order_of_abc_l611_61170

theorem order_of_abc (a b c : ℝ) : 
  a = 2 * Real.log 1.01 →
  b = Real.log 1.02 →
  c = Real.sqrt 1.04 - 1 →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_of_abc_l611_61170


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l611_61116

theorem sum_remainder_by_eight (n : ℤ) : (8 - n + (n + 5)) % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l611_61116


namespace NUMINAMATH_CALUDE_problem_statement_l611_61183

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

theorem problem_statement (a : ℝ) :
  f a (Real.log (Real.log 5 / Real.log 2)) = 5 →
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l611_61183


namespace NUMINAMATH_CALUDE_one_carton_per_case_l611_61155

/-- The number of cartons in a case -/
def cartons_per_case : ℕ := 1

/-- The number of boxes in each carton -/
def boxes_per_carton : ℕ := 1

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 500

/-- The total number of paper clips in two cases -/
def total_clips : ℕ := 1000

/-- Theorem stating that there is exactly one carton in a case -/
theorem one_carton_per_case :
  (∀ b : ℕ, b > 0 → 2 * cartons_per_case * b * clips_per_box = total_clips) →
  cartons_per_case = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_carton_per_case_l611_61155


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l611_61113

open Set

def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l611_61113


namespace NUMINAMATH_CALUDE_square_inequality_l611_61135

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l611_61135


namespace NUMINAMATH_CALUDE_deepak_investment_l611_61122

/-- Proves that Deepak's investment is 15000 given the conditions of the business problem -/
theorem deepak_investment (total_profit : ℝ) (anand_investment : ℝ) (deepak_profit : ℝ) 
  (h1 : total_profit = 13800)
  (h2 : anand_investment = 22500)
  (h3 : deepak_profit = 5400) :
  ∃ deepak_investment : ℝ, 
    deepak_investment = 15000 ∧ 
    deepak_profit / total_profit = deepak_investment / (anand_investment + deepak_investment) :=
by
  sorry


end NUMINAMATH_CALUDE_deepak_investment_l611_61122


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_15000_l611_61120

theorem last_four_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1250]) :
  5^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_to_15000_l611_61120


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l611_61190

def is_valid_sequence (seq : List Nat) : Prop :=
  (∀ i, i + 2 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]!) % 2 = 0) ∧
  (∀ i, i + 3 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]! + seq[i+3]!) % 2 = 1)

theorem max_valid_sequence_length :
  (∃ (seq : List Nat), is_valid_sequence seq ∧ seq.length = 5) ∧
  (∀ (seq : List Nat), is_valid_sequence seq → seq.length ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l611_61190


namespace NUMINAMATH_CALUDE_invariant_preserved_cannot_transform_l611_61149

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- A 4x4 matrix of letters -/
def LetterMatrix := Matrix (Fin 4) (Fin 4) Letter

/-- The operation of incrementing a letter (with wrapping) -/
def nextLetter (l : Letter) : Letter :=
  ⟨(l.val + 1) % 26, by sorry⟩

/-- The invariant property for a 2x2 submatrix -/
def invariant (a b c d : Letter) : ℤ :=
  (a.val + d.val : ℤ) - (b.val + c.val : ℤ)

/-- Theorem: The invariant is preserved under row and column operations -/
theorem invariant_preserved (a b c d : Letter) :
  (invariant a b c d = invariant (nextLetter a) (nextLetter b) c d) ∧
  (invariant a b c d = invariant (nextLetter a) b (nextLetter c) d) :=
sorry

/-- The initial matrix (a) -/
def matrix_a : LetterMatrix := sorry

/-- The target matrix (b) -/
def matrix_b : LetterMatrix := sorry

/-- Theorem: Matrix (a) cannot be transformed into matrix (b) -/
theorem cannot_transform (ops : ℕ) :
  ∀ (m : LetterMatrix), 
    (∃ (i : Fin 4), ∀ (j : Fin 4), m i j = nextLetter (matrix_a i j)) ∨
    (∃ (j : Fin 4), ∀ (i : Fin 4), m i j = nextLetter (matrix_a i j)) →
    m ≠ matrix_b :=
sorry

end NUMINAMATH_CALUDE_invariant_preserved_cannot_transform_l611_61149


namespace NUMINAMATH_CALUDE_odd_numbers_properties_l611_61166

theorem odd_numbers_properties (x y : ℤ) (hx : ∃ k : ℤ, x = 2 * k + 1) (hy : ∃ k : ℤ, y = 2 * k + 1) :
  (∃ m : ℤ, x + y = 2 * m) ∧ 
  (∃ n : ℤ, x - y = 2 * n) ∧ 
  (∃ p : ℤ, x * y = 2 * p + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_properties_l611_61166


namespace NUMINAMATH_CALUDE_success_arrangements_eq_420_l611_61142

/-- The number of letters in the word "SUCCESS" -/
def word_length : ℕ := 7

/-- The number of occurrences of the letter 'S' in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of occurrences of the letter 'C' in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := word_length.factorial / (s_count.factorial * c_count.factorial)

theorem success_arrangements_eq_420 : success_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_eq_420_l611_61142


namespace NUMINAMATH_CALUDE_intersection_properties_y₁_gt_y₂_l611_61112

/-- The quadratic function y₁ -/
def y₁ (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- The linear function y₂ -/
def y₂ (x : ℝ) : ℝ := x + 1

/-- Theorem stating the properties of the intersection points and the resulting quadratic function -/
theorem intersection_properties :
  ∀ b m : ℝ,
  (y₁ b (-1) = y₂ (-1)) →
  (y₁ b 4 = y₂ 4) →
  (y₁ b (-1) = 0) →
  (y₁ b 4 = m) →
  (b = -2 ∧ m = 5) :=
sorry

/-- Theorem stating when y₁ > y₂ -/
theorem y₁_gt_y₂ :
  ∀ x : ℝ,
  (y₁ (-2) x > y₂ x) ↔ (x < -1 ∨ x > 4) :=
sorry

end NUMINAMATH_CALUDE_intersection_properties_y₁_gt_y₂_l611_61112


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l611_61111

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- Predicate to check if a point lies on the circle with diameter between two other points -/
def lies_on_circle_diameter (p q r : ℝ × ℝ) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) : 
  lies_on_circle_diameter (0, h.b) (left_vertex h) (right_focus h) →
  eccentricity h = (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l611_61111


namespace NUMINAMATH_CALUDE_trisha_take_home_pay_l611_61173

/-- Calculates the annual take-home pay for an hourly worker. -/
def annual_take_home_pay (hourly_rate : ℚ) (hours_per_week : ℕ) (weeks_per_year : ℕ) (withholding_rate : ℚ) : ℚ :=
  let gross_pay := hourly_rate * hours_per_week * weeks_per_year
  let withholding := withholding_rate * gross_pay
  gross_pay - withholding

/-- Proves that Trisha's annual take-home pay is $24,960 given the specified conditions. -/
theorem trisha_take_home_pay :
  annual_take_home_pay 15 40 52 (1/5) = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 (1/5)

end NUMINAMATH_CALUDE_trisha_take_home_pay_l611_61173


namespace NUMINAMATH_CALUDE_length_PQ_is_4_l611_61100

-- Define the semicircle (C)
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the polar equation of line (l)
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop :=
  θ = Real.pi / 3

-- Define the point P as the intersection of semicircle (C) and ray OM
def point_P (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ ∧ ray_OM θ

-- Define the point Q as the intersection of line (l) and ray OM
def point_Q (ρ θ : ℝ) : Prop :=
  line_l ρ θ ∧ ray_OM θ

-- Theorem statement
theorem length_PQ_is_4 :
  ∀ (ρ_P θ_P ρ_Q θ_Q : ℝ),
    point_P ρ_P θ_P →
    point_Q ρ_Q θ_Q →
    |ρ_P - ρ_Q| = 4 :=
sorry

end NUMINAMATH_CALUDE_length_PQ_is_4_l611_61100


namespace NUMINAMATH_CALUDE_apple_cost_l611_61192

/-- The cost of apples given specific pricing rules and total costs for certain weights. -/
theorem apple_cost (l q : ℝ) : 
  (∀ x, x ≤ 30 → x * l = x * 0.362) →  -- Cost for first 30 kgs
  (∀ x, x > 30 → x * l + (x - 30) * q = 30 * l + (x - 30) * q) →  -- Cost for additional kgs
  (33 * l + 3 * q = 11.67) →  -- Price for 33 kgs
  (36 * l + 6 * q = 12.48) →  -- Price for 36 kgs
  (10 * l = 3.62) :=  -- Cost of first 10 kgs
by sorry

end NUMINAMATH_CALUDE_apple_cost_l611_61192


namespace NUMINAMATH_CALUDE_total_students_proof_l611_61178

def school_problem (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  let class_sizes := List.range n |>.map (fun i => largest_class - i * diff)
  class_sizes.sum

theorem total_students_proof :
  school_problem 5 25 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_students_proof_l611_61178


namespace NUMINAMATH_CALUDE_simultaneous_sequence_probability_l611_61195

-- Define the probabilities for each coin
def coin_a_heads : ℝ := 0.3
def coin_a_tails : ℝ := 0.7
def coin_b_heads : ℝ := 0.4
def coin_b_tails : ℝ := 0.6

-- Define the number of consecutive flips
def num_flips : ℕ := 6

-- Define the probability of the desired sequence for each coin
def prob_a_sequence : ℝ := coin_a_tails * coin_a_tails * coin_a_heads
def prob_b_sequence : ℝ := coin_b_heads * coin_b_tails * coin_b_tails

-- Theorem to prove
theorem simultaneous_sequence_probability :
  prob_a_sequence * prob_b_sequence = 0.021168 :=
sorry

end NUMINAMATH_CALUDE_simultaneous_sequence_probability_l611_61195


namespace NUMINAMATH_CALUDE_system_solutions_l611_61186

-- Define the logarithm base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y - 20 = 0 ∧ log4 x + log4 y = 1 + log4 9

-- Theorem stating the solutions
theorem system_solutions :
  ∃ (x y : ℝ), system x y ∧ ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l611_61186


namespace NUMINAMATH_CALUDE_equation_solutions_l611_61127

theorem equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l611_61127


namespace NUMINAMATH_CALUDE_exam_score_calculation_l611_61168

theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_score : ℕ) 
  (wrong_answer_penalty : ℕ) :
  total_questions = 75 →
  correct_answers = 40 →
  total_score = 125 →
  wrong_answer_penalty = 1 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - 
    wrong_answer_penalty * (total_questions - correct_answers) = total_score ∧
    score_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l611_61168


namespace NUMINAMATH_CALUDE_first_fun_friday_is_april_28_l611_61105

/-- Represents a date in a calendar year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Friday -/
def isFriday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given month has five Fridays -/
def hasFiveFridays (month : Nat) (year : Nat) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns the date of the first Fun Friday after the given start date -/
def firstFunFriday (startDate : Date) (startDay : DayOfWeek) : Date :=
  sorry

theorem first_fun_friday_is_april_28 :
  let fiscalYearStart : Date := ⟨3, 1⟩
  let fiscalYearStartDay : DayOfWeek := DayOfWeek.Wednesday
  firstFunFriday fiscalYearStart fiscalYearStartDay = ⟨4, 28⟩ := by
  sorry

end NUMINAMATH_CALUDE_first_fun_friday_is_april_28_l611_61105


namespace NUMINAMATH_CALUDE_unique_intersection_l611_61159

/-- The coefficient of x^2 in the quadratic equation -/
def b : ℚ := 49 / 16

/-- The quadratic function -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function -/
def g (x : ℝ) : ℝ := -2 * x - 2

/-- The difference between the quadratic and linear functions -/
def h (x : ℝ) : ℝ := f x - g x

theorem unique_intersection :
  ∃! x, h x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l611_61159


namespace NUMINAMATH_CALUDE_popped_kernels_problem_l611_61179

theorem popped_kernels_problem (bag1_popped bag1_total bag2_popped bag2_total bag3_popped : ℕ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_popped = 82)
  (h6 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + (bag3_popped : ℚ) / bag3_total = 82 * 3 / 100) :
  bag3_total = 100 := by
  sorry

end NUMINAMATH_CALUDE_popped_kernels_problem_l611_61179


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l611_61123

theorem matrix_equation_proof :
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![16/7, -36/7; -12/7, 27/7]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 5; 16, -4]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; -4, 1]
  N * A = B + C := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l611_61123


namespace NUMINAMATH_CALUDE_imoProof_l611_61147

theorem imoProof (d : ℕ) (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) (h4 : d > 0) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end NUMINAMATH_CALUDE_imoProof_l611_61147


namespace NUMINAMATH_CALUDE_amys_tickets_proof_l611_61160

/-- The total number of tickets Amy has after buying more at the fair -/
def amys_total_tickets (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Amy's total tickets is 54 given her initial and additional tickets -/
theorem amys_tickets_proof :
  amys_total_tickets 33 21 = 54 := by
  sorry

end NUMINAMATH_CALUDE_amys_tickets_proof_l611_61160


namespace NUMINAMATH_CALUDE_meaningful_expression_l611_61194

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l611_61194


namespace NUMINAMATH_CALUDE_sum_even_numbers_1_to_200_l611_61191

/- Define the sum of even numbers from 1 to n -/
def sumEvenNumbers (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

/- Theorem statement -/
theorem sum_even_numbers_1_to_200 : sumEvenNumbers 200 = 10100 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_numbers_1_to_200_l611_61191


namespace NUMINAMATH_CALUDE_mildred_oranges_proof_l611_61165

/-- Calculates the remaining oranges after Mildred's father and friend take some. -/
def remaining_oranges (initial : Float) (father_eats : Float) (friend_takes : Float) : Float :=
  initial - father_eats - friend_takes

/-- Proves that Mildred has 71.5 oranges left after her father and friend take some. -/
theorem mildred_oranges_proof (initial : Float) (father_eats : Float) (friend_takes : Float)
    (h1 : initial = 77.5)
    (h2 : father_eats = 2.25)
    (h3 : friend_takes = 3.75) :
    remaining_oranges initial father_eats friend_takes = 71.5 := by
  sorry

end NUMINAMATH_CALUDE_mildred_oranges_proof_l611_61165


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l611_61172

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x + y = x^2 - x*y + y^2 ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = 0 ∧ y = 1) ∨ 
    (x = 1 ∧ y = 0) ∨ 
    (x = 1 ∧ y = 2) ∨ 
    (x = 2 ∧ y = 1) ∨ 
    (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l611_61172


namespace NUMINAMATH_CALUDE_binary_representation_of_37_l611_61108

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 37 -/
def binary37 : List Bool := [true, false, true, false, false, true]

/-- Theorem stating that the binary representation of 37 is [true, false, true, false, false, true] -/
theorem binary_representation_of_37 : toBinary 37 = binary37 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_37_l611_61108


namespace NUMINAMATH_CALUDE_prob_n_minus_one_matches_is_zero_l611_61197

/-- Represents the number of pairs in the matching problem -/
def n : ℕ := 10

/-- Represents a function that returns the probability of correctly matching
    exactly k pairs out of n pairs when choosing randomly -/
noncomputable def probability_exact_matches (k : ℕ) : ℝ := sorry

/-- Theorem stating that the probability of correctly matching exactly n-1 pairs
    out of n pairs is 0 when choosing randomly -/
theorem prob_n_minus_one_matches_is_zero :
  probability_exact_matches (n - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_prob_n_minus_one_matches_is_zero_l611_61197


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_plus_one_l611_61106

theorem divisibility_of_factorial_plus_one (p : ℕ) : 
  (Nat.Prime p → p ∣ (Nat.factorial (p - 1) + 1)) ∧
  (¬Nat.Prime p → ¬(p ∣ (Nat.factorial (p - 1) + 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_plus_one_l611_61106


namespace NUMINAMATH_CALUDE_impossible_to_reach_all_plus_l611_61193

/- Define the sign type -/
inductive Sign : Type
| Plus : Sign
| Minus : Sign

/- Define the 4x4 grid type -/
def Grid := Matrix (Fin 4) (Fin 4) Sign

/- Define the initial grid -/
def initial_grid : Grid :=
  λ i j => match i, j with
  | 0, 1 => Sign.Minus
  | 3, 1 => Sign.Minus
  | _, _ => Sign.Plus

/- Define a move (flipping a row or column) -/
def flip_row (g : Grid) (row : Fin 4) : Grid :=
  λ i j => if i = row then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

def flip_column (g : Grid) (col : Fin 4) : Grid :=
  λ i j => if j = col then
    match g i j with
    | Sign.Plus => Sign.Minus
    | Sign.Minus => Sign.Plus
    else g i j

/- Define the goal state (all Plus signs) -/
def all_plus (g : Grid) : Prop :=
  ∀ i j, g i j = Sign.Plus

/- The main theorem -/
theorem impossible_to_reach_all_plus :
  ¬ ∃ (moves : List (Sum (Fin 4) (Fin 4))),
    all_plus (moves.foldl (λ g move => match move with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col) initial_grid) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_all_plus_l611_61193


namespace NUMINAMATH_CALUDE_multiset_permutations_eq_1680_l611_61145

/-- The number of permutations of a multiset with 9 elements, where there are 3 elements of each of 3 types -/
def multiset_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3)

/-- Theorem stating that the number of permutations of the described multiset is 1680 -/
theorem multiset_permutations_eq_1680 : multiset_permutations = 1680 := by
  sorry

end NUMINAMATH_CALUDE_multiset_permutations_eq_1680_l611_61145


namespace NUMINAMATH_CALUDE_unique_integer_with_16_divisors_l611_61121

def hasSixteenDivisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16

def divisorsOrdered (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 16 → (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe i sorry <
    (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe j sorry

def divisorProperty (n : ℕ) : Prop :=
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  let d₂ := divisors.nthLe 1 sorry
  let d₄ := divisors.nthLe 3 sorry
  let d₅ := divisors.nthLe 4 sorry
  let d₆ := divisors.nthLe 5 sorry
  divisors.nthLe (d₅ - 1) sorry = (d₂ + d₄) * d₆

theorem unique_integer_with_16_divisors :
  ∃! n : ℕ, n > 0 ∧ hasSixteenDivisors n ∧ divisorsOrdered n ∧ divisorProperty n ∧ n = 2002 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_with_16_divisors_l611_61121


namespace NUMINAMATH_CALUDE_volume_integrals_l611_61162

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x^3)

theorem volume_integrals (π : ℝ) (h₁ : π > 0) :
  (∫ (x : ℝ) in Set.Ioi 0, π * (f x)^2) = π / 6 ∧
  (∫ (x : ℝ) in Set.Icc 0 (Real.rpow 3 (1/3)), π * x^2 * (1 - 3*x^3) * Real.exp (-x^3)) = π * (Real.exp (-1/3) - 2/3) :=
sorry

end NUMINAMATH_CALUDE_volume_integrals_l611_61162


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l611_61107

noncomputable def x (θ : Real) : Real := |Real.sin (θ / 2) + Real.cos (θ / 2)|
noncomputable def y (θ : Real) : Real := 1 + Real.sin θ

theorem parametric_to_ordinary_equation :
  ∀ θ : Real, 0 ≤ θ ∧ θ < 2 * Real.pi →
  ∃ x_val y_val : Real,
    x θ = x_val ∧
    y θ = y_val ∧
    x_val ^ 2 = y_val ∧
    0 ≤ x_val ∧ x_val ≤ Real.sqrt 2 ∧
    0 ≤ y_val ∧ y_val ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l611_61107


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l611_61157

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 5 = 64) :
  a 4 = 8 ∨ a 4 = -8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l611_61157


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l611_61164

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-3) 1) :
  b < 0 ∧ c > 0 ∧
  {x : ℝ | a * x - b < 0} = Set.Ioi 2 ∧
  {x : ℝ | a * x^2 - b * x + c < 0} = Set.Iic (-1) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l611_61164


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l611_61169

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^3 = 2) :
  a^4 + 1/a^4 = (Real.rpow 4 (1/3) - 2)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l611_61169


namespace NUMINAMATH_CALUDE_license_plate_combinations_license_plate_count_l611_61110

theorem license_plate_combinations : ℕ :=
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_positions : ℕ := 3
  let digit_positions : ℕ := 4
  letter_choices ^ letter_positions * digit_choices ^ digit_positions

theorem license_plate_count : license_plate_combinations = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_license_plate_count_l611_61110


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l611_61132

theorem power_mod_seventeen : 2^2023 % 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l611_61132


namespace NUMINAMATH_CALUDE_number_of_female_democrats_l611_61153

/-- Given a meeting with male and female participants, prove the number of female Democrats --/
theorem number_of_female_democrats 
  (total_participants : ℕ) 
  (female_participants : ℕ) 
  (male_participants : ℕ) 
  (h1 : total_participants = 780)
  (h2 : female_participants + male_participants = total_participants)
  (h3 : 2 * (total_participants / 3) = female_participants / 2 + male_participants / 4) :
  female_participants / 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_democrats_l611_61153


namespace NUMINAMATH_CALUDE_stating_sum_of_intersections_theorem_l611_61150

/-- The number of lines passing through the origin -/
def num_lines : ℕ := 180

/-- The angle between each line in degrees -/
def angle_between : ℝ := 1

/-- The equation of the line that intersects with all other lines -/
def intersecting_line (x : ℝ) : ℝ := 100 - x

/-- The sum of x-coordinates of intersection points -/
def sum_of_intersections : ℝ := 8950

/-- 
Theorem stating that the sum of x-coordinates of intersections between 
180 lines passing through the origin (forming 1 degree angles) and the 
line y = 100 - x is equal to 8950.
-/
theorem sum_of_intersections_theorem :
  let lines := List.range num_lines
  let intersection_points := lines.map (λ i => 
    let angle := i * angle_between
    let m := Real.tan (angle * π / 180)
    100 / (1 + m))
  intersection_points.sum = sum_of_intersections := by
  sorry


end NUMINAMATH_CALUDE_stating_sum_of_intersections_theorem_l611_61150


namespace NUMINAMATH_CALUDE_total_rainfall_l611_61139

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 15) ∧
  (first_week + second_week = 25)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l611_61139


namespace NUMINAMATH_CALUDE_incorrect_copy_difference_l611_61146

theorem incorrect_copy_difference (square : ℝ) : 
  let x := 4 * (square - 3)
  let y := 4 * square - 3
  x - y = -9 := by sorry

end NUMINAMATH_CALUDE_incorrect_copy_difference_l611_61146


namespace NUMINAMATH_CALUDE_sqrt_2_plus_x_real_range_l611_61167

theorem sqrt_2_plus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 + x) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_plus_x_real_range_l611_61167


namespace NUMINAMATH_CALUDE_book_set_cost_l611_61103

/-- The cost of a book set given lawn mowing parameters -/
theorem book_set_cost 
  (charge_rate : ℚ)
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (lawns_mowed : ℕ)
  (additional_area : ℕ)
  (h1 : charge_rate = 1 / 10)
  (h2 : lawn_length = 20)
  (h3 : lawn_width = 15)
  (h4 : lawns_mowed = 3)
  (h5 : additional_area = 600) :
  (lawn_length * lawn_width * lawns_mowed + additional_area) * charge_rate = 150 := by
  sorry

#check book_set_cost

end NUMINAMATH_CALUDE_book_set_cost_l611_61103


namespace NUMINAMATH_CALUDE_sine_characterization_l611_61114

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem sine_characterization (f : ℝ → ℝ) 
  (h1 : IsPeriodic f π)
  (h2 : IsSymmetricAbout f (π/3))
  (h3 : IsIncreasingOn f (-π/6) (π/3)) :
  ∀ x, f x = Real.sin (2*x - π/6) := by
sorry

end NUMINAMATH_CALUDE_sine_characterization_l611_61114


namespace NUMINAMATH_CALUDE_boys_count_is_sixty_l611_61144

/-- Represents the number of boys in a group of 3 students -/
inductive GroupComposition
  | ThreeGirls
  | TwoGirlsOneBoy
  | OneGirlTwoBoys
  | ThreeBoys

/-- Represents the distribution of groups -/
structure GroupDistribution where
  total_groups : Nat
  one_boy_groups : Nat
  at_least_two_boys_groups : Nat
  three_boys_groups : Nat
  three_girls_groups : Nat

/-- Calculates the total number of boys given a group distribution -/
def count_boys (gd : GroupDistribution) : Nat :=
  gd.one_boy_groups + 2 * (gd.at_least_two_boys_groups - gd.three_boys_groups) + 3 * gd.three_boys_groups

/-- The main theorem to be proved -/
theorem boys_count_is_sixty (gd : GroupDistribution) 
  (h1 : gd.total_groups = 35)
  (h2 : gd.one_boy_groups = 10)
  (h3 : gd.at_least_two_boys_groups = 19)
  (h4 : gd.three_boys_groups = 2 * gd.three_girls_groups)
  (h5 : gd.three_girls_groups = gd.total_groups - gd.one_boy_groups - gd.at_least_two_boys_groups) :
  count_boys gd = 60 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_is_sixty_l611_61144
