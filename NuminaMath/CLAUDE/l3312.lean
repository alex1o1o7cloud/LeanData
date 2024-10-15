import Mathlib

namespace NUMINAMATH_CALUDE_no_valid_stacking_l3312_331238

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of crates -/
def numCrates : ℕ := 12

/-- The dimensions of each crate -/
def crateDims : CrateDimensions := ⟨3, 4, 5⟩

/-- The target total height -/
def targetHeight : ℕ := 50

/-- Theorem stating that there are no valid ways to stack the crates to reach the target height -/
theorem no_valid_stacking :
  ¬∃ (a b c : ℕ), a + b + c = numCrates ∧
                  a * crateDims.length + b * crateDims.width + c * crateDims.height = targetHeight :=
by sorry

end NUMINAMATH_CALUDE_no_valid_stacking_l3312_331238


namespace NUMINAMATH_CALUDE_study_time_for_target_average_l3312_331225

/-- Calculates the number of minutes needed to study on the 12th day to achieve a given average -/
def minutes_to_study_on_last_day (days_30min : ℕ) (days_45min : ℕ) (target_average : ℕ) : ℕ :=
  let total_days := days_30min + days_45min + 1
  let total_minutes_needed := total_days * target_average
  let minutes_already_studied := days_30min * 30 + days_45min * 45
  total_minutes_needed - minutes_already_studied

/-- Theorem stating that given the specific study pattern, 90 minutes are needed on the 12th day -/
theorem study_time_for_target_average :
  minutes_to_study_on_last_day 7 4 40 = 90 := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_target_average_l3312_331225


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331209

-- Define the sets A and B
variable (A B : Set ℤ)

-- Define the function f
def f (x : ℤ) : ℤ := x^2

-- State the theorem
theorem intersection_of_A_and_B :
  (∀ x ∈ A, f x ∈ B) →
  B = {1, 2} →
  (A ∩ B = ∅ ∨ A ∩ B = {1}) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331209


namespace NUMINAMATH_CALUDE_units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l3312_331210

theorem units_digit_of_27_cubed_minus_17_cubed : ℕ → Prop :=
  fun d => (27^3 - 17^3) % 10 = d

theorem units_digit_is_zero :
  units_digit_of_27_cubed_minus_17_cubed 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_cubed_minus_17_cubed_units_digit_is_zero_l3312_331210


namespace NUMINAMATH_CALUDE_system_two_solutions_l3312_331228

/-- The system of equations has exactly two solutions iff a = 4 or a = 100 -/
theorem system_two_solutions (a : ℝ) :
  (∃! (x y : ℝ), |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ∧
  (∃! (x' y' : ℝ), (x', y') ≠ (x, y) ∧ |x' - 6 - y'| + |x' - 6 + y'| = 12 ∧ (|x'| - 6)^2 + (|y'| - 8)^2 = a) ↔
  (a = 4 ∨ a = 100) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l3312_331228


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_fourth_l3312_331253

theorem units_digit_of_six_to_fourth (n : ℕ) : n = 6^4 → n % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_fourth_l3312_331253


namespace NUMINAMATH_CALUDE_product_xy_is_60_l3312_331206

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem product_xy_is_60 (x y : ℝ) :
  line_k x 6 ∧ line_k 10 y → x * y = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_60_l3312_331206


namespace NUMINAMATH_CALUDE_x_eleven_percent_greater_than_90_l3312_331275

theorem x_eleven_percent_greater_than_90 :
  ∀ x : ℝ, x = 90 * (1 + 11 / 100) → x = 99.9 := by
  sorry

end NUMINAMATH_CALUDE_x_eleven_percent_greater_than_90_l3312_331275


namespace NUMINAMATH_CALUDE_combined_work_time_l3312_331256

def worker_a_time : ℝ := 10
def worker_b_time : ℝ := 15

theorem combined_work_time : 
  let combined_rate := (1 / worker_a_time) + (1 / worker_b_time)
  1 / combined_rate = 6 := by sorry

end NUMINAMATH_CALUDE_combined_work_time_l3312_331256


namespace NUMINAMATH_CALUDE_train_length_l3312_331246

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 24 → 
  speed * crossing_time - platform_length = 230 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3312_331246


namespace NUMINAMATH_CALUDE_divisible_by_nine_l3312_331218

theorem divisible_by_nine (a b : ℤ) : ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l3312_331218


namespace NUMINAMATH_CALUDE_blaine_win_probability_l3312_331237

/-- Probability of Amelia getting heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine getting heads -/
def p_blaine : ℚ := 1/3

/-- Probability of Charlie getting heads -/
def p_charlie : ℚ := 1/5

/-- The probability that Blaine wins the game -/
def p_blaine_wins : ℚ := 25/36

theorem blaine_win_probability :
  let p_cycle : ℚ := (1 - p_amelia) * (1 - p_blaine) * (1 - p_charlie)
  p_blaine_wins = (1 - p_amelia) * p_blaine / (1 - p_cycle) := by
  sorry

end NUMINAMATH_CALUDE_blaine_win_probability_l3312_331237


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3312_331288

theorem least_integer_absolute_value_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y - 4| ≤ 25 → x ≤ y) ∧ |3*x - 4| ≤ 25 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_inequality_l3312_331288


namespace NUMINAMATH_CALUDE_seconds_in_week_l3312_331272

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem: The product of seconds per minute, minutes per hour, hours per day, and days per week
    equals the number of seconds in a week -/
theorem seconds_in_week :
  seconds_per_minute * minutes_per_hour * hours_per_day * days_per_week =
  (seconds_per_minute * minutes_per_hour * hours_per_day) * days_per_week :=
by sorry

end NUMINAMATH_CALUDE_seconds_in_week_l3312_331272


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3312_331213

/-- The y-intercept of the line 3x - 5y = 10 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 3*x - 5*y = 10 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3312_331213


namespace NUMINAMATH_CALUDE_smallest_multiple_37_congruent_7_mod_76_l3312_331203

theorem smallest_multiple_37_congruent_7_mod_76 : ∃ (n : ℕ), n > 0 ∧ 37 ∣ n ∧ n ≡ 7 [MOD 76] ∧ ∀ (m : ℕ), m > 0 ∧ 37 ∣ m ∧ m ≡ 7 [MOD 76] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_37_congruent_7_mod_76_l3312_331203


namespace NUMINAMATH_CALUDE_sum_of_roots_l3312_331267

theorem sum_of_roots (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3*x^3 - 7*x^2 + 2*x
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3312_331267


namespace NUMINAMATH_CALUDE_somu_age_problem_l3312_331268

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 5 = (father_age - 5) / 5 →
  somu_age = 10 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l3312_331268


namespace NUMINAMATH_CALUDE_two_positive_solutions_l3312_331224

theorem two_positive_solutions (a : ℝ) :
  (∃! x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
   (|2*x₁ - 1| - a = 0) ∧ (|2*x₂ - 1| - a = 0)) ↔ 
  (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_two_positive_solutions_l3312_331224


namespace NUMINAMATH_CALUDE_jessica_cut_thirteen_roses_l3312_331265

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses vase_roses garden_roses : ℕ) : ℕ :=
  vase_roses - initial_roses

/-- Theorem stating that Jessica cut 13 roses -/
theorem jessica_cut_thirteen_roses :
  ∃ (initial_roses vase_roses garden_roses : ℕ),
    initial_roses = 7 ∧
    garden_roses = 59 ∧
    vase_roses = 20 ∧
    roses_cut initial_roses vase_roses garden_roses = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_thirteen_roses_l3312_331265


namespace NUMINAMATH_CALUDE_compacted_cans_space_l3312_331264

/-- The space occupied by compacted cans -/
def space_occupied (num_cans : ℕ) (initial_space : ℝ) (compaction_ratio : ℝ) : ℝ :=
  (num_cans : ℝ) * initial_space * compaction_ratio

/-- Theorem: 60 cans, each initially 30 sq inches, compacted to 20%, occupy 360 sq inches -/
theorem compacted_cans_space :
  space_occupied 60 30 0.2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_compacted_cans_space_l3312_331264


namespace NUMINAMATH_CALUDE_hazel_walk_l3312_331235

/-- Hazel's walk problem -/
theorem hazel_walk (first_hour_distance : ℝ) (h1 : first_hour_distance = 2) :
  let second_hour_distance := 2 * first_hour_distance
  first_hour_distance + second_hour_distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_hazel_walk_l3312_331235


namespace NUMINAMATH_CALUDE_sequence_properties_l3312_331297

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℚ := n^2 + 2*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℚ := 2*n + 1

/-- The nth term of the sequence b_n -/
def b (n : ℕ+) : ℚ := 1 / (a n * a (n + 1))

/-- The sum of the first n terms of the sequence b_n -/
def T (n : ℕ+) : ℚ := n / (3 * (2*n + 3))

theorem sequence_properties (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → S k = k^2 + 2*k) →
  (a n = 2*n + 1) ∧
  (T n = n / (3 * (2*n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3312_331297


namespace NUMINAMATH_CALUDE_v_2015_equals_2_l3312_331236

/-- Function g as defined in the problem -/
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

/-- Sequence v defined recursively -/
def v : ℕ → ℕ
| 0 => 3
| n + 1 => g (v n)

/-- Theorem stating that the 2015th term of the sequence is 2 -/
theorem v_2015_equals_2 : v 2015 = 2 := by
  sorry

end NUMINAMATH_CALUDE_v_2015_equals_2_l3312_331236


namespace NUMINAMATH_CALUDE_sixth_term_constant_coefficient_x_squared_l3312_331293

/-- Expansion term of (x^(1/3) - 1/(2x^(1/3)))^n -/
def expansion_term (n : ℕ) (r : ℕ) : ℚ → ℚ :=
  λ x => (-1/2)^r * (n.choose r) * x^((n - 2*r : ℤ)/3)

/-- The 6th term (r = 5) is constant when n = 10 -/
theorem sixth_term_constant (n : ℕ) :
  (∀ x, expansion_term n 5 x = expansion_term n 5 1) → n = 10 :=
sorry

/-- When n = 10, the coefficient of x^2 is 45/4 -/
theorem coefficient_x_squared :
  expansion_term 10 2 = λ x => (45/4 : ℚ) * x^2 :=
sorry

end NUMINAMATH_CALUDE_sixth_term_constant_coefficient_x_squared_l3312_331293


namespace NUMINAMATH_CALUDE_equation_solution_l3312_331214

theorem equation_solution (a : ℝ) : 
  (∀ x, 2 * x + a = 3 ↔ x = -1) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3312_331214


namespace NUMINAMATH_CALUDE_smallest_obtuse_consecutive_triangle_perimeter_l3312_331259

/-- A triangle with consecutive integer side lengths -/
structure ConsecutiveTriangle where
  a : ℕ
  sides : Fin 3 → ℕ
  consecutive : ∀ i : Fin 2, sides i.succ = sides i + 1
  valid : a > 0 ∧ sides 0 = a

/-- Checks if a triangle is obtuse -/
def isObtuse (t : ConsecutiveTriangle) : Prop :=
  let a := t.sides 0
  let b := t.sides 1
  let c := t.sides 2
  a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveTriangle) : ℕ :=
  t.sides 0 + t.sides 1 + t.sides 2

/-- The main theorem -/
theorem smallest_obtuse_consecutive_triangle_perimeter :
  ∃ (t : ConsecutiveTriangle), isObtuse t ∧
    (∀ (t' : ConsecutiveTriangle), isObtuse t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_obtuse_consecutive_triangle_perimeter_l3312_331259


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3312_331222

theorem arithmetic_to_geometric_sequence :
  ∀ (a b c : ℝ),
  (∃ (x : ℝ), a = 3*x ∧ b = 4*x ∧ c = 5*x) →
  (b - a = c - b) →
  ((a + 1) * c = b^2) →
  (a = 15 ∧ b = 20 ∧ c = 25) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l3312_331222


namespace NUMINAMATH_CALUDE_derivative_cos_times_exp_sin_l3312_331202

/-- The derivative of f(x) = cos(x) * e^(sin(x)) -/
theorem derivative_cos_times_exp_sin (x : ℝ) :
  deriv (fun x => Real.cos x * Real.exp (Real.sin x)) x =
  (Real.cos x ^ 2 - Real.sin x) * Real.exp (Real.sin x) := by
sorry

end NUMINAMATH_CALUDE_derivative_cos_times_exp_sin_l3312_331202


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3312_331286

/-- The speed of a boat in still water, given the speed of the current and the upstream speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 20) 
  (h2 : upstream_speed = 30) : 
  ∃ (still_water_speed : ℝ), still_water_speed = 50 ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3312_331286


namespace NUMINAMATH_CALUDE_product_equals_243_l3312_331240

theorem product_equals_243 : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l3312_331240


namespace NUMINAMATH_CALUDE_sum_of_squares_implies_sum_l3312_331207

theorem sum_of_squares_implies_sum : ∀ (a b c : ℝ), 
  (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0 → a + 2*b + 3*c = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_implies_sum_l3312_331207


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l3312_331290

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60. -/
theorem first_term_of_geometric_series : ∀ (a : ℝ),
  (∑' n, a * (1/4)^n) = 80 → a = 60 := by sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l3312_331290


namespace NUMINAMATH_CALUDE_power_product_equals_two_l3312_331219

theorem power_product_equals_two :
  (-1/2)^2022 * 2^2023 = 2 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_two_l3312_331219


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l3312_331294

/-- Represents a triangular pyramid with vertex P and base ABC -/
structure TriangularPyramid where
  BC : ℝ
  CA : ℝ
  AB : ℝ
  dihedral_angle : ℝ

/-- The volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The volume of the specific triangular pyramid is 2 -/
theorem specific_pyramid_volume :
  let p : TriangularPyramid := {
    BC := 3,
    CA := 4,
    AB := 5,
    dihedral_angle := π / 4  -- 45° in radians
  }
  volume p = 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l3312_331294


namespace NUMINAMATH_CALUDE_subset_implies_bound_l3312_331223

theorem subset_implies_bound (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | 1 < x ∧ x < 2} →
  B = {x : ℝ | x < a} →
  A ⊆ B →
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_bound_l3312_331223


namespace NUMINAMATH_CALUDE_jess_height_l3312_331291

/-- Given the heights of Jana, Kelly, and Jess, prove that Jess is 72 inches tall. -/
theorem jess_height (jana kelly jess : ℕ) 
  (h1 : jana = kelly + 5)
  (h2 : kelly = jess - 3)
  (h3 : jana = 74) : 
  jess = 72 := by
  sorry

end NUMINAMATH_CALUDE_jess_height_l3312_331291


namespace NUMINAMATH_CALUDE_cubic_three_roots_l3312_331284

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_three_roots : ∃ (a b c : ℝ), (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_roots_l3312_331284


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3312_331296

theorem sum_of_a_and_b (a b : ℝ) : a^2 + b^2 + 2*a - 4*b + 5 = 0 → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3312_331296


namespace NUMINAMATH_CALUDE_fraction_square_sum_l3312_331266

theorem fraction_square_sum (a b c d : ℚ) (h : a / b + c / d = 1) :
  (a / b)^2 + c / d = (c / d)^2 + a / b := by sorry

end NUMINAMATH_CALUDE_fraction_square_sum_l3312_331266


namespace NUMINAMATH_CALUDE_remainder_of_polynomial_division_l3312_331289

theorem remainder_of_polynomial_division (x : ℤ) : 
  (x^2030 + 1) % (x^6 - x^4 + x^2 - 1) = x^2 - 1 := by sorry

end NUMINAMATH_CALUDE_remainder_of_polynomial_division_l3312_331289


namespace NUMINAMATH_CALUDE_f_value_at_3_l3312_331244

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (x + Real.sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = 4) : f a b 3 = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l3312_331244


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l3312_331258

/-- Given a square and a circle that intersect such that each side of the square
    contains a chord of the circle equal in length to twice the radius of the circle,
    the ratio of the area of the square to the area of the circle is 2/π. -/
theorem square_circle_area_ratio (s : Real) (r : Real) (π : Real) :
  s > 0 ∧ r > 0 ∧ π > 0 ∧ 
  (2 * r = s) ∧  -- chord length equals side length
  (π * r^2 = π * r * r) →  -- definition of circle area
  (s^2 / (π * r^2) = 2 / π) :=
sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l3312_331258


namespace NUMINAMATH_CALUDE_joes_test_scores_l3312_331216

theorem joes_test_scores (scores : Fin 4 → ℝ) 
  (avg_before : (scores 0 + scores 1 + scores 2 + scores 3) / 4 = 35)
  (avg_after : ∃ i, (scores 0 + scores 1 + scores 2 + scores 3 - scores i) / 3 = 40)
  (lowest : ∃ i, ∀ j, scores i ≤ scores j) :
  ∃ i, scores i = 20 ∧ ∀ j, scores i ≤ scores j :=
by sorry

end NUMINAMATH_CALUDE_joes_test_scores_l3312_331216


namespace NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l3312_331279

/-- Represents a frustum of a right circular cone -/
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

/-- Calculates the altitude of the small cone cut off from a frustum -/
def small_cone_altitude (f : Frustum) : ℝ :=
  f.altitude

/-- Theorem stating that for a specific frustum, the altitude of the small cone is 18 cm -/
theorem specific_frustum_small_cone_altitude :
  let f : Frustum := {
    altitude := 18,
    lower_base_area := 144 * Real.pi,
    upper_base_area := 36 * Real.pi
  }
  small_cone_altitude f = 18 := by sorry

end NUMINAMATH_CALUDE_specific_frustum_small_cone_altitude_l3312_331279


namespace NUMINAMATH_CALUDE_dogs_wearing_neither_l3312_331233

theorem dogs_wearing_neither (total : ℕ) (tags : ℕ) (collars : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : tags = 45)
  (h3 : collars = 40)
  (h4 : both = 6) :
  total - (tags + collars - both) = 1 := by
  sorry

end NUMINAMATH_CALUDE_dogs_wearing_neither_l3312_331233


namespace NUMINAMATH_CALUDE_total_shingles_needed_l3312_331205

/-- The number of shingles needed to cover a given area of roof --/
def shingles_per_square_foot : ℕ := 8

/-- The number of roofs to be shingled --/
def number_of_roofs : ℕ := 3

/-- The length of each rectangular side of a roof in feet --/
def roof_side_length : ℕ := 40

/-- The width of each rectangular side of a roof in feet --/
def roof_side_width : ℕ := 20

/-- The number of rectangular sides per roof --/
def sides_per_roof : ℕ := 2

/-- Theorem stating the total number of shingles needed --/
theorem total_shingles_needed :
  (number_of_roofs * sides_per_roof * roof_side_length * roof_side_width * shingles_per_square_foot) = 38400 := by
  sorry

end NUMINAMATH_CALUDE_total_shingles_needed_l3312_331205


namespace NUMINAMATH_CALUDE_tan_thirty_degrees_l3312_331242

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirty_degrees_l3312_331242


namespace NUMINAMATH_CALUDE_left_handed_women_percentage_l3312_331229

theorem left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * women / 2)
  (h3 : total = right_handed + left_handed)
  (h4 : total = men + women)
  (h5 : right_handed ≥ men)
  (h6 : right_handed = men) :
  women = left_handed ∧ left_handed * 100 / total = 25 :=
sorry

end NUMINAMATH_CALUDE_left_handed_women_percentage_l3312_331229


namespace NUMINAMATH_CALUDE_largest_non_representable_integer_l3312_331250

theorem largest_non_representable_integer 
  (a b c : ℕ+) 
  (h1 : Nat.gcd a b = 1) 
  (h2 : Nat.gcd b c = 1) 
  (h3 : Nat.gcd c a = 1) :
  ∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = b*c*x + c*a*y + a*b*z ∧
  ¬∃ (x y z : ℕ), 2*a*b*c - a*b - b*c - c*a = b*c*x + c*a*y + a*b*z :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_integer_l3312_331250


namespace NUMINAMATH_CALUDE_smallest_number_with_sum_l3312_331261

/-- Calculates the sum of all unique permutations of digits in a number -/
def sumOfPermutations (n : ℕ) : ℕ := sorry

/-- Checks if a number is the smallest with a given sum of permutations -/
def isSmallestWithSum (n : ℕ) (sum : ℕ) : Prop :=
  (sumOfPermutations n = sum) ∧ 
  (∀ m : ℕ, m < n → sumOfPermutations m ≠ sum)

/-- The main theorem stating that 47899 is the smallest number 
    whose sum of digit permutations is 4,933,284 -/
theorem smallest_number_with_sum :
  isSmallestWithSum 47899 4933284 := by sorry

end NUMINAMATH_CALUDE_smallest_number_with_sum_l3312_331261


namespace NUMINAMATH_CALUDE_friday_temperature_l3312_331254

theorem friday_temperature
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 46)
  (h3 : temp_mon = 43)
  : temp_fri = 35 := by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l3312_331254


namespace NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l3312_331260

/-- The number of possible outcomes for each die -/
def dice_outcomes : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := dice_outcomes * dice_outcomes

/-- The number of ways to get a sum of 7 with two dice -/
def favorable_outcomes : ℕ := 6

/-- The probability of getting a sum of 7 when throwing two fair dice -/
def probability_sum_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_seven_is_one_sixth :
  probability_sum_seven = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_seven_is_one_sixth_l3312_331260


namespace NUMINAMATH_CALUDE_smallest_d_value_l3312_331270

theorem smallest_d_value (d : ℝ) : 
  (4 * Real.sqrt 3)^2 + (d - 2)^2 = (4 * d)^2 → d ≥ 2.006 := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_value_l3312_331270


namespace NUMINAMATH_CALUDE_students_doing_homework_l3312_331204

theorem students_doing_homework (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) 
  (h1 : total = 60)
  (h2 : silent_reading = 3/8)
  (h3 : board_games = 1/4) :
  total - (Int.floor (silent_reading * total) + Int.floor (board_games * total)) = 22 :=
by sorry

end NUMINAMATH_CALUDE_students_doing_homework_l3312_331204


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3312_331278

/-- Given two isosceles right triangles with leg lengths 1, let x be the side length of a square
    inscribed in the first triangle with one vertex at the right angle, and y be the side length
    of a square inscribed in the second triangle with one side on the hypotenuse. -/
theorem inscribed_squares_ratio (x y : ℝ) 
  (hx : x = (1 : ℝ) / 2)  -- x is the side length of the square in the first triangle
  (hy : y = Real.sqrt 2 / 2) -- y is the side length of the square in the second triangle
  : x / y = Real.sqrt 2 := by
  sorry

#check inscribed_squares_ratio

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3312_331278


namespace NUMINAMATH_CALUDE_hiker_speed_difference_l3312_331212

/-- A hiker's journey over three days -/
def hiker_journey (v : ℝ) : Prop :=
  let day1_distance : ℝ := 18
  let day1_speed : ℝ := 3
  let day1_hours : ℝ := day1_distance / day1_speed
  let day2_hours : ℝ := day1_hours - 1
  let day2_distance : ℝ := day2_hours * v
  let day3_distance : ℝ := 5 * 3
  day1_distance + day2_distance + day3_distance = 53

theorem hiker_speed_difference : ∃ v : ℝ, hiker_journey v ∧ v - 3 = 1 := by
  sorry

#check hiker_speed_difference

end NUMINAMATH_CALUDE_hiker_speed_difference_l3312_331212


namespace NUMINAMATH_CALUDE_infinite_complementary_sequences_with_arithmetic_l3312_331292

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

def infinite_complementary_sequences (a b : ℕ → ℕ) : Prop :=
  (is_strictly_increasing a) ∧ 
  (is_strictly_increasing b) ∧
  (∀ n : ℕ, ∃ m : ℕ, n = a m ∨ n = b m) ∧
  (∀ n : ℕ, ¬(∃ m k : ℕ, n = a m ∧ n = b k))

def arithmetic_sequence (s : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, s (n + 1) = s n + d

theorem infinite_complementary_sequences_with_arithmetic (a b : ℕ → ℕ) :
  infinite_complementary_sequences a b →
  (∃ d : ℕ, arithmetic_sequence a d) →
  a 16 = 36 →
  (∀ n : ℕ, a n = 2 * n + 4) ∧
  (∀ n : ℕ, b n = if n ≤ 5 then n else 2 * n - 5) :=
sorry

end NUMINAMATH_CALUDE_infinite_complementary_sequences_with_arithmetic_l3312_331292


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l3312_331231

theorem unique_solution_is_two : 
  ∃! n : ℕ+, 
    (n : ℕ) ∣ (Nat.totient n)^(Nat.divisors n).card + 1 ∧ 
    ¬((Nat.divisors n).card^5 ∣ (n : ℕ)^(Nat.totient n) - 1) ∧
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l3312_331231


namespace NUMINAMATH_CALUDE_smallest_side_difference_l3312_331276

theorem smallest_side_difference (PQ QR PR : ℕ) : 
  PQ + QR + PR = 2021 →
  PQ < QR →
  QR ≤ PR →
  (∀ PQ' QR' PR' : ℕ, 
    PQ' + QR' + PR' = 2021 → 
    PQ' < QR' → 
    QR' ≤ PR' → 
    QR' - PQ' ≥ 1) →
  QR - PQ = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l3312_331276


namespace NUMINAMATH_CALUDE_break_room_tables_l3312_331248

/-- The number of people each table can seat -/
def seating_capacity_per_table : ℕ := 8

/-- The total seating capacity of the break room -/
def total_seating_capacity : ℕ := 32

/-- The number of tables in the break room -/
def number_of_tables : ℕ := total_seating_capacity / seating_capacity_per_table

theorem break_room_tables : number_of_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_break_room_tables_l3312_331248


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3312_331295

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - (a + 2) * x + b

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a b x ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2

-- Define the inequality function
def g (a b c x : ℝ) := (x - c) * (a * x - b)

-- Theorem statement
theorem quadratic_inequality_theorem (a b c : ℝ) (h : c ≠ 2) :
  solution_set a b →
  (a = 1 ∧ b = 2) ∧
  (∀ x, g a b c x > 0 ↔ 
    (c > 2 ∧ (x > c ∨ x < 2)) ∨
    (c < 2 ∧ (x > 2 ∨ x < c))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l3312_331295


namespace NUMINAMATH_CALUDE_minimum_n_for_inequality_l3312_331230

theorem minimum_n_for_inequality :
  (∃ (n : ℕ), ∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < 3 → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_n_for_inequality_l3312_331230


namespace NUMINAMATH_CALUDE_post_office_distance_l3312_331221

/-- The distance from the village to the post office satisfies the given conditions -/
theorem post_office_distance (D : ℝ) : D > 0 →
  (D / 25 + D / 4 = 5.8) → D = 20 := by sorry

end NUMINAMATH_CALUDE_post_office_distance_l3312_331221


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l3312_331200

/-- A line y = mx + b intersecting a circle and a hyperbola -/
structure LineIntersection where
  m : ℝ
  b : ℝ
  h_m : |m| < 1
  h_b : |b| < 1

/-- Points of intersection -/
structure IntersectionPoints (l : LineIntersection) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_circle_P : P.1^2 + P.2^2 = 1 ∧ P.2 = l.m * P.1 + l.b
  h_circle_Q : Q.1^2 + Q.2^2 = 1 ∧ Q.2 = l.m * Q.1 + l.b
  h_hyperbola_R : R.1^2 - R.2^2 = 1 ∧ R.2 = l.m * R.1 + l.b
  h_hyperbola_S : S.1^2 - S.2^2 = 1 ∧ S.2 = l.m * S.1 + l.b
  h_trisect : dist P R = dist P Q ∧ dist Q S = dist P Q

/-- The main theorem -/
theorem line_intersection_theorem (l : LineIntersection) (p : IntersectionPoints l) :
  (l.m = 0 ∧ l.b^2 = 4/5) ∨ (l.b = 0 ∧ l.m^2 = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l3312_331200


namespace NUMINAMATH_CALUDE_number_difference_l3312_331281

/-- 
Theorem: Given a three-digit number x and an even two-digit number y, 
if their difference is 3, then x = 101 and y = 98.
-/
theorem number_difference (x y : ℕ) : 
  (100 ≤ x ∧ x ≤ 999) →  -- x is a three-digit number
  (10 ≤ y ∧ y ≤ 98) →    -- y is a two-digit number
  Even y →               -- y is even
  x - y = 3 →            -- difference is 3
  x = 101 ∧ y = 98 :=
by sorry

end NUMINAMATH_CALUDE_number_difference_l3312_331281


namespace NUMINAMATH_CALUDE_coprime_pairs_count_l3312_331257

def count_coprime_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    1 ≤ a ∧ a ≤ b ∧ b ≤ 5 ∧ Nat.gcd a b = 1) 
    (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem coprime_pairs_count : count_coprime_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_coprime_pairs_count_l3312_331257


namespace NUMINAMATH_CALUDE_model_n_time_proof_l3312_331262

/-- Represents the time (in minutes) taken by a model N computer to complete the task -/
def model_n_time : ℝ := 12

/-- Represents the time (in minutes) taken by a model M computer to complete the task -/
def model_m_time : ℝ := 24

/-- Represents the number of model M computers used -/
def num_model_m : ℕ := 8

/-- Represents the total time (in minutes) taken by both models working together -/
def total_time : ℝ := 1

theorem model_n_time_proof :
  (num_model_m : ℝ) / model_m_time + (num_model_m : ℝ) / model_n_time = 1 / total_time :=
sorry

end NUMINAMATH_CALUDE_model_n_time_proof_l3312_331262


namespace NUMINAMATH_CALUDE_f_5_equals_18556_l3312_331299

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 1] x

theorem f_5_equals_18556 : f 5 = 18556 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_18556_l3312_331299


namespace NUMINAMATH_CALUDE_modular_multiplication_l3312_331277

theorem modular_multiplication (m : ℕ) : 
  0 ≤ m ∧ m < 25 ∧ m ≡ (66 * 77 * 88) [ZMOD 25] → m = 16 := by
  sorry

end NUMINAMATH_CALUDE_modular_multiplication_l3312_331277


namespace NUMINAMATH_CALUDE_triangle_ratio_l3312_331234

/-- Given a triangle ABC with the following properties:
  - M is the midpoint of BC
  - AB = 15
  - AC = 20
  - E is on AC
  - F is on AB
  - G is the intersection of EF and AM
  - AE = 3AF
  Prove that EG/GF = 2/3 -/
theorem triangle_ratio (A B C M E F G : ℝ × ℝ) : 
  (M = (B + C) / 2) →
  (dist A B = 15) →
  (dist A C = 20) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C) →
  (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (1 - s) • A + s • B) →
  (∃ r : ℝ, G = (1 - r) • E + r • F) →
  (∃ q : ℝ, G = (1 - q) • A + q • M) →
  (dist A E = 3 * dist A F) →
  (dist E G) / (dist G F) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3312_331234


namespace NUMINAMATH_CALUDE_centroid_perpendicular_distance_l3312_331263

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perpendicular distance from a point to a line
def perpendicularDistance (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

-- Define the centroid of a triangle
def centroid (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem centroid_perpendicular_distance (t : Triangle) (l : Line) :
  perpendicularDistance (centroid t) l =
    (perpendicularDistance t.A l + perpendicularDistance t.B l + perpendicularDistance t.C l) / 3 :=
  sorry

end NUMINAMATH_CALUDE_centroid_perpendicular_distance_l3312_331263


namespace NUMINAMATH_CALUDE_triangle_area_l3312_331215

theorem triangle_area (t : ℝ) : 
  let base : ℝ := 2 * t
  let height : ℝ := 3 * t + 1
  (1 / 2) * base * height = 3 * t^2 + t :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3312_331215


namespace NUMINAMATH_CALUDE_fraction_equality_l3312_331273

theorem fraction_equality (a b : ℝ) (h1 : b ≠ 0) (h2 : 2*a ≠ b) (h3 : a/b = 2/3) : 
  b/(2*a - b) = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3312_331273


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331232

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {-1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331232


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3312_331220

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-21, 19; 15, -13]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -5; 0.5, 3.5]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3312_331220


namespace NUMINAMATH_CALUDE_impossible_three_shell_piles_l3312_331241

/-- Represents the number of seashells at step n -/
def S (n : ℕ) : ℤ := 637 - n

/-- Represents the number of piles at step n -/
def P (n : ℕ) : ℤ := 1 + n

/-- Theorem stating that it's impossible to end up with only piles of exactly three seashells -/
theorem impossible_three_shell_piles : ¬ ∃ n : ℕ, S n = 3 * P n ∧ S n > 0 := by
  sorry

end NUMINAMATH_CALUDE_impossible_three_shell_piles_l3312_331241


namespace NUMINAMATH_CALUDE_primitive_existence_l3312_331271

open Set

theorem primitive_existence (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : a < c) (h2 : c < b)
  (h3 : ContinuousAt f c)
  (h4 : ∃ F1 : ℝ → ℝ, ∀ x ∈ Icc a c, HasDerivAt F1 (f x) x)
  (h5 : ∃ F2 : ℝ → ℝ, ∀ x ∈ Ico c b, HasDerivAt F2 (f x) x) :
  ∃ F : ℝ → ℝ, ∀ x ∈ Icc a b, HasDerivAt F (f x) x :=
sorry

end NUMINAMATH_CALUDE_primitive_existence_l3312_331271


namespace NUMINAMATH_CALUDE_point_P_coordinates_l3312_331217

def C (x : ℝ) : ℝ := x^3 - 10*x + 3

theorem point_P_coordinates :
  ∃! (x y : ℝ), 
    y = C x ∧ 
    x < 0 ∧ 
    y > 0 ∧ 
    (3 * x^2 - 10 = 2) ∧ 
    x = -2 ∧ 
    y = 15 := by
  sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l3312_331217


namespace NUMINAMATH_CALUDE_parallelogram_analogous_to_parallelepiped_l3312_331287

/-- A parallelepiped is a 3D shape with opposite faces parallel -/
structure Parallelepiped :=
  (opposite_faces_parallel : Bool)

/-- A parallelogram is a 2D shape with opposite sides parallel -/
structure Parallelogram :=
  (opposite_sides_parallel : Bool)

/-- An analogy between 3D and 2D shapes -/
def is_analogous (shape3D : Type) (shape2D : Type) : Prop :=
  ∃ (property3D : shape3D → Prop) (property2D : shape2D → Prop),
    ∀ (s3D : shape3D) (s2D : shape2D), property3D s3D ↔ property2D s2D

/-- Theorem: A parallelogram is the most analogous 2D shape to a parallelepiped -/
theorem parallelogram_analogous_to_parallelepiped :
  is_analogous Parallelepiped Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_analogous_to_parallelepiped_l3312_331287


namespace NUMINAMATH_CALUDE_odot_solution_l3312_331285

-- Define the binary operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

-- Theorem statement
theorem odot_solution (h : ℝ) :
  odot 9 h = 12 → h = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_odot_solution_l3312_331285


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331255

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 2*x + 5)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3312_331255


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3312_331274

def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3312_331274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3312_331243

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 6 and a₆ = 3, prove a₉ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 6)
  (h_6 : a 6 = 3) :
  a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3312_331243


namespace NUMINAMATH_CALUDE_intersection_vector_sum_l3312_331208

noncomputable def f (x : ℝ) : ℝ := (2 * Real.cos x ^ 2 + 1) / Real.log ((2 + x) / (2 - x))

theorem intersection_vector_sum (a : ℝ) (h_a : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), 
    (∀ x : ℝ, a * x - (f x) = 0 → x = A.1 ∨ x = B.1) →
    (A ≠ B) →
    (∀ m n : ℝ, 
      (A.1 - m, A.2 - n) + (B.1 - m, B.2 - n) = (m - 6, n) →
      m + n = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_vector_sum_l3312_331208


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l3312_331201

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![1, -1]
def c (m n : ℝ) : Fin 2 → ℝ := ![m - 2, -n]

theorem vector_parallel_proof (m n : ℝ) (hm : m > 0) (hn : n > 0)
  (h_parallel : ∃ (k : ℝ), ∀ i, (a - b) i = k * c m n i) :
  (2 * m + n = 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 4 → x * y ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l3312_331201


namespace NUMINAMATH_CALUDE_words_with_A_count_l3312_331282

/-- The number of letters in our alphabet -/
def n : ℕ := 4

/-- The length of words we're considering -/
def k : ℕ := 3

/-- The number of letters in our alphabet excluding 'A' -/
def m : ℕ := 3

/-- The number of 3-letter words that can be made from the letters A, B, C, and D, 
    with at least one A being used and allowing repetition of letters -/
def words_with_A : ℕ := n^k - m^k

theorem words_with_A_count : words_with_A = 37 := by sorry

end NUMINAMATH_CALUDE_words_with_A_count_l3312_331282


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l3312_331211

/-- The determinant of the matrix
    [1, cos(a-b), sin(a);
     cos(a-b), 1, sin(b);
     sin(a), sin(b), 1]
    is equal to 0 for any real numbers a and b. -/
theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.cos (a - b), Real.sin a;
                                        Real.cos (a - b), 1, Real.sin b;
                                        Real.sin a, Real.sin b, 1]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l3312_331211


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3312_331298

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3312_331298


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3312_331226

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + 2 * Real.sqrt (b + 3 * Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 3 x = 18 ∧ x = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3312_331226


namespace NUMINAMATH_CALUDE_factor_expression_l3312_331227

theorem factor_expression (x : ℝ) : 60 * x^5 - 180 * x^9 = 60 * x^5 * (1 - 3 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3312_331227


namespace NUMINAMATH_CALUDE_percentage_difference_l3312_331247

theorem percentage_difference : 
  (67.5 / 100 * 250) - (52.3 / 100 * 180) = 74.61 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3312_331247


namespace NUMINAMATH_CALUDE_unique_integer_sum_l3312_331239

theorem unique_integer_sum (C y M A : ℕ) : 
  C > 0 ∧ y > 0 ∧ M > 0 ∧ A > 0 →
  C ≠ y ∧ C ≠ M ∧ C ≠ A ∧ y ≠ M ∧ y ≠ A ∧ M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_sum_l3312_331239


namespace NUMINAMATH_CALUDE_one_true_proposition_l3312_331245

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1

-- Define the inverse of the proposition
def inverse_proposition (a b : ℝ) : Prop :=
  a + b ≠ 1 → a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0

-- Define the negation of the proposition
def negation_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 = 0 → a + b = 1

-- Define the contrapositive of the proposition
def contrapositive_proposition (a b : ℝ) : Prop :=
  a + b = 1 → a^2 + 2*a*b + b^2 + a + b - 2 = 0

-- Theorem statement
theorem one_true_proposition :
  ∃! p : (ℝ → ℝ → Prop), 
    (p = inverse_proposition ∨ p = negation_proposition ∨ p = contrapositive_proposition) ∧
    (∀ a b : ℝ, p a b) :=
  sorry

end NUMINAMATH_CALUDE_one_true_proposition_l3312_331245


namespace NUMINAMATH_CALUDE_probability_two_blue_marbles_l3312_331269

def total_marbles : ℕ := 3 + 4 + 8 + 5

def blue_marbles : ℕ := 8

theorem probability_two_blue_marbles :
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) = 14 / 95 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_blue_marbles_l3312_331269


namespace NUMINAMATH_CALUDE_students_above_120_l3312_331280

/-- Normal distribution parameters -/
structure NormalDist where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability function for normal distribution -/
noncomputable def prob (nd : NormalDist) (a b : ℝ) : ℝ := sorry

/-- Theorem: Number of students scoring above 120 -/
theorem students_above_120 (nd : NormalDist) (total_students : ℕ) :
  nd.μ = 90 →
  prob nd 60 120 = 0.8 →
  total_students = 780 →
  ⌊(1 - prob nd 60 120) / 2 * total_students⌋ = 78 := by sorry

end NUMINAMATH_CALUDE_students_above_120_l3312_331280


namespace NUMINAMATH_CALUDE_smallest_divisible_ones_l3312_331252

/-- A number composed of n ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- A number composed of n threes -/
def threes (n : ℕ) : ℕ := 3 * ones n

theorem smallest_divisible_ones (n : ℕ) : 
  (∀ k < n, ¬ (threes 100 ∣ ones k)) ∧ (threes 100 ∣ ones n) → n = 300 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_ones_l3312_331252


namespace NUMINAMATH_CALUDE_function_lower_bound_l3312_331251

open Real

theorem function_lower_bound (a x : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x => a * (exp x + a) - x
  f x > 2 * log a + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l3312_331251


namespace NUMINAMATH_CALUDE_office_age_problem_l3312_331283

/-- Given information about the ages of people in an office, prove that the average age of a specific group is 14 years. -/
theorem office_age_problem (total_people : ℕ) (avg_age_all : ℕ) (group1_size : ℕ) (group1_avg_age : ℕ) (group2_size : ℕ) (person15_age : ℕ) :
  total_people = 17 →
  avg_age_all = 15 →
  group1_size = 9 →
  group1_avg_age = 16 →
  group2_size = 5 →
  person15_age = 41 →
  (total_people * avg_age_all - group1_size * group1_avg_age - person15_age) / group2_size = 14 :=
by sorry

end NUMINAMATH_CALUDE_office_age_problem_l3312_331283


namespace NUMINAMATH_CALUDE_sample_size_is_80_l3312_331249

/-- Represents the ratio of product models A, B, and C -/
def productRatio : Fin 3 → ℕ
  | 0 => 2  -- Model A
  | 1 => 3  -- Model B
  | 2 => 5  -- Model C
  | _ => 0  -- This case should never occur due to Fin 3

/-- Calculates the total ratio sum -/
def totalRatio : ℕ := (productRatio 0) + (productRatio 1) + (productRatio 2)

/-- Represents the number of units of model A in the sample -/
def modelAUnits : ℕ := 16

/-- Theorem stating that the sample size is 80 given the conditions -/
theorem sample_size_is_80 :
  ∃ (n : ℕ), n * (productRatio 0) / totalRatio = modelAUnits ∧ n = 80 :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_80_l3312_331249
