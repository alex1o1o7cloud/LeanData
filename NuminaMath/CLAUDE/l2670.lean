import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_ages_seven_years_hence_l2670_267040

-- Define X's current age
def X_current : ℕ := 45

-- Define Y's current age as a function of X's current age
def Y_current : ℕ := X_current - 21

-- Theorem to prove
theorem sum_of_ages_seven_years_hence : 
  X_current + Y_current + 14 = 83 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_seven_years_hence_l2670_267040


namespace NUMINAMATH_CALUDE_largest_n_power_inequality_l2670_267065

theorem largest_n_power_inequality : ∃ (n : ℕ), n = 11 ∧ 
  (∀ m : ℕ, m^200 < 5^300 → m ≤ n) ∧ n^200 < 5^300 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_power_inequality_l2670_267065


namespace NUMINAMATH_CALUDE_kevin_sells_50_crates_l2670_267059

/-- Kevin's weekly fruit sales --/
def weekly_fruit_sales (grapes mangoes passion_fruits : ℕ) : ℕ :=
  grapes + mangoes + passion_fruits

/-- Theorem: Kevin sells 50 crates of fruit per week --/
theorem kevin_sells_50_crates :
  weekly_fruit_sales 13 20 17 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kevin_sells_50_crates_l2670_267059


namespace NUMINAMATH_CALUDE_worker_idle_days_l2670_267019

theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_work_day : ℕ) 
  (deduction_per_idle_day : ℕ) 
  (total_payment : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_work_day = 20)
  (h3 : deduction_per_idle_day = 3)
  (h4 : total_payment = 280) :
  ∃ (idle_days : ℕ) (work_days : ℕ),
    idle_days + work_days = total_days ∧
    pay_per_work_day * work_days - deduction_per_idle_day * idle_days = total_payment ∧
    idle_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_worker_idle_days_l2670_267019


namespace NUMINAMATH_CALUDE_teacher_selection_plans_l2670_267077

theorem teacher_selection_plans (male_teachers female_teachers selected_teachers : ℕ) 
  (h1 : male_teachers = 5)
  (h2 : female_teachers = 4)
  (h3 : selected_teachers = 3) :
  (Nat.choose male_teachers 2 * Nat.choose female_teachers 1 * Nat.factorial selected_teachers) +
  (Nat.choose male_teachers 1 * Nat.choose female_teachers 2 * Nat.factorial selected_teachers) = 420 := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_plans_l2670_267077


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_216_l2670_267022

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let square_area : ℝ := square_side_length ^ 2
  let total_area : ℝ := 2 * square_area
  let overlap_area : ℝ := square_area / 2
  total_area - overlap_area

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square, is 216 square units. -/
theorem area_of_overlapping_squares_is_216 : area_of_overlapping_squares = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_216_l2670_267022


namespace NUMINAMATH_CALUDE_slower_train_speed_l2670_267051

/-- Prove that given two trains moving in the same direction, with the faster train
    traveling at 50 km/hr, taking 15 seconds to pass a man in the slower train,
    and having a length of 75 meters, the speed of the slower train is 32 km/hr. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (passing_time : ℝ)
  (faster_train_length : ℝ)
  (h1 : faster_train_speed = 50)
  (h2 : passing_time = 15)
  (h3 : faster_train_length = 75) :
  ∃ (slower_train_speed : ℝ),
    slower_train_speed = 32 ∧
    (faster_train_speed - slower_train_speed) * 1000 / 3600 = faster_train_length / passing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l2670_267051


namespace NUMINAMATH_CALUDE_talent_show_participants_l2670_267041

theorem talent_show_participants (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 34 →
  difference = 22 →
  girls = (total + difference) / 2 →
  girls = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_talent_show_participants_l2670_267041


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2670_267052

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 - 2*x^5 + 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 12 = 
  (x - 1) * (x^5 - x^4 + 2*x^3 - 2*x^2 + 3*x - 3) + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2670_267052


namespace NUMINAMATH_CALUDE_remaining_length_is_twelve_l2670_267066

/-- Represents a rectangle with specific dimensions -/
structure Rectangle :=
  (left : ℝ)
  (top1 : ℝ)
  (top2 : ℝ)
  (top3 : ℝ)

/-- Calculates the total length of remaining segments after removing sides -/
def remaining_length (r : Rectangle) : ℝ :=
  r.left + r.top1 + r.top2 + r.top3

/-- Theorem stating that for a rectangle with given dimensions, 
    the remaining length after removing sides is 12 units -/
theorem remaining_length_is_twelve (r : Rectangle) 
  (h1 : r.left = 8)
  (h2 : r.top1 = 2)
  (h3 : r.top2 = 1)
  (h4 : r.top3 = 1) :
  remaining_length r = 12 := by
  sorry

#check remaining_length_is_twelve

end NUMINAMATH_CALUDE_remaining_length_is_twelve_l2670_267066


namespace NUMINAMATH_CALUDE_ratio_AD_DC_is_3_to_2_l2670_267024

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def point_D_on_AC (A C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define BD = 7
def BD_equals_7 (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 7

-- Theorem statement
theorem ratio_AD_DC_is_3_to_2 
  (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) 
  (h_D_on_AC : point_D_on_AC A C D) 
  (h_BD : BD_equals_7 B D) : 
  ∃ (AD DC : ℝ), AD / DC = 3 / 2 := 
sorry

end NUMINAMATH_CALUDE_ratio_AD_DC_is_3_to_2_l2670_267024


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2670_267070

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors Ice-cream-o-rama can create -/
theorem ice_cream_flavors : distribute 6 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2670_267070


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l2670_267089

theorem fraction_difference_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + b = a * b) : 
  1 / a - 1 / b = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l2670_267089


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2670_267079

theorem interest_rate_calculation (principal : ℝ) (difference : ℝ) (time : ℕ) (rate : ℝ) : 
  principal = 15000 →
  difference = 150 →
  time = 2 →
  principal * ((1 + rate)^time - 1) - principal * rate * time = difference →
  rate = 0.1 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2670_267079


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2670_267006

theorem unique_solution_for_equation : ∃! (n : ℕ), n > 0 ∧ n^2 + n + 6*n = 210 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2670_267006


namespace NUMINAMATH_CALUDE_line_segment_coordinates_l2670_267075

theorem line_segment_coordinates (y : ℝ) : 
  y > 0 → 
  ((2 - 6)^2 + (y - 10)^2 = 10^2) →
  (y = 10 - 2 * Real.sqrt 21 ∨ y = 10 + 2 * Real.sqrt 21) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_coordinates_l2670_267075


namespace NUMINAMATH_CALUDE_probability_at_least_two_succeed_l2670_267073

theorem probability_at_least_two_succeed (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 1/2) (h₂ : p₂ = 1/4) (h₃ : p₃ = 1/5) : 
  p₁ * p₂ * (1 - p₃) + p₁ * (1 - p₂) * p₃ + (1 - p₁) * p₂ * p₃ + p₁ * p₂ * p₃ = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_succeed_l2670_267073


namespace NUMINAMATH_CALUDE_system_solution_cubic_equation_solution_l2670_267001

-- Problem 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

-- Problem 2: Cubic equation
theorem cubic_equation_solution :
  ∃! x : ℝ, (2 * x - 1)^3 = -8 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_cubic_equation_solution_l2670_267001


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l2670_267007

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l2670_267007


namespace NUMINAMATH_CALUDE_fraction_equality_l2670_267023

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (1/x - 1/y) / (1/x + 1/y) = 1001 → (x + y) / (x - y) = -(1/1001) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2670_267023


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2670_267078

/-- Given a triangle with vertices (0, 0), (x, 3x), and (2x, 0), 
    if its area is 150 square units and x > 0, then x = 5√2 -/
theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * (2*x) * (3*x) = 150 → x = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2670_267078


namespace NUMINAMATH_CALUDE_square_diagonal_length_l2670_267032

theorem square_diagonal_length (A : ℝ) (d : ℝ) : 
  A = 392 → d = 28 → d^2 = 2 * A :=
by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_length_l2670_267032


namespace NUMINAMATH_CALUDE_gcf_of_90_and_135_l2670_267030

theorem gcf_of_90_and_135 : Nat.gcd 90 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_135_l2670_267030


namespace NUMINAMATH_CALUDE_sin_sum_product_zero_l2670_267044

theorem sin_sum_product_zero : 
  Real.sin (523 * π / 180) * Real.sin (943 * π / 180) + 
  Real.sin (1333 * π / 180) * Real.sin (313 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_zero_l2670_267044


namespace NUMINAMATH_CALUDE_cos_negative_45_degrees_l2670_267015

theorem cos_negative_45_degrees : Real.cos (-(Real.pi / 4)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_45_degrees_l2670_267015


namespace NUMINAMATH_CALUDE_trig_identity_l2670_267043

theorem trig_identity : 
  Real.sin (44 * π / 180) * Real.cos (14 * π / 180) - 
  Real.cos (44 * π / 180) * Real.cos (76 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2670_267043


namespace NUMINAMATH_CALUDE_sandys_carrots_l2670_267085

/-- Sandy's carrot problem -/
theorem sandys_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) 
  (h1 : initial_carrots = 6)
  (h2 : taken_carrots = 3) :
  initial_carrots - taken_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_carrots_l2670_267085


namespace NUMINAMATH_CALUDE_savings_calculation_l2670_267071

/-- Calculates the total savings over a 4-month period with varying weekly savings rates --/
def total_savings (smartphone_cost initial_savings gym_membership first_period_weekly_savings second_period_weekly_savings : ℕ) : ℕ :=
  let first_period_savings := first_period_weekly_savings * 4 * 2
  let second_period_savings := second_period_weekly_savings * 4 * 2
  first_period_savings + second_period_savings

/-- Proves that given the specified conditions, the total savings after 4 months is $1040 --/
theorem savings_calculation (smartphone_cost : ℕ) (initial_savings : ℕ) (gym_membership : ℕ) 
  (h1 : smartphone_cost = 800)
  (h2 : initial_savings = 200)
  (h3 : gym_membership = 50)
  (h4 : total_savings 800 200 50 50 80 = 1040) :
  total_savings smartphone_cost initial_savings gym_membership 50 80 = 1040 :=
by sorry

#eval total_savings 800 200 50 50 80

end NUMINAMATH_CALUDE_savings_calculation_l2670_267071


namespace NUMINAMATH_CALUDE_intersection_A_B_l2670_267050

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | x * (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2670_267050


namespace NUMINAMATH_CALUDE_function_solution_set_l2670_267064

-- Define the function f
def f (x a : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_set_l2670_267064


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2670_267058

theorem largest_multiple_of_15_under_500 :
  ∃ (n : ℕ), n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ (m : ℕ), m * 15 < 500 → m * 15 ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l2670_267058


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2670_267013

theorem quadratic_equation_roots (k : ℝ) : 
  (2 : ℝ) ^ 2 + k * 2 - 10 = 0 → k = 3 ∧ ∃ x : ℝ, x ≠ 2 ∧ x ^ 2 + k * x - 10 = 0 ∧ x = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2670_267013


namespace NUMINAMATH_CALUDE_no_distinct_roots_l2670_267042

theorem no_distinct_roots : ¬∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (a^2 - 2*b*a + c^2 = 0) ∧
  (b^2 - 2*c*b + a^2 = 0) ∧
  (c^2 - 2*a*c + b^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_roots_l2670_267042


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2670_267097

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 2, y := -5 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2670_267097


namespace NUMINAMATH_CALUDE_points_collinear_l2670_267084

/-- Three points in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for three points -/
def collinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem points_collinear : 
  let p1 : Point2D := ⟨1, 2⟩
  let p2 : Point2D := ⟨3, 8⟩
  let p3 : Point2D := ⟨4, 11⟩
  collinear p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_points_collinear_l2670_267084


namespace NUMINAMATH_CALUDE_value_of_y_l2670_267017

theorem value_of_y : ∃ y : ℝ, (3 * y) / 4 = 15 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2670_267017


namespace NUMINAMATH_CALUDE_max_coefficients_bound_l2670_267098

variable (p q x y A B C α β γ : ℝ)

theorem max_coefficients_bound 
  (h_p : 0 ≤ p ∧ p ≤ 1) 
  (h_q : 0 ≤ q ∧ q ≤ 1) 
  (h_eq1 : ∀ x y, (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2)
  (h_eq2 : ∀ x y, (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2) :
  max A (max B C) ≥ 4/9 ∧ max α (max β γ) ≥ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_max_coefficients_bound_l2670_267098


namespace NUMINAMATH_CALUDE_max_value_cubic_ratio_l2670_267002

theorem max_value_cubic_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 ∧
  (x + y)^3 / (x^3 + y^3) = 4 ↔ x = y :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_ratio_l2670_267002


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l2670_267038

theorem unique_number_with_properties : ∃! x : ℕ, 
  (x ≥ 10000 ∧ x < 100000) ∧ 
  (x * (x - 1) % 100000 = 0) ∧
  ((x / 1000) % 10 = 0) ∧
  ((x % 3125 = 0 ∧ (x - 1) % 32 = 0) ∨ ((x - 1) % 3125 = 0 ∧ x % 32 = 0)) :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l2670_267038


namespace NUMINAMATH_CALUDE_power_of_five_reciprocal_l2670_267090

theorem power_of_five_reciprocal (x y : ℕ) : 
  (2^x : ℕ) ∣ 144 ∧ 
  (∀ k > x, ¬((2^k : ℕ) ∣ 144)) ∧ 
  (3^y : ℕ) ∣ 144 ∧ 
  (∀ k > y, ¬((3^k : ℕ) ∣ 144)) →
  (1/5 : ℚ)^(y - x) = 25 := by
sorry

end NUMINAMATH_CALUDE_power_of_five_reciprocal_l2670_267090


namespace NUMINAMATH_CALUDE_largest_angle_and_sinC_l2670_267083

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle with a = 7, b = 3, c = 5 -/
def givenTriangle : Triangle where
  a := 7
  b := 3
  c := 5
  A := sorry
  B := sorry
  C := sorry

theorem largest_angle_and_sinC (t : Triangle) (h : t = givenTriangle) :
  (t.A > t.B ∧ t.A > t.C) ∧ t.A = Real.pi * (2/3) ∧ Real.sin t.C = 5 * Real.sqrt 3 / 14 := by
  sorry

#check largest_angle_and_sinC

end NUMINAMATH_CALUDE_largest_angle_and_sinC_l2670_267083


namespace NUMINAMATH_CALUDE_subset_condition_l2670_267068

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| < a}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

theorem subset_condition (a : ℝ) : B ⊆ A a ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2670_267068


namespace NUMINAMATH_CALUDE_daniel_driving_speed_l2670_267063

/-- The speed at which Daniel drove on Monday for the first 32 miles -/
def monday_first_speed (x : ℝ) : ℝ := 2 * x

theorem daniel_driving_speed (x : ℝ) (h_x_pos : x > 0) :
  let total_distance : ℝ := 100
  let monday_first_distance : ℝ := 32
  let monday_second_distance : ℝ := total_distance - monday_first_distance
  let sunday_time : ℝ := total_distance / x
  let monday_time : ℝ := monday_first_distance / (monday_first_speed x) + monday_second_distance / (x / 2)
  let time_increase_ratio : ℝ := 1.52
  monday_time = time_increase_ratio * sunday_time :=
by sorry

#check daniel_driving_speed

end NUMINAMATH_CALUDE_daniel_driving_speed_l2670_267063


namespace NUMINAMATH_CALUDE_percentage_and_reduction_l2670_267067

-- Define the relationship between two numbers
def is_five_percent_more (a b : ℝ) : Prop := a = b * 1.05

-- Define the reduction of 10 kilograms by 10%
def reduced_by_ten_percent (x : ℝ) : ℝ := x * 0.9

theorem percentage_and_reduction :
  (∀ a b : ℝ, is_five_percent_more a b → a = b * 1.05) ∧
  (reduced_by_ten_percent 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_percentage_and_reduction_l2670_267067


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l2670_267035

theorem greatest_integer_problem (n : ℕ) : n < 50 ∧
  (∃ a : ℤ, n = 6 * a - 1) ∧
  (∃ b : ℤ, n = 8 * b - 5) ∧
  (∃ c : ℤ, n = 3 * c + 2) ∧
  (∀ m : ℕ, m < 50 →
    (∃ a : ℤ, m = 6 * a - 1) →
    (∃ b : ℤ, m = 8 * b - 5) →
    (∃ c : ℤ, m = 3 * c + 2) →
    m ≤ n) →
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l2670_267035


namespace NUMINAMATH_CALUDE_sin_alpha_minus_cos_alpha_l2670_267045

theorem sin_alpha_minus_cos_alpha (α : Real) (h : Real.tan α = -3/4) :
  Real.sin α * (Real.sin α - Real.cos α) = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_cos_alpha_l2670_267045


namespace NUMINAMATH_CALUDE_total_legs_is_22_l2670_267031

/-- The number of legs for each type of animal -/
def dog_legs : ℕ := 4
def bird_legs : ℕ := 2
def insect_legs : ℕ := 6

/-- The number of each type of animal -/
def num_dogs : ℕ := 3
def num_birds : ℕ := 2
def num_insects : ℕ := 2

/-- The total number of legs -/
def total_legs : ℕ := num_dogs * dog_legs + num_birds * bird_legs + num_insects * insect_legs

theorem total_legs_is_22 : total_legs = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_22_l2670_267031


namespace NUMINAMATH_CALUDE_expression_simplification_l2670_267028

theorem expression_simplification :
  4 * Real.sqrt 2 * Real.sqrt 3 - Real.sqrt 12 / Real.sqrt 2 + Real.sqrt 24 = 5 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2670_267028


namespace NUMINAMATH_CALUDE_angle_B_is_70_l2670_267088

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the properties of the triangle
def rightTriangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180 ∧ t.A = 20 ∧ t.C = 90

-- Theorem statement
theorem angle_B_is_70 (t : Triangle) (h : rightTriangle t) : t.B = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_70_l2670_267088


namespace NUMINAMATH_CALUDE_cats_weight_l2670_267009

/-- The weight of two cats, where one cat weighs 2 kilograms and the other is twice as heavy, is 6 kilograms. -/
theorem cats_weight (weight_cat1 weight_cat2 : ℝ) : 
  weight_cat1 = 2 → weight_cat2 = 2 * weight_cat1 → weight_cat1 + weight_cat2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cats_weight_l2670_267009


namespace NUMINAMATH_CALUDE_equation_transformation_l2670_267091

theorem equation_transformation (x : ℝ) : 
  ((x - 1) / 2 - 1 = (3 * x + 1) / 3) ↔ (3 * (x - 1) - 6 = 2 * (3 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l2670_267091


namespace NUMINAMATH_CALUDE_polygon_perimeter_is_52_l2670_267061

/-- The perimeter of a polygon formed by removing six 2x2 squares from the sides of an 8x12 rectangle -/
def polygon_perimeter (rectangle_length : ℕ) (rectangle_width : ℕ) (square_side : ℕ) (num_squares : ℕ) : ℕ :=
  2 * (rectangle_length + rectangle_width) + 2 * num_squares * square_side

theorem polygon_perimeter_is_52 :
  polygon_perimeter 12 8 2 6 = 52 := by
  sorry

end NUMINAMATH_CALUDE_polygon_perimeter_is_52_l2670_267061


namespace NUMINAMATH_CALUDE_answer_key_combinations_l2670_267000

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- Represents the total number of possible true-false combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- Represents the number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The main theorem stating the number of ways to create the answer key -/
theorem answer_key_combinations : 
  (total_true_false_combinations - invalid_true_false_combinations) * 
  (multiple_choice_options^multiple_choice_questions) = 96 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l2670_267000


namespace NUMINAMATH_CALUDE_product_minus_constant_l2670_267093

theorem product_minus_constant (P Q R S : ℕ+) : 
  (P + Q + R + S : ℝ) = 104 →
  (P : ℝ) + 5 = (Q : ℝ) - 5 →
  (P : ℝ) + 5 = (R : ℝ) * 2 →
  (P : ℝ) + 5 = (S : ℝ) / 2 →
  (P : ℝ) * (Q : ℝ) * (R : ℝ) * (S : ℝ) - 200 = 267442.5 := by
sorry

end NUMINAMATH_CALUDE_product_minus_constant_l2670_267093


namespace NUMINAMATH_CALUDE_absolute_value_integral_l2670_267062

theorem absolute_value_integral : ∫ x in (0:ℝ)..(4:ℝ), |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l2670_267062


namespace NUMINAMATH_CALUDE_flagpole_height_l2670_267048

/-- Given a right triangle with hypotenuse 5 meters, where a person of height 1.6 meters
    touches the hypotenuse at a point 4 meters from one end of the base,
    prove that the height of the triangle (perpendicular to the base) is 8 meters. -/
theorem flagpole_height (base : ℝ) (hypotenuse : ℝ) (person_height : ℝ) (person_distance : ℝ) :
  base = 5 →
  person_height = 1.6 →
  person_distance = 4 →
  ∃ (height : ℝ), height = 8 ∧ height * base = person_height * person_distance :=
by sorry

end NUMINAMATH_CALUDE_flagpole_height_l2670_267048


namespace NUMINAMATH_CALUDE_optimal_garden_length_l2670_267036

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  parallel_length : ℝ  -- Length of the side parallel to the house
  perpendicular_length : ℝ  -- Length of the sides perpendicular to the house
  house_length : ℝ  -- Length of the house wall
  fence_cost_per_foot : ℝ  -- Cost of the fence per foot
  total_fence_cost : ℝ  -- Total cost of the fence

/-- The area of the garden. -/
def garden_area (g : Garden) : ℝ :=
  g.parallel_length * g.perpendicular_length

/-- The total length of the fence. -/
def fence_length (g : Garden) : ℝ :=
  g.parallel_length + 2 * g.perpendicular_length

/-- Theorem stating that the optimal garden length is 100 feet. -/
theorem optimal_garden_length (g : Garden) 
    (h1 : g.house_length = 500)
    (h2 : g.fence_cost_per_foot = 10)
    (h3 : g.total_fence_cost = 2000)
    (h4 : fence_length g = g.total_fence_cost / g.fence_cost_per_foot) :
  ∃ (optimal_length : ℝ), 
    optimal_length = 100 ∧ 
    ∀ (other_length : ℝ), 
      0 < other_length → 
      other_length ≤ fence_length g / 2 →
      garden_area { g with parallel_length := other_length, 
                           perpendicular_length := (fence_length g - other_length) / 2 } ≤ 
      garden_area { g with parallel_length := optimal_length, 
                           perpendicular_length := (fence_length g - optimal_length) / 2 } :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_length_l2670_267036


namespace NUMINAMATH_CALUDE_arccos_zero_l2670_267056

theorem arccos_zero (h : Set.Icc 0 π = Set.range acos) : acos 0 = π / 2 := by sorry

end NUMINAMATH_CALUDE_arccos_zero_l2670_267056


namespace NUMINAMATH_CALUDE_arithmetic_progression_quadratic_roots_l2670_267034

/-- Given non-zero real numbers a, b, c forming an arithmetic progression with b as the middle term,
    the quadratic equation ax^2 + 2√2bx + c = 0 has two distinct real roots. -/
theorem arithmetic_progression_quadratic_roots (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 2 * Real.sqrt 2 * b * x₁ + c = 0 ∧
                a * x₂^2 + 2 * Real.sqrt 2 * b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_quadratic_roots_l2670_267034


namespace NUMINAMATH_CALUDE_library_visitors_on_sunday_l2670_267076

/-- The average number of visitors on non-Sunday days -/
def avg_visitors_non_sunday : ℕ := 240

/-- The total number of days in the month -/
def total_days : ℕ := 30

/-- The number of Sundays in the month -/
def num_sundays : ℕ := 5

/-- The average number of visitors per day in the month -/
def avg_visitors_per_day : ℕ := 300

/-- The average number of visitors on Sundays -/
def avg_visitors_sunday : ℕ := 600

theorem library_visitors_on_sunday :
  num_sundays * avg_visitors_sunday + (total_days - num_sundays) * avg_visitors_non_sunday =
  total_days * avg_visitors_per_day := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_on_sunday_l2670_267076


namespace NUMINAMATH_CALUDE_four_color_theorem_l2670_267020

/-- Represents a country on the map -/
structure Country where
  borders : ℕ
  border_divisible_by_three : borders % 3 = 0

/-- Represents a map of countries -/
structure Map where
  countries : List Country

/-- Represents a coloring of the map -/
def Coloring := Map → Country → Fin 4

/-- A coloring is proper if no adjacent countries have the same color -/
def is_proper_coloring (m : Map) (c : Coloring) : Prop := sorry

/-- Volynsky's theorem -/
axiom volynsky_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c

/-- Main theorem: If the number of borders of each country on a normal map
    is divisible by 3, then the map can be properly colored with four colors -/
theorem four_color_theorem (m : Map) : 
  (∀ country ∈ m.countries, country.borders % 3 = 0) → 
  ∃ c : Coloring, is_proper_coloring m c :=
by
  sorry

end NUMINAMATH_CALUDE_four_color_theorem_l2670_267020


namespace NUMINAMATH_CALUDE_probability_not_red_l2670_267099

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10
def non_red_jelly_beans : ℕ := 8 + 9 + 10

theorem probability_not_red :
  (non_red_jelly_beans : ℚ) / total_jelly_beans = 27 / 34 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_l2670_267099


namespace NUMINAMATH_CALUDE_bike_route_total_length_l2670_267037

/-- The total length of a rectangular bike route -/
def bike_route_length (h1 h2 h3 v1 v2 : ℝ) : ℝ :=
  2 * ((h1 + h2 + h3) + (v1 + v2))

/-- Theorem stating the total length of the specific bike route -/
theorem bike_route_total_length :
  bike_route_length 4 7 2 6 7 = 52 := by
  sorry

#eval bike_route_length 4 7 2 6 7

end NUMINAMATH_CALUDE_bike_route_total_length_l2670_267037


namespace NUMINAMATH_CALUDE_investment_change_l2670_267092

theorem investment_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * 1.4
  let day2_value := day1_value * 0.75
  (day2_value - initial_value) / initial_value = 0.05 := by
sorry

end NUMINAMATH_CALUDE_investment_change_l2670_267092


namespace NUMINAMATH_CALUDE_train_length_calculation_l2670_267039

/-- Calculates the length of a train given the speeds of two trains, time to cross, and length of the other train -/
theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (l2 : ℝ) (h1 : v1 = 120) (h2 : v2 = 80) (h3 : t = 9) (h4 : l2 = 410.04) :
  let relative_speed := (v1 + v2) * 1000 / 3600
  let total_length := relative_speed * t
  let l1 := total_length - l2
  l1 = 90 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2670_267039


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2670_267087

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 48
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = -12
def centerLine (x y : ℝ) : Prop := x - y = 0

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  isTangentToLine1 : line1 center.1 center.2
  isTangentToLine2 : line2 center.1 center.2
  centerOnLine : centerLine center.1 center.2

-- Theorem statement
theorem circle_center_coordinates :
  ∀ (c : Circle), c.center = (18/7, 18/7) :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2670_267087


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l2670_267021

theorem quadratic_unique_solution :
  ∃! (k x : ℚ), 5 * k * x^2 + 30 * x + 10 = 0 ∧ k = 9/2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l2670_267021


namespace NUMINAMATH_CALUDE_sqrt_3_simplest_l2670_267003

-- Define a function to represent the concept of simplicity for square roots
def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (x = Real.sqrt y) → ¬∃ z : ℝ, z ≠ y ∧ Real.sqrt z = Real.sqrt y

-- State the theorem
theorem sqrt_3_simplest :
  is_simplest_sqrt (Real.sqrt 3) ∧
  ¬is_simplest_sqrt (Real.sqrt (a^2)) ∧
  ¬is_simplest_sqrt (Real.sqrt 0.3) ∧
  ¬is_simplest_sqrt (Real.sqrt 27) :=
sorry

end NUMINAMATH_CALUDE_sqrt_3_simplest_l2670_267003


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2670_267011

theorem arithmetic_calculations : 
  (78 - 14 * 2 = 50) ∧ 
  (500 - 296 - 104 = 100) ∧ 
  (360 - 300 / 5 = 300) ∧ 
  (84 / (16 / 4) = 21) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2670_267011


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l2670_267086

/-- Represents a right triangle with sides 6, 8, and 10, where the vertices
    are centers of mutually externally tangent circles -/
structure TriangleWithCircles where
  /-- Radius of the circle centered at the vertex opposite the side of length 8 -/
  r : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 6 -/
  s : ℝ
  /-- Radius of the circle centered at the vertex opposite the side of length 10 -/
  t : ℝ
  /-- The sum of radii of circles centered at vertices adjacent to side 6 equals 6 -/
  adj_6 : r + s = 6
  /-- The sum of radii of circles centered at vertices adjacent to side 8 equals 8 -/
  adj_8 : s + t = 8
  /-- The sum of radii of circles centered at vertices adjacent to side 10 equals 10 -/
  adj_10 : r + t = 10

/-- The sum of the areas of the three circles in a TriangleWithCircles is 56π -/
theorem sum_of_circle_areas (twc : TriangleWithCircles) :
  π * (twc.r^2 + twc.s^2 + twc.t^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l2670_267086


namespace NUMINAMATH_CALUDE_ned_remaining_pieces_l2670_267055

/-- The number of boxes Ned originally bought -/
def total_boxes : ℝ := 14.0

/-- The number of boxes Ned gave to his little brother -/
def given_boxes : ℝ := 7.0

/-- The number of pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of pieces Ned still had -/
def remaining_pieces : ℝ := (total_boxes - given_boxes) * pieces_per_box

theorem ned_remaining_pieces :
  remaining_pieces = 42.0 := by sorry

end NUMINAMATH_CALUDE_ned_remaining_pieces_l2670_267055


namespace NUMINAMATH_CALUDE_race_distance_l2670_267069

/-- Calculates the total distance of a race given the conditions --/
theorem race_distance (speed_a speed_b : ℝ) (head_start winning_margin : ℝ) : 
  speed_a / speed_b = 5 / 4 →
  head_start = 100 →
  winning_margin = 200 →
  (speed_a * ((head_start + winning_margin) / speed_b)) - head_start = 600 :=
by
  sorry


end NUMINAMATH_CALUDE_race_distance_l2670_267069


namespace NUMINAMATH_CALUDE_triangles_cover_two_thirds_l2670_267060

/-- Represents a tiling unit in the pattern -/
structure TilingUnit where
  /-- Side length of smaller shapes (triangles and squares) -/
  small_side : ℝ
  /-- Number of triangles in the unit -/
  num_triangles : ℕ
  /-- Number of squares in the unit -/
  num_squares : ℕ
  /-- Assertion that there are 2 triangles and 3 squares -/
  shape_count : num_triangles = 2 ∧ num_squares = 3
  /-- Assertion that all shapes have equal area -/
  equal_area : small_side^2 = 2 * (small_side^2 / 2)
  /-- Side length of the larger square formed by the unit -/
  large_side : ℝ
  /-- Assertion that large side is 3 times the small side -/
  side_relation : large_side = 3 * small_side

/-- Theorem stating that triangles cover 2/3 of the total area -/
theorem triangles_cover_two_thirds (u : TilingUnit) :
  (u.num_triangles * (u.small_side^2 / 2)) / u.large_side^2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangles_cover_two_thirds_l2670_267060


namespace NUMINAMATH_CALUDE_range_when_p_range_when_p_or_q_l2670_267025

/-- Proposition p: The range of the function y=log(x^2+2ax+2-a) is ℝ -/
def p (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*a*x + 2 - a)

/-- Proposition q: ∀x ∈ [0,1], x^2+2x+a ≥ 0 -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2*x + a ≥ 0

/-- If p is true, then a ≤ -2 or a ≥ 1 -/
theorem range_when_p (a : ℝ) : p a → a ≤ -2 ∨ a ≥ 1 := by
  sorry

/-- If p ∨ q is true, then a ≤ -2 or a ≥ 0 -/
theorem range_when_p_or_q (a : ℝ) : p a ∨ q a → a ≤ -2 ∨ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_when_p_range_when_p_or_q_l2670_267025


namespace NUMINAMATH_CALUDE_one_fourth_of_six_times_eight_l2670_267057

theorem one_fourth_of_six_times_eight : (1 / 4 : ℚ) * (6 * 8) = 12 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_six_times_eight_l2670_267057


namespace NUMINAMATH_CALUDE_sara_picked_37_peaches_l2670_267080

/-- The number of peaches Sara picked -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sara picked 37 peaches -/
theorem sara_picked_37_peaches (initial_peaches final_peaches : ℕ) 
  (h1 : initial_peaches = 24)
  (h2 : final_peaches = 61) :
  peaches_picked initial_peaches final_peaches = 37 := by
  sorry

#check sara_picked_37_peaches

end NUMINAMATH_CALUDE_sara_picked_37_peaches_l2670_267080


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l2670_267029

-- Define the given values
def num_steaks : ℕ := 4
def steak_weight : ℚ := 1/2
def steak_price_per_pound : ℚ := 15
def chicken_weight : ℚ := 3/2
def total_spent : ℚ := 42

-- Define the theorem
theorem chicken_price_per_pound :
  (total_spent - (num_steaks * steak_weight * steak_price_per_pound)) / chicken_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l2670_267029


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2670_267047

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) :
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2670_267047


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2670_267094

variable {n : Type*} [DecidableEq n] [Fintype n]

theorem matrix_equation_solution 
  (B : Matrix n n ℝ) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 • (1 : Matrix n n ℝ)) * (B - 5 • (1 : Matrix n n ℝ)) = 0) :
  B + 9 • B⁻¹ = 8 • (1 : Matrix n n ℝ) := by
sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2670_267094


namespace NUMINAMATH_CALUDE_correct_categorization_l2670_267012

def numbers : List ℚ := [2020, 1, -1, -2021, 1/2, 1/10, -1/3, -3/4, 0, 1/5]

def is_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n
def is_positive_integer (q : ℚ) : Prop := ∃ (n : ℕ), q = n ∧ n > 0
def is_negative_integer (q : ℚ) : Prop := ∃ (n : ℤ), q = n ∧ n < 0
def is_positive_fraction (q : ℚ) : Prop := q > 0 ∧ q < 1
def is_negative_fraction (q : ℚ) : Prop := q < 0 ∧ q > -1

def integers : List ℚ := [2020, 1, -1, -2021, 0]
def positive_integers : List ℚ := [2020, 1]
def negative_integers : List ℚ := [-1, -2021]
def positive_fractions : List ℚ := [1/2, 1/10, 1/5]
def negative_fractions : List ℚ := [-1/3, -3/4]

theorem correct_categorization :
  (∀ q ∈ integers, is_integer q) ∧
  (∀ q ∈ positive_integers, is_positive_integer q) ∧
  (∀ q ∈ negative_integers, is_negative_integer q) ∧
  (∀ q ∈ positive_fractions, is_positive_fraction q) ∧
  (∀ q ∈ negative_fractions, is_negative_fraction q) ∧
  (∀ q ∈ numbers, 
    (is_integer q → q ∈ integers) ∧
    (is_positive_integer q → q ∈ positive_integers) ∧
    (is_negative_integer q → q ∈ negative_integers) ∧
    (is_positive_fraction q → q ∈ positive_fractions) ∧
    (is_negative_fraction q → q ∈ negative_fractions)) :=
by sorry

end NUMINAMATH_CALUDE_correct_categorization_l2670_267012


namespace NUMINAMATH_CALUDE_solar_panel_installation_l2670_267082

theorem solar_panel_installation
  (total_homes : ℕ)
  (panels_per_home : ℕ)
  (shortage : ℕ)
  (h1 : total_homes = 20)
  (h2 : panels_per_home = 10)
  (h3 : shortage = 50)
  : (total_homes * panels_per_home - shortage) / panels_per_home = 15 := by
  sorry

end NUMINAMATH_CALUDE_solar_panel_installation_l2670_267082


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2670_267010

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 = 40 →
  a 3 + a 4 = 60 →
  a 5 + a 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2670_267010


namespace NUMINAMATH_CALUDE_triangle_theorem_l2670_267074

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = t.a * t.b

def angleBisectorIntersection (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the angle bisector condition
  True

def cdLength (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This represents CD = 2
  True

-- Theorem statement
theorem triangle_theorem (t : Triangle) (D : ℝ × ℝ) :
  satisfiesCondition t →
  angleBisectorIntersection t D →
  cdLength t D →
  (t.C = 2 * Real.pi / 3) ∧
  (∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧
    ∀ (a b : ℝ), a > 0 ∧ b > 0 → 2 * a + b ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2670_267074


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l2670_267008

theorem smallest_five_digit_multiple_of_18 : ∀ n : ℕ, 
  n ≥ 10000 ∧ n ≤ 99999 ∧ n % 18 = 0 → n ≥ 10008 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l2670_267008


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_25_l2670_267016

-- Define functions to convert numbers from different bases to base 10
def base8ToBase10 (n : ℕ) : ℕ := sorry

def base4ToBase10 (n : ℕ) : ℕ := sorry

def base5ToBase10 (n : ℕ) : ℕ := sorry

def base3ToBase10 (n : ℕ) : ℕ := sorry

-- Define the numbers in their respective bases
def num1 : ℕ := 254  -- in base 8
def den1 : ℕ := 14   -- in base 4
def num2 : ℕ := 132  -- in base 5
def den2 : ℕ := 26   -- in base 3

-- Theorem to prove
theorem sum_of_fractions_equals_25 :
  (base8ToBase10 num1 / base4ToBase10 den1) + (base5ToBase10 num2 / base3ToBase10 den2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_25_l2670_267016


namespace NUMINAMATH_CALUDE_officer_hopps_ticket_goal_l2670_267081

theorem officer_hopps_ticket_goal :
  let days_in_may : ℕ := 31
  let first_period_days : ℕ := 15
  let first_period_average : ℕ := 8
  let second_period_average : ℕ := 5
  let second_period_days : ℕ := days_in_may - first_period_days
  let first_period_tickets : ℕ := first_period_days * first_period_average
  let second_period_tickets : ℕ := second_period_days * second_period_average
  let total_tickets : ℕ := first_period_tickets + second_period_tickets
  total_tickets = 200 := by
sorry

end NUMINAMATH_CALUDE_officer_hopps_ticket_goal_l2670_267081


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2670_267054

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 3 + a 8 = 3 →
  a 1 + a 10 = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2670_267054


namespace NUMINAMATH_CALUDE_kiwis_for_18_apples_l2670_267014

-- Define the costs of fruits in terms of an arbitrary unit
variable (apple_cost banana_cost cucumber_cost kiwi_cost : ℚ)

-- Define the conditions
axiom apple_banana_ratio : 9 * apple_cost = 3 * banana_cost
axiom banana_cucumber_ratio : banana_cost = 2 * cucumber_cost
axiom cucumber_kiwi_ratio : 3 * cucumber_cost = 4 * kiwi_cost

-- Define the theorem
theorem kiwis_for_18_apples : 
  ∃ n : ℕ, (18 * apple_cost = n * kiwi_cost) ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_kiwis_for_18_apples_l2670_267014


namespace NUMINAMATH_CALUDE_max_min_f_l2670_267004

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem max_min_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    (∀ x ∈ Set.Icc 0 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    max = 0 ∧ min = -Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_l2670_267004


namespace NUMINAMATH_CALUDE_simplify_fourth_root_l2670_267005

theorem simplify_fourth_root (x y : ℕ+) :
  (2^6 * 3^5 * 5^2 : ℝ)^(1/4) = x * y^(1/4) →
  x + y = 306 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_root_l2670_267005


namespace NUMINAMATH_CALUDE_apple_packing_problem_l2670_267027

theorem apple_packing_problem (apples_per_crate : ℕ) (num_crates : ℕ) 
  (rotten_percentage : ℚ) (apples_per_box : ℕ) (available_boxes : ℕ) :
  apples_per_crate = 400 →
  num_crates = 35 →
  rotten_percentage = 11/100 →
  apples_per_box = 30 →
  available_boxes = 1000 →
  ∃ (boxes_needed : ℕ), 
    boxes_needed = 416 ∧ 
    boxes_needed * apples_per_box ≥ 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    (boxes_needed - 1) * apples_per_box < 
      (1 - rotten_percentage) * (apples_per_crate * num_crates) ∧
    boxes_needed ≤ available_boxes :=
by sorry

end NUMINAMATH_CALUDE_apple_packing_problem_l2670_267027


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l2670_267018

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the parabola at a given x -/
def Parabola.eval (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Calculates the derivative of the parabola at a given x -/
def Parabola.derivative (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem parabola_unique_coefficients :
  ∀ p : Parabola,
    p.eval 1 = 1 →                        -- Parabola passes through (1, 1)
    p.eval 2 = -1 →                       -- Parabola passes through (2, -1)
    p.derivative 2 = 1 →                  -- Tangent line at (2, -1) has slope 1
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l2670_267018


namespace NUMINAMATH_CALUDE_quadratic_common_root_theorem_l2670_267049

theorem quadratic_common_root_theorem (a b : ℕ+) :
  (∃ x : ℝ, (a - 1 : ℝ) * x^2 - (a^2 + 2 : ℝ) * x + (a^2 + 2*a : ℝ) = 0 ∧
             (b - 1 : ℝ) * x^2 - (b^2 + 2 : ℝ) * x + (b^2 + 2*b : ℝ) = 0) →
  (a^(b : ℕ) + b^(a : ℕ) : ℝ) / (a^(-(b : ℤ)) + b^(-(a : ℤ)) : ℝ) = 256 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_common_root_theorem_l2670_267049


namespace NUMINAMATH_CALUDE_camel_height_is_28_feet_l2670_267095

/-- The height of a hare in inches -/
def hare_height : ℕ := 14

/-- The factor by which a camel is taller than a hare -/
def camel_height_factor : ℕ := 24

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the height of a camel in feet -/
def camel_height_in_feet : ℕ :=
  (hare_height * camel_height_factor) / inches_per_foot

/-- Theorem stating that the camel's height is 28 feet -/
theorem camel_height_is_28_feet : camel_height_in_feet = 28 := by
  sorry

end NUMINAMATH_CALUDE_camel_height_is_28_feet_l2670_267095


namespace NUMINAMATH_CALUDE_some_number_value_l2670_267053

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * x * 49) : x = 315 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2670_267053


namespace NUMINAMATH_CALUDE_correct_total_crayons_l2670_267033

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 41

/-- The number of crayons Sam added to the drawer -/
def added_crayons : ℕ := 12

/-- The total number of crayons after Sam's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem correct_total_crayons : total_crayons = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_crayons_l2670_267033


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l2670_267072

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 6 = 0) (h2 : n ≥ 72) : ∃ (blue red green yellow : ℕ),
  blue = n / 2 ∧
  red = n / 3 ∧
  green = 12 ∧
  yellow = n - (blue + red + green) ∧
  yellow = 0 ∧
  blue + red + green + yellow = n :=
sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l2670_267072


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l2670_267026

/-- A parabola is defined by the equation y^2 = -4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -4 * p.1}

/-- The focus of a parabola is a point from which all points on the parabola are equidistant -/
def FocusOfParabola (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem focus_of_specific_parabola :
  FocusOfParabola Parabola = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l2670_267026


namespace NUMINAMATH_CALUDE_quadratic_expansion_sum_l2670_267096

theorem quadratic_expansion_sum (a b : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 3 = (x - 1)^2 + a*(x - 1) + b) → a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expansion_sum_l2670_267096


namespace NUMINAMATH_CALUDE_lending_time_problem_l2670_267046

/-- The problem of finding the lending time for the second part of a sum --/
theorem lending_time_problem (total_sum : ℝ) (second_part : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) :
  total_sum = 2743 →
  second_part = 1688 →
  rate1 = 0.03 →
  time1 = 8 →
  rate2 = 0.05 →
  (total_sum - second_part) * rate1 * time1 = second_part * rate2 * 3 :=
by
  sorry

#check lending_time_problem

end NUMINAMATH_CALUDE_lending_time_problem_l2670_267046
