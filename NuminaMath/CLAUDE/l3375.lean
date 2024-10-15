import Mathlib

namespace NUMINAMATH_CALUDE_set_operations_and_range_l3375_337589

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_range :
  (∃ a : ℝ,
    (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
    ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
    (A ∩ C a = A → a ≥ 7)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l3375_337589


namespace NUMINAMATH_CALUDE_dress_cost_calculation_l3375_337532

/-- The cost of a dress in dinars -/
def dress_cost : ℚ := 10/9

/-- The monthly pay in dinars (excluding the dress) -/
def monthly_pay : ℚ := 10

/-- The number of days in a month -/
def days_in_month : ℕ := 30

/-- The number of days worked to earn a dress -/
def days_worked : ℕ := 3

/-- Theorem stating the cost of the dress -/
theorem dress_cost_calculation :
  dress_cost = (monthly_pay + dress_cost) * days_worked / days_in_month :=
by sorry

end NUMINAMATH_CALUDE_dress_cost_calculation_l3375_337532


namespace NUMINAMATH_CALUDE_product_of_roots_l3375_337590

theorem product_of_roots (x : ℝ) (hx : x + 16 / x = 12) : 
  ∃ y : ℝ, y + 16 / y = 12 ∧ x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l3375_337590


namespace NUMINAMATH_CALUDE_three_lines_intersect_once_l3375_337563

/-- A parabola defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is outside a parabola -/
def is_outside (pt : Point) (par : Parabola) : Prop :=
  pt.y^2 > 2 * par.p * pt.x

/-- Predicate to check if a line intersects a parabola at exactly one point -/
def intersects_once (l : Line) (par : Parabola) : Prop :=
  sorry -- Definition of intersection at exactly one point

/-- The main theorem -/
theorem three_lines_intersect_once (par : Parabola) (M : Point) 
  (h_outside : is_outside M par) : 
  ∃ (l₁ l₂ l₃ : Line), 
    (∀ l : Line, (intersects_once l par ∧ l.a * M.x + l.b * M.y + l.c = 0) ↔ 
      (l = l₁ ∨ l = l₂ ∨ l = l₃)) :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersect_once_l3375_337563


namespace NUMINAMATH_CALUDE_is_671st_term_l3375_337522

/-- The arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- 2011 is the 671st term in the arithmetic sequence -/
theorem is_671st_term : arithmetic_sequence 671 = 2011 := by sorry

end NUMINAMATH_CALUDE_is_671st_term_l3375_337522


namespace NUMINAMATH_CALUDE_least_k_squared_divisible_by_240_l3375_337592

theorem least_k_squared_divisible_by_240 : 
  ∃ k : ℕ+, k.val = 60 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(240 ∣ m.val^2)) ∧
  (240 ∣ k.val^2) := by
  sorry

end NUMINAMATH_CALUDE_least_k_squared_divisible_by_240_l3375_337592


namespace NUMINAMATH_CALUDE_sine_product_equality_l3375_337521

theorem sine_product_equality : 
  3.438 * Real.sin (84 * π / 180) * Real.sin (24 * π / 180) * 
  Real.sin (48 * π / 180) * Real.sin (12 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_equality_l3375_337521


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3375_337575

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - 3*x + 2 < 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3375_337575


namespace NUMINAMATH_CALUDE_circle_radius_zero_l3375_337544

theorem circle_radius_zero (x y : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 - 10*y + 41 = 0) → 
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2) ∧ r = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_l3375_337544


namespace NUMINAMATH_CALUDE_factor_implies_coefficients_l3375_337559

/-- If (x + 5) is a factor of x^4 - mx^3 + nx^2 - px + q, then m = 0, n = 0, p = 0, and q = -625 -/
theorem factor_implies_coefficients (m n p q : ℝ) : 
  (∀ x : ℝ, (x + 5) ∣ (x^4 - m*x^3 + n*x^2 - p*x + q)) →
  (m = 0 ∧ n = 0 ∧ p = 0 ∧ q = -625) := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_coefficients_l3375_337559


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l3375_337584

theorem fixed_point_of_exponential_shift (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l3375_337584


namespace NUMINAMATH_CALUDE_bread_butter_price_ratio_l3375_337577

/-- Proves that the ratio of bread price to butter price is 1:2 given the problem conditions --/
theorem bread_butter_price_ratio : 
  ∀ (butter bread cheese tea : ℝ),
  butter + bread + cheese + tea = 21 →
  butter = 0.8 * cheese →
  tea = 2 * cheese →
  tea = 10 →
  bread / butter = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_bread_butter_price_ratio_l3375_337577


namespace NUMINAMATH_CALUDE_girls_in_class_l3375_337524

theorem girls_in_class (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 35 →
  boys + girls = total →
  girls = (2 / 5 : ℚ) * boys →
  girls = 10 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l3375_337524


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l3375_337525

theorem chinese_chess_draw_probability 
  (p_xiao_ming_not_lose : ℝ)
  (p_xiao_dong_lose : ℝ)
  (h1 : p_xiao_ming_not_lose = 3/4)
  (h2 : p_xiao_dong_lose = 1/2) :
  p_xiao_ming_not_lose - p_xiao_dong_lose = 1/4 := by
sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l3375_337525


namespace NUMINAMATH_CALUDE_average_speed_calculation_toms_trip_average_speed_l3375_337557

theorem average_speed_calculation (total_distance : Real) (first_part_distance : Real) 
  (first_part_speed : Real) (second_part_speed : Real) : Real :=
  let second_part_distance := total_distance - first_part_distance
  let first_part_time := first_part_distance / first_part_speed
  let second_part_time := second_part_distance / second_part_speed
  let total_time := first_part_time + second_part_time
  total_distance / total_time

theorem toms_trip_average_speed : 
  average_speed_calculation 60 12 24 48 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_toms_trip_average_speed_l3375_337557


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3375_337594

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (l m : Line) (α : Plane) :
  l ≠ m →  -- l and m are different lines
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3375_337594


namespace NUMINAMATH_CALUDE_divisor_pairs_count_l3375_337512

theorem divisor_pairs_count (n : ℕ) (h : n = 2^6 * 3^3) :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card = 28 :=
by sorry

end NUMINAMATH_CALUDE_divisor_pairs_count_l3375_337512


namespace NUMINAMATH_CALUDE_trig_product_equality_l3375_337529

theorem trig_product_equality : 
  Real.sin (-15 * Real.pi / 6) * Real.cos (20 * Real.pi / 3) * Real.tan (-7 * Real.pi / 6) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equality_l3375_337529


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l3375_337558

theorem lcm_gcf_problem (n : ℕ+) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 12 → n = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l3375_337558


namespace NUMINAMATH_CALUDE_cost_per_serving_l3375_337531

/-- The cost per serving of a meal given the costs of ingredients and number of servings -/
theorem cost_per_serving 
  (pasta_cost : ℚ) 
  (sauce_cost : ℚ) 
  (meatballs_cost : ℚ) 
  (num_servings : ℕ) 
  (h1 : pasta_cost = 1)
  (h2 : sauce_cost = 2)
  (h3 : meatballs_cost = 5)
  (h4 : num_servings = 8) :
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings = 1 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_serving_l3375_337531


namespace NUMINAMATH_CALUDE_company_reduction_l3375_337514

/-- The original number of employees before reductions -/
def original_employees : ℕ := 344

/-- The number of employees after both reductions -/
def final_employees : ℕ := 263

/-- The reduction factor after the first quarter -/
def first_reduction : ℚ := 9/10

/-- The reduction factor after the second quarter -/
def second_reduction : ℚ := 85/100

theorem company_reduction :
  ⌊(second_reduction * first_reduction * original_employees : ℚ)⌋ = final_employees := by
  sorry

end NUMINAMATH_CALUDE_company_reduction_l3375_337514


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l3375_337542

/-- Calculates the total number of orange sacks kept after a given number of harvest days. -/
def total_sacks_kept (daily_harvest : ℕ) (daily_discard : ℕ) (harvest_days : ℕ) : ℕ :=
  (daily_harvest - daily_discard) * harvest_days

/-- Proves that given the specified harvest conditions, the total number of sacks kept is 1425. -/
theorem orange_harvest_theorem :
  total_sacks_kept 150 135 95 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l3375_337542


namespace NUMINAMATH_CALUDE_triangle_side_b_l3375_337553

theorem triangle_side_b (a b c : ℝ) (A B C : ℝ) : 
  a = 8 → B = π/3 → C = 5*π/12 → b = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_l3375_337553


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_n_arithmetic_sequence_sum_17_arithmetic_sequence_sum_13_l3375_337541

-- Problem 1
theorem arithmetic_sequence_sum_n (a : ℕ → ℤ) (n : ℕ) :
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence
  a 4 = 10 →
  a 10 = -2 →
  (n : ℤ) * (a 1 + a n) / 2 = 60 →
  n = 5 ∨ n = 6 := by sorry

-- Problem 2
theorem arithmetic_sequence_sum_17 (a : ℕ → ℤ) :
  a 1 = -7 →
  (∀ n, a (n + 1) = a n + 2) →
  (17 : ℤ) * (a 1 + a 17) / 2 = 153 := by sorry

-- Problem 3
theorem arithmetic_sequence_sum_13 (a : ℕ → ℤ) :
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence
  a 2 + a 7 + a 12 = 24 →
  (13 : ℤ) * (a 1 + a 13) / 2 = 104 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_n_arithmetic_sequence_sum_17_arithmetic_sequence_sum_13_l3375_337541


namespace NUMINAMATH_CALUDE_jane_sequins_count_l3375_337561

/-- The number of rows of blue sequins -/
def blue_rows : ℕ := 6

/-- The number of blue sequins in each row -/
def blue_per_row : ℕ := 8

/-- The number of rows of purple sequins -/
def purple_rows : ℕ := 5

/-- The number of purple sequins in each row -/
def purple_per_row : ℕ := 12

/-- The number of rows of green sequins -/
def green_rows : ℕ := 9

/-- The number of green sequins in each row -/
def green_per_row : ℕ := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : ℕ := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end NUMINAMATH_CALUDE_jane_sequins_count_l3375_337561


namespace NUMINAMATH_CALUDE_custom_mul_four_three_l3375_337538

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem stating that 4 * 3 = 37 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 37 := by sorry

end NUMINAMATH_CALUDE_custom_mul_four_three_l3375_337538


namespace NUMINAMATH_CALUDE_mean_score_of_all_students_l3375_337587

/-- Calculates the mean score of all students given the mean scores of two classes and the ratio of students in those classes. -/
theorem mean_score_of_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (morning_students : ℕ)
  (afternoon_students : ℕ)
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : morning_students = 2 * afternoon_students / 5) :
  let total_students := morning_students + afternoon_students
  let total_score := morning_mean * morning_students + afternoon_mean * afternoon_students
  total_score / total_students = 79 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_score_of_all_students_l3375_337587


namespace NUMINAMATH_CALUDE_polygon_sides_l3375_337566

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + 360 = 1800) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3375_337566


namespace NUMINAMATH_CALUDE_factorial_simplification_l3375_337500

theorem factorial_simplification : 
  Nat.factorial 15 / (Nat.factorial 11 + 3 * Nat.factorial 10) = 25740 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3375_337500


namespace NUMINAMATH_CALUDE_x₃_value_l3375_337510

noncomputable def x₃ (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := Real.exp x₁
  let y₂ := Real.exp x₂
  let yC := (2/3) * y₁ + (1/3) * y₂
  Real.log ((2/3) + (1/3) * Real.exp 2)

theorem x₃_value :
  let x₁ : ℝ := 0
  let x₂ : ℝ := 2
  let f : ℝ → ℝ := Real.exp
  x₃ x₁ x₂ = Real.log ((2/3) + (1/3) * Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_x₃_value_l3375_337510


namespace NUMINAMATH_CALUDE_town_population_problem_l3375_337534

theorem town_population_problem (original_population : ℝ) : 
  (original_population * 1.15 * 0.87 = original_population - 50) → 
  original_population = 100000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3375_337534


namespace NUMINAMATH_CALUDE_chord_midpoint_line_l3375_337535

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_midpoint_line :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →  -- A and B are on the ellipse
  P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →  -- P is the midpoint of AB
  line_equation A.1 A.2 ∧ line_equation B.1 B.2  -- A and B satisfy the line equation
  := by sorry

end NUMINAMATH_CALUDE_chord_midpoint_line_l3375_337535


namespace NUMINAMATH_CALUDE_div_by_eleven_iff_alternating_sum_div_by_eleven_l3375_337564

/-- Calculates the alternating sum of digits of a natural number -/
def alternatingDigitSum (n : ℕ) : ℤ :=
  sorry

/-- Proves the equivalence of divisibility by 11 and divisibility of alternating digit sum by 11 -/
theorem div_by_eleven_iff_alternating_sum_div_by_eleven (n : ℕ) :
  11 ∣ n ↔ 11 ∣ (alternatingDigitSum n) :=
sorry

end NUMINAMATH_CALUDE_div_by_eleven_iff_alternating_sum_div_by_eleven_l3375_337564


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3375_337574

/-- Given a population of 1000 people and a simple random sampling method where
    the probability of each person being selected is 0.2, prove that the sample size is 200. -/
theorem sample_size_calculation (population : ℕ) (prob : ℝ) (sample_size : ℕ) :
  population = 1000 →
  prob = 0.2 →
  sample_size = population * prob →
  sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l3375_337574


namespace NUMINAMATH_CALUDE_parabola_tangent_ellipse_l3375_337513

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := 4*x - 4

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

theorem parabola_tangent_ellipse :
  -- Conditions
  (∀ x, parabola x = x^2) →
  (parabola 2 = 4) →
  (tangent_line 2 = 4) →
  (tangent_line 1 = 0) →
  (tangent_line 0 = -4) →
  -- Conclusion
  ellipse (Real.sqrt 17) 4 1 0 ∧ ellipse (Real.sqrt 17) 4 0 (-4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_ellipse_l3375_337513


namespace NUMINAMATH_CALUDE_cd_purchase_total_l3375_337548

/-- The total cost of purchasing 3 copies each of three different CDs -/
def total_cost (price1 price2 price3 : ℕ) : ℕ :=
  3 * (price1 + price2 + price3)

theorem cd_purchase_total : total_cost 100 50 85 = 705 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_total_l3375_337548


namespace NUMINAMATH_CALUDE_ellipse_curve_l3375_337543

-- Define the set of points (x,y) parametrized by t
def ellipse_points : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, p.1 = 2 * Real.cos t ∧ p.2 = 3 * Real.sin t}

-- Define the standard form equation of an ellipse
def is_ellipse (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ p ∈ S, (p.1 / a)^2 + (p.2 / b)^2 = 1

-- Theorem statement
theorem ellipse_curve : is_ellipse ellipse_points := by
  sorry

end NUMINAMATH_CALUDE_ellipse_curve_l3375_337543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3375_337570

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 6 = 2 →
  a 8 = 4 →
  a 10 + a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3375_337570


namespace NUMINAMATH_CALUDE_min_overlap_percentage_l3375_337509

theorem min_overlap_percentage (laptop_users smartphone_users : ℚ) 
  (h1 : laptop_users = 90/100) 
  (h2 : smartphone_users = 80/100) : 
  (laptop_users + smartphone_users - 1 : ℚ) ≥ 70/100 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_percentage_l3375_337509


namespace NUMINAMATH_CALUDE_triangle_side_length_l3375_337591

/-- Given a triangle ABC with area 3√3/4, side a = 3, and angle B = π/3, prove that side b = √7 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) →  -- Area formula
  (a = 3) →  -- Given side length
  (B = π/3) →  -- Given angle
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →  -- Law of cosines
  (b = Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3375_337591


namespace NUMINAMATH_CALUDE_lebesgue_stieltjes_countable_zero_l3375_337540

-- Define the Lebesgue-Stieltjes measure
def LebesgueStieltjesMeasure (ν : Set ℝ → ℝ) : Prop :=
  -- Add properties of Lebesgue-Stieltjes measure here
  sorry

-- Define continuous generalized distribution function
def ContinuousGeneralizedDistributionFunction (F : ℝ → ℝ) : Prop :=
  -- Add properties of continuous generalized distribution function here
  sorry

-- Define the correspondence between ν and F
def CorrespondsTo (ν : Set ℝ → ℝ) (F : ℝ → ℝ) : Prop :=
  -- Add the correspondence condition here
  sorry

-- Theorem statement
theorem lebesgue_stieltjes_countable_zero 
  (ν : Set ℝ → ℝ) 
  (F : ℝ → ℝ) 
  (A : Set ℝ) :
  LebesgueStieltjesMeasure ν →
  ContinuousGeneralizedDistributionFunction F →
  CorrespondsTo ν F →
  (Set.Countable A ∨ A = ∅) →
  ν A = 0 :=
sorry

end NUMINAMATH_CALUDE_lebesgue_stieltjes_countable_zero_l3375_337540


namespace NUMINAMATH_CALUDE_map_distance_to_real_distance_l3375_337501

/-- Proves that for a map with scale 1:500,000, a 4 cm distance on the map represents 20 km in reality -/
theorem map_distance_to_real_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 500000)
  (h_map_distance : map_distance = 4)
  : map_distance * scale * 100000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_to_real_distance_l3375_337501


namespace NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l3375_337567

theorem base_10_to_base_8_conversion : 
  (2 * 8^2 + 3 * 8^1 + 5 * 8^0 : ℕ) = 157 := by sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_conversion_l3375_337567


namespace NUMINAMATH_CALUDE_candy_problem_l3375_337552

theorem candy_problem (n : ℕ) (x : ℕ) (h1 : n > 1) (h2 : x > 1) 
  (h3 : ∀ i : ℕ, i < n → x = (n - 1) * x - 7) : 
  n * x = 21 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l3375_337552


namespace NUMINAMATH_CALUDE_fraction_addition_l3375_337537

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3375_337537


namespace NUMINAMATH_CALUDE_andrew_expenses_l3375_337511

def game_night_expenses (game_count : Nat) 
  (game_cost_1 : Nat) (game_count_1 : Nat)
  (game_cost_2 : Nat) (game_count_2 : Nat)
  (game_cost_3 : Nat) (game_count_3 : Nat)
  (snack_cost : Nat) (drink_cost : Nat) : Nat :=
  game_cost_1 * game_count_1 + game_cost_2 * game_count_2 + game_cost_3 * game_count_3 + snack_cost + drink_cost

theorem andrew_expenses : 
  game_night_expenses 7 900 3 1250 2 1500 2 2500 2000 = 12700 := by
  sorry

end NUMINAMATH_CALUDE_andrew_expenses_l3375_337511


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3375_337503

theorem line_slope_intercept_product (m b : ℝ) : 
  m = 3/4 → b = -2 → m * b < -3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3375_337503


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3375_337504

theorem complex_fraction_simplification (i : ℂ) :
  i^2 = -1 →
  (2 + i) * (3 - 4*i) / (2 - i) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3375_337504


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l3375_337549

theorem simplify_fraction_expression (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  1 / (2 * x) - 1 / (x + y) * ((x + y) / (2 * x) - x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l3375_337549


namespace NUMINAMATH_CALUDE_trip_length_proof_average_efficiency_proof_l3375_337579

/-- The total length of the trip in miles -/
def trip_length : ℝ := 180

/-- The distance the car ran on battery -/
def battery_distance : ℝ := 60

/-- The rate of gasoline consumption in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- The average fuel efficiency for the entire trip in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Theorem stating that the trip length satisfies the given conditions -/
theorem trip_length_proof :
  trip_length = battery_distance + 
  (trip_length - battery_distance) * gasoline_rate * average_efficiency :=
by sorry

/-- Theorem stating that the average efficiency is correct -/
theorem average_efficiency_proof :
  average_efficiency = trip_length / (gasoline_rate * (trip_length - battery_distance)) :=
by sorry

end NUMINAMATH_CALUDE_trip_length_proof_average_efficiency_proof_l3375_337579


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l3375_337565

theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 432 →
  total_cost = 54 →
  (total_cost / (total_miles / miles_per_gallon)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l3375_337565


namespace NUMINAMATH_CALUDE_ages_solution_l3375_337560

/-- Represents the ages of three individuals -/
structure Ages where
  shekhar : ℚ
  shobha : ℚ
  kapil : ℚ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The ratio of ages is 4:3:2
  ages.shekhar / ages.shobha = 4 / 3 ∧
  ages.shekhar / ages.kapil = 2 ∧
  -- In 10 years, Kapil's age will equal Shekhar's present age
  ages.kapil + 10 = ages.shekhar ∧
  -- Shekhar's age will be 30 in 8 years
  ages.shekhar + 8 = 30

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.shekhar = 22 ∧ ages.shobha = 33/2 ∧ ages.kapil = 10 := by
  sorry


end NUMINAMATH_CALUDE_ages_solution_l3375_337560


namespace NUMINAMATH_CALUDE_concert_expense_l3375_337517

def ticket_price : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem concert_expense : 
  ticket_price * (tickets_for_friends + extra_tickets) = 60 := by
  sorry

end NUMINAMATH_CALUDE_concert_expense_l3375_337517


namespace NUMINAMATH_CALUDE_repeating_decimal_equality_l3375_337572

theorem repeating_decimal_equality (a : ℕ) : 
  1 ≤ a ∧ a ≤ 9 → (0.1 * a : ℚ) = 1 / a → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equality_l3375_337572


namespace NUMINAMATH_CALUDE_diagonal_length_16_12_rectangle_l3375_337576

/-- The length of a diagonal in a 16 cm by 12 cm rectangle is 20 cm -/
theorem diagonal_length_16_12_rectangle : 
  ∀ (a b : ℝ), a = 16 ∧ b = 12 → Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_16_12_rectangle_l3375_337576


namespace NUMINAMATH_CALUDE_fort_sixty_percent_complete_l3375_337598

/-- Calculates the percentage of fort completion given the required sticks, 
    sticks collected per week, and number of weeks collecting. -/
def fort_completion_percentage 
  (required_sticks : ℕ) 
  (sticks_per_week : ℕ) 
  (weeks_collecting : ℕ) : ℚ :=
  (sticks_per_week * weeks_collecting : ℚ) / required_sticks * 100

/-- Theorem stating that given the specific conditions, 
    the fort completion percentage is 60%. -/
theorem fort_sixty_percent_complete : 
  fort_completion_percentage 400 3 80 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fort_sixty_percent_complete_l3375_337598


namespace NUMINAMATH_CALUDE_rectangle_19_65_parts_l3375_337562

/-- Calculates the number of parts a rectangle is divided into when split into unit squares and crossed by a diagonal -/
def rectangle_parts (width : ℕ) (height : ℕ) : ℕ :=
  let unit_squares := width * height
  let diagonal_crossings := width + height - Nat.gcd width height
  unit_squares + diagonal_crossings

/-- The number of parts a 19 cm by 65 cm rectangle is divided into when split into 1 cm squares and crossed by a diagonal -/
theorem rectangle_19_65_parts : rectangle_parts 19 65 = 1318 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_19_65_parts_l3375_337562


namespace NUMINAMATH_CALUDE_stone_falling_in_water_l3375_337568

/-- Stone falling in water problem -/
theorem stone_falling_in_water
  (stone_density : ℝ)
  (lake_depth : ℝ)
  (gravity : ℝ)
  (water_density : ℝ)
  (h_stone_density : stone_density = 2.1)
  (h_lake_depth : lake_depth = 8.5)
  (h_gravity : gravity = 980.8)
  (h_water_density : water_density = 1.0) :
  ∃ (time velocity : ℝ),
    (abs (time - 1.82) < 0.01) ∧
    (abs (velocity - 935) < 1) ∧
    time = Real.sqrt ((2 * lake_depth * 100) / ((stone_density - water_density) / stone_density * gravity)) ∧
    velocity = ((stone_density - water_density) / stone_density * gravity) * time :=
  sorry


end NUMINAMATH_CALUDE_stone_falling_in_water_l3375_337568


namespace NUMINAMATH_CALUDE_no_real_roots_implications_l3375_337515

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_real_roots_implications
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implications_l3375_337515


namespace NUMINAMATH_CALUDE_factory_work_hours_l3375_337533

/-- Calculates the number of hours a factory works per day given its production rates and total output. -/
theorem factory_work_hours 
  (refrigerators_per_hour : ℕ)
  (extra_coolers : ℕ)
  (total_products : ℕ)
  (days : ℕ)
  (h : refrigerators_per_hour = 90)
  (h' : extra_coolers = 70)
  (h'' : total_products = 11250)
  (h''' : days = 5) :
  (total_products / (days * (refrigerators_per_hour + (refrigerators_per_hour + extra_coolers)))) = 9 :=
by sorry

end NUMINAMATH_CALUDE_factory_work_hours_l3375_337533


namespace NUMINAMATH_CALUDE_initial_limes_count_l3375_337573

def limes_given_to_sara : ℕ := 4
def limes_dan_has_now : ℕ := 5

theorem initial_limes_count : 
  limes_given_to_sara + limes_dan_has_now = 9 := by sorry

end NUMINAMATH_CALUDE_initial_limes_count_l3375_337573


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3375_337519

-- Define points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

-- Define vector a as a function of x
def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_equation_solution :
  ∃ x : ℝ, a x = AB ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3375_337519


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l3375_337518

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 7 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l3375_337518


namespace NUMINAMATH_CALUDE_milk_left_l3375_337523

theorem milk_left (initial_milk : ℚ) (given_milk : ℚ) (remaining_milk : ℚ) : 
  initial_milk = 5 → given_milk = 18 / 7 → remaining_milk = initial_milk - given_milk → 
  remaining_milk = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_l3375_337523


namespace NUMINAMATH_CALUDE_age_ratio_is_two_l3375_337539

/-- The age difference between Yuan and David -/
def age_difference : ℕ := 7

/-- David's age -/
def david_age : ℕ := 7

/-- Yuan's age -/
def yuan_age : ℕ := david_age + age_difference

/-- The ratio of Yuan's age to David's age -/
def age_ratio : ℚ := yuan_age / david_age

theorem age_ratio_is_two : age_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_l3375_337539


namespace NUMINAMATH_CALUDE_a_4_equals_11_l3375_337582

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

def a (n : ℕ+) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_4_equals_11 : a 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_a_4_equals_11_l3375_337582


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l3375_337505

/-- Given a line y = kx + m intersecting a parabola y^2 = 4x at two points,
    if the midpoint of these intersection points has y-coordinate 2,
    then k = 1. -/
theorem line_parabola_intersection (k m x₀ : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- Line equation
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = x₀ ∧
    (y₁ + y₂) / 2 = 2) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l3375_337505


namespace NUMINAMATH_CALUDE_vector_combination_equals_result_l3375_337520

/-- Given vectors a, b, and c in ℝ³, prove that 2a - 3b + 4c equals the specified result -/
theorem vector_combination_equals_result (a b c : ℝ × ℝ × ℝ) 
  (ha : a = (3, 5, -1)) 
  (hb : b = (2, 2, 3)) 
  (hc : c = (4, -1, -3)) : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -23) := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_equals_result_l3375_337520


namespace NUMINAMATH_CALUDE_area_of_triangle_PF₁F₂_l3375_337580

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry

-- Assert that P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Define the distances
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Assert the ratio of distances
axiom distance_ratio : PF₁ / PF₂ = 2

-- Theorem to prove
theorem area_of_triangle_PF₁F₂ : 
  let triangle_area := sorry
  triangle_area = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_PF₁F₂_l3375_337580


namespace NUMINAMATH_CALUDE_taxi_count_2008_l3375_337546

/-- Represents the number of taxis (in thousands) at the end of a given year -/
def taxiCount : ℕ → ℝ
| 0 => 100  -- End of 2005
| n + 1 => taxiCount n * 1.1 - 20  -- Subsequent years

/-- The year we're interested in (2008 is 3 years after 2005) -/
def targetYear : ℕ := 3

theorem taxi_count_2008 :
  12 ≤ taxiCount targetYear ∧ taxiCount targetYear < 13 := by
  sorry

end NUMINAMATH_CALUDE_taxi_count_2008_l3375_337546


namespace NUMINAMATH_CALUDE_valid_fraction_l3375_337526

theorem valid_fraction (x : ℝ) (h : x ≠ 3) : ∃ (f : ℝ → ℝ), f x = 1 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_valid_fraction_l3375_337526


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3375_337508

def age_problem (A_current B_current : ℕ) : Prop :=
  B_current = 37 ∧
  A_current = B_current + 7 ∧
  (A_current + 10) / (B_current - 10) = 2

theorem age_ratio_proof :
  ∃ A_current B_current : ℕ, age_problem A_current B_current :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3375_337508


namespace NUMINAMATH_CALUDE_f_of_one_eq_two_l3375_337596

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + |x - 2|

-- State the theorem
theorem f_of_one_eq_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_eq_two_l3375_337596


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l3375_337530

theorem purely_imaginary_complex_equation (z : ℂ) (a : ℝ) : 
  (z.re = 0) → ((2 - I) * z = a + I) → (a = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l3375_337530


namespace NUMINAMATH_CALUDE_perpendicular_lines_in_parallel_planes_l3375_337551

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (lies_in : Line → Plane → Prop)
variable (not_lies_on : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_in_parallel_planes
  (α β : Plane) (l m : Line)
  (h1 : lies_in l α)
  (h2 : not_lies_on m α)
  (h3 : parallel α β)
  (h4 : perpendicular m β) :
  line_perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_in_parallel_planes_l3375_337551


namespace NUMINAMATH_CALUDE_triangle_area_l3375_337502

/-- The area of a triangle with sides 9, 40, and 41 is 180 -/
theorem triangle_area : ℝ → Prop := fun area =>
  let a := 9
  let b := 40
  let c := 41
  (a * a + b * b = c * c) → (area = (1 / 2) * a * b)

#check triangle_area 180

end NUMINAMATH_CALUDE_triangle_area_l3375_337502


namespace NUMINAMATH_CALUDE_tire_price_proof_l3375_337556

theorem tire_price_proof :
  let regular_price : ℝ := 90
  let third_tire_price : ℝ := 5
  let total_cost : ℝ := 185
  (2 * regular_price + third_tire_price = total_cost) →
  regular_price = 90 := by
sorry

end NUMINAMATH_CALUDE_tire_price_proof_l3375_337556


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3375_337588

theorem arithmetic_progression_x_value :
  ∀ (x : ℚ),
  let a₁ := 2 * x - 4
  let a₂ := 2 * x + 2
  let a₃ := 4 * x + 1
  (a₂ - a₁ = a₃ - a₂) → x = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3375_337588


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3375_337545

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 42*z + 350 ≤ 4 ↔ 21 - Real.sqrt 95 ≤ z ∧ z ≤ 21 + Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3375_337545


namespace NUMINAMATH_CALUDE_candy_division_l3375_337585

theorem candy_division (total_candy : ℚ) (num_piles : ℕ) (piles_for_carlos : ℕ) :
  total_candy = 75 / 7 →
  num_piles = 5 →
  piles_for_carlos = 2 →
  piles_for_carlos * (total_candy / num_piles) = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l3375_337585


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3375_337516

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3375_337516


namespace NUMINAMATH_CALUDE_function_determination_l3375_337583

theorem function_determination (f : ℝ → ℝ) 
  (h0 : f 0 = 1) 
  (h1 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) : 
  ∀ x : ℝ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3375_337583


namespace NUMINAMATH_CALUDE_garden_area_l3375_337550

/-- The area of a rectangular garden with dimensions 90 cm and 4.5 meters is 4.05 square meters. -/
theorem garden_area : 
  let length_cm : ℝ := 90
  let width_m : ℝ := 4.5
  let length_m : ℝ := length_cm / 100
  let area_m2 : ℝ := length_m * width_m
  area_m2 = 4.05 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l3375_337550


namespace NUMINAMATH_CALUDE_floretta_balloon_count_l3375_337554

/-- The number of water balloons Floretta is left with after Milly takes extra -/
def florettas_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (extra_taken : ℕ) : ℕ :=
  (total_packs * balloons_per_pack) / 2 - extra_taken

/-- Theorem stating the number of balloons Floretta is left with -/
theorem floretta_balloon_count :
  florettas_balloons 5 6 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_floretta_balloon_count_l3375_337554


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3375_337569

/-- Given a quadratic inequality ax^2 + bx - 2 > 0 with solution set (1,4), prove a + b = 2 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x - 2 > 0 ↔ 1 < x ∧ x < 4) → 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3375_337569


namespace NUMINAMATH_CALUDE_pages_read_difference_l3375_337593

/-- Given a book with 270 pages, prove that reading 2/3 of it results in 90 more pages read than left to read. -/
theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 270 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages : ℚ) - (total_pages - fraction_read * total_pages : ℚ) = 90 :=
by
  sorry

#check pages_read_difference

end NUMINAMATH_CALUDE_pages_read_difference_l3375_337593


namespace NUMINAMATH_CALUDE_m_range_theorem_l3375_337586

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 - m * x + 1 < 0

-- Define proposition q
def q (m : ℝ) : Prop := (m - 1) * (3 - m) < 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (0 < m ∧ m ≤ 1) ∨ (3 ≤ m ∧ m < 4)

-- Theorem statement
theorem m_range_theorem (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3375_337586


namespace NUMINAMATH_CALUDE_unique_k_term_l3375_337506

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n^2 - 7*n

/-- The k-th term of the sequence -/
def a (k : ℕ) : ℤ := S k - S (k-1)

theorem unique_k_term (k : ℕ) (h : 9 < a k ∧ a k < 12) : k = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_term_l3375_337506


namespace NUMINAMATH_CALUDE_tree_ratio_l3375_337578

/-- The number of streets in the neighborhood -/
def num_streets : ℕ := 18

/-- The number of plum trees planted -/
def num_plum_trees : ℕ := 3

/-- The number of pear trees planted -/
def num_pear_trees : ℕ := 3

/-- The number of apricot trees planted -/
def num_apricot_trees : ℕ := 3

/-- Theorem stating that the ratio of plum trees to pear trees to apricot trees is 1:1:1 -/
theorem tree_ratio : 
  num_plum_trees = num_pear_trees ∧ num_pear_trees = num_apricot_trees :=
sorry

end NUMINAMATH_CALUDE_tree_ratio_l3375_337578


namespace NUMINAMATH_CALUDE_paint_time_together_l3375_337581

-- Define the rates of work for Harish and Ganpat
def harish_rate : ℚ := 1 / 3
def ganpat_rate : ℚ := 1 / 6

-- Define the total rate when working together
def total_rate : ℚ := harish_rate + ganpat_rate

-- Theorem to prove
theorem paint_time_together : (1 : ℚ) / total_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_together_l3375_337581


namespace NUMINAMATH_CALUDE_project_distribution_theorem_l3375_337547

def number_of_arrangements (total_projects : ℕ) 
                            (company_a_projects : ℕ) 
                            (company_b_projects : ℕ) 
                            (company_c_projects : ℕ) 
                            (company_d_projects : ℕ) : ℕ :=
  (Nat.choose total_projects company_a_projects) * 
  (Nat.choose (total_projects - company_a_projects) company_b_projects) * 
  (Nat.choose (total_projects - company_a_projects - company_b_projects) company_c_projects)

theorem project_distribution_theorem :
  number_of_arrangements 8 3 1 2 2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_theorem_l3375_337547


namespace NUMINAMATH_CALUDE_factors_of_2012_l3375_337599

theorem factors_of_2012 : Finset.card (Nat.divisors 2012) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2012_l3375_337599


namespace NUMINAMATH_CALUDE_owen_daily_chores_hours_l3375_337597

/-- 
Given that:
- There are 24 hours in a day
- Owen spends 6 hours at work
- Owen sleeps for 11 hours

Prove that Owen spends 7 hours on other daily chores.
-/
theorem owen_daily_chores_hours : 
  let total_hours : ℕ := 24
  let work_hours : ℕ := 6
  let sleep_hours : ℕ := 11
  total_hours - work_hours - sleep_hours = 7 := by sorry

end NUMINAMATH_CALUDE_owen_daily_chores_hours_l3375_337597


namespace NUMINAMATH_CALUDE_yan_journey_ratio_l3375_337528

/-- Represents a point on a line --/
structure Point :=
  (position : ℝ)

/-- Represents the scenario of Yan's journey --/
structure Journey :=
  (home : Point)
  (stadium : Point)
  (yan : Point)
  (walking_speed : ℝ)
  (cycling_speed : ℝ)

/-- The conditions of the journey --/
def journey_conditions (j : Journey) : Prop :=
  j.home.position < j.yan.position ∧
  j.yan.position < j.stadium.position ∧
  j.cycling_speed = 5 * j.walking_speed ∧
  (j.stadium.position - j.yan.position) / j.walking_speed =
    (j.yan.position - j.home.position) / j.walking_speed +
    (j.stadium.position - j.home.position) / j.cycling_speed

/-- The theorem to be proved --/
theorem yan_journey_ratio (j : Journey) (h : journey_conditions j) :
  (j.yan.position - j.home.position) / (j.stadium.position - j.yan.position) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_yan_journey_ratio_l3375_337528


namespace NUMINAMATH_CALUDE_transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l3375_337571

-- Define the transformations
def transformation_A (x y : ℝ) : Prop := x = y → x + 3 = y + 3
def transformation_B (x y : ℝ) : Prop := -2 * x = -2 * y → x = y
def transformation_C (x y m : ℝ) : Prop := x / m = y / m → x = y
def transformation_D (x y m : ℝ) : Prop := x = y → x / m = y / m

-- Define a property that checks if a transformation satisfies equation properties
def satisfies_equation_properties (t : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y : ℝ, t x y ↔ x = y

-- Theorem stating that transformation D does not always satisfy equation properties
theorem transformation_D_not_always_valid :
  ¬(∀ m : ℝ, satisfies_equation_properties (transformation_D · · m)) :=
sorry

-- Theorems stating that transformations A, B, and C satisfy equation properties
theorem transformation_A_valid :
  satisfies_equation_properties transformation_A :=
sorry

theorem transformation_B_valid :
  satisfies_equation_properties transformation_B :=
sorry

theorem transformation_C_valid :
  ∀ m : ℝ, m ≠ 0 → satisfies_equation_properties (transformation_C · · m) :=
sorry

end NUMINAMATH_CALUDE_transformation_D_not_always_valid_transformation_A_valid_transformation_B_valid_transformation_C_valid_l3375_337571


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3375_337527

-- Define the function f(x) = x^2 + 3x - 1
def f (x : ℝ) : ℝ := x^2 + 3*x - 1

-- State the theorem
theorem root_exists_in_interval :
  Continuous f →
  f 0 < 0 →
  f 0.5 > 0 →
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 0.5 ∧ f x₀ = 0 :=
by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_root_exists_in_interval_l3375_337527


namespace NUMINAMATH_CALUDE_problem_statement_l3375_337507

noncomputable def f (x : ℝ) : ℝ := 3^x + 2 / (1 - x)

theorem problem_statement 
  (x₀ x₁ x₂ : ℝ) 
  (h_root : f x₀ = 0)
  (h_x₁ : 1 < x₁ ∧ x₁ < x₀)
  (h_x₂ : x₀ < x₂) :
  f x₁ < 0 ∧ f x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3375_337507


namespace NUMINAMATH_CALUDE_max_area_PAB_l3375_337595

-- Define the fixed points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define the lines passing through A and B
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

-- Define the intersection point P
def P (m : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle PAB
def area_PAB (m : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_area_PAB :
  ∃ (max_area : ℝ), 
    (∀ m : ℝ, area_PAB m ≤ max_area) ∧ 
    (∃ m : ℝ, area_PAB m = max_area) ∧
    max_area = 5/2 :=
sorry

end NUMINAMATH_CALUDE_max_area_PAB_l3375_337595


namespace NUMINAMATH_CALUDE_problem_statement_l3375_337555

theorem problem_statement (x y : ℝ) (h : |x - 3| + (y + 4)^2 = 0) : (x + y)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3375_337555


namespace NUMINAMATH_CALUDE_meal_prep_combinations_l3375_337536

def total_people : Nat := 6
def meal_preparers : Nat := 3

theorem meal_prep_combinations :
  Nat.choose total_people meal_preparers = 20 := by
  sorry

end NUMINAMATH_CALUDE_meal_prep_combinations_l3375_337536
