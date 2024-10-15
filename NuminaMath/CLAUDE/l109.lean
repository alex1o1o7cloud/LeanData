import Mathlib

namespace NUMINAMATH_CALUDE_smallest_root_of_unity_for_equation_l109_10987

theorem smallest_root_of_unity_for_equation : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_for_equation_l109_10987


namespace NUMINAMATH_CALUDE_nancy_shoe_multiple_l109_10902

/-- Given Nancy's shoe collection, prove the multiple relating heels to boots and slippers --/
theorem nancy_shoe_multiple :
  ∀ (boots slippers heels : ℕ),
  boots = 6 →
  slippers = boots + 9 →
  2 * (boots + slippers + heels) = 168 →
  ∃ (m : ℕ), heels = m * (boots + slippers) ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_shoe_multiple_l109_10902


namespace NUMINAMATH_CALUDE_infected_and_positive_probability_l109_10936

/-- The infection rate of the novel coronavirus -/
def infection_rate : ℝ := 0.005

/-- The probability of testing positive given infection -/
def positive_given_infection : ℝ := 0.99

/-- The probability that a citizen is infected and tests positive -/
def infected_and_positive : ℝ := infection_rate * positive_given_infection

theorem infected_and_positive_probability :
  infected_and_positive = 0.00495 := by sorry

end NUMINAMATH_CALUDE_infected_and_positive_probability_l109_10936


namespace NUMINAMATH_CALUDE_integral_inequality_l109_10986

open Real MeasureTheory

theorem integral_inequality : 
  ∫ x in (1:ℝ)..2, (1 / x) < ∫ x in (1:ℝ)..2, x ∧ ∫ x in (1:ℝ)..2, x < ∫ x in (1:ℝ)..2, exp x := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l109_10986


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l109_10909

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l109_10909


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l109_10993

/-- Given two polynomial equations:
    1. x^2 - ax + b = 0 with roots α and β
    2. x^2 - px + q = 0 with roots α^2 + β^2 and αβ
    Prove that p = a^2 - b -/
theorem polynomial_root_relation (a b p q α β : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = α ∨ x = β) →
  (∀ x, x^2 - p*x + q = 0 ↔ x = α^2 + β^2 ∨ x = α*β) →
  p = a^2 - b := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l109_10993


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l109_10950

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Proof that the discriminant of 5x² + 3x - 8 is 169 -/
theorem discriminant_of_5x2_plus_3x_minus_8 :
  discriminant 5 3 (-8) = 169 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_plus_3x_minus_8_l109_10950


namespace NUMINAMATH_CALUDE_parallel_lines_equal_angles_l109_10966

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles between lines
variable (angle : Line → Line → ℝ)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles (a b c : Line) (θ : ℝ) :
  parallel a b → angle a c = θ → angle b c = θ := by sorry

end NUMINAMATH_CALUDE_parallel_lines_equal_angles_l109_10966


namespace NUMINAMATH_CALUDE_blue_left_handed_fraction_proof_l109_10964

/-- The fraction of "blue" world participants who are left-handed -/
def blue_left_handed_fraction : ℝ := 0.66

theorem blue_left_handed_fraction_proof :
  let red_to_blue_ratio : ℝ := 2
  let red_left_handed_fraction : ℝ := 1/3
  let total_left_handed_fraction : ℝ := 0.44222222222222224
  blue_left_handed_fraction = 
    (3 * total_left_handed_fraction - 2 * red_left_handed_fraction) / red_to_blue_ratio :=
by sorry

#check blue_left_handed_fraction_proof

end NUMINAMATH_CALUDE_blue_left_handed_fraction_proof_l109_10964


namespace NUMINAMATH_CALUDE_apple_bags_sum_l109_10947

theorem apple_bags_sum : 
  let golden_delicious : ℚ := 17/100
  let macintosh : ℚ := 17/100
  let cortland : ℚ := 33/100
  golden_delicious + macintosh + cortland = 67/100 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_sum_l109_10947


namespace NUMINAMATH_CALUDE_odd_function_value_at_negative_two_l109_10990

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value_at_negative_two
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (x - 1)) :
  f (-2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_at_negative_two_l109_10990


namespace NUMINAMATH_CALUDE_concentric_circles_equal_areas_l109_10924

theorem concentric_circles_equal_areas (R : ℝ) (R₁ R₂ : ℝ) 
  (h₁ : R > 0) 
  (h₂ : R₁ > 0) 
  (h₃ : R₂ > 0) 
  (h₄ : R₁ < R₂) 
  (h₅ : R₂ < R) 
  (h₆ : π * R₁^2 = (π * R^2) / 3) 
  (h₇ : π * R₂^2 - π * R₁^2 = (π * R^2) / 3) 
  (h₈ : π * R^2 - π * R₂^2 = (π * R^2) / 3) : 
  R₁ = (R * Real.sqrt 3) / 3 ∧ R₂ = (R * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_equal_areas_l109_10924


namespace NUMINAMATH_CALUDE_finite_solutions_of_equation_l109_10930

theorem finite_solutions_of_equation : 
  Finite {xyz : ℕ × ℕ × ℕ | (1 : ℚ) / xyz.1 + (1 : ℚ) / xyz.2.1 + (1 : ℚ) / xyz.2.2 = (1 : ℚ) / 1983} :=
by sorry

end NUMINAMATH_CALUDE_finite_solutions_of_equation_l109_10930


namespace NUMINAMATH_CALUDE_select_four_shoes_with_one_match_l109_10991

/-- The number of ways to select four shoes from four different pairs, such that exactly one pair matches. -/
def selectFourShoesWithOneMatch : ℕ := 48

/-- The number of different pairs of shoes. -/
def numPairs : ℕ := 4

/-- The number of shoes to be selected. -/
def shoesToSelect : ℕ := 4

theorem select_four_shoes_with_one_match :
  selectFourShoesWithOneMatch = 
    numPairs * (Nat.choose (numPairs - 1) 2) * 2^2 := by
  sorry

end NUMINAMATH_CALUDE_select_four_shoes_with_one_match_l109_10991


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l109_10965

/-- Given a line with slope -4 passing through the point (5, 2),
    prove that the sum of the slope and y-intercept is 18. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -4 ∧ 2 = m * 5 + b → m + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l109_10965


namespace NUMINAMATH_CALUDE_pen_and_pencil_cost_l109_10908

theorem pen_and_pencil_cost (pencil_cost : ℝ) (pen_cost : ℝ) : 
  pencil_cost = 8 → pen_cost = pencil_cost / 2 → pencil_cost + pen_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_pen_and_pencil_cost_l109_10908


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l109_10935

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a - 1) * x + 2 * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = -1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * 1 + 2 * a = 0) ↔ a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l109_10935


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l109_10994

theorem two_numbers_sum_product : ∃ x y : ℝ, x + y = 20 ∧ x * y = 96 ∧ ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l109_10994


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l109_10941

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_sum1 : a 1 + a 2 = 4/9)
  (h_sum2 : a 3 + a 4 + a 5 + a 6 = 40) :
  (a 7 + a 8 + a 9) / 9 = 117 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l109_10941


namespace NUMINAMATH_CALUDE_fraction_simplification_l109_10957

variable (x : ℝ)

theorem fraction_simplification :
  (x^3 + 4*x^2 + 7*x + 4) / (x^3 + 2*x^2 + x - 4) = (x + 1) / (x - 1) ∧
  2 * (24*x^3 + 46*x^2 + 33*x + 9) / (24*x^3 + 10*x^2 - 9*x - 9) = (4*x + 3) / (4*x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l109_10957


namespace NUMINAMATH_CALUDE_tim_cell_phone_cost_l109_10946

/-- Calculates the total cost of a cell phone plan -/
def calculate_total_cost (base_cost : ℝ) (text_cost : ℝ) (extra_minute_cost : ℝ) 
                         (free_hours : ℝ) (texts_sent : ℝ) (hours_talked : ℝ) : ℝ :=
  let text_total := text_cost * texts_sent
  let extra_minutes := (hours_talked - free_hours) * 60
  let extra_minute_total := extra_minute_cost * extra_minutes
  base_cost + text_total + extra_minute_total

theorem tim_cell_phone_cost :
  let base_cost : ℝ := 30
  let text_cost : ℝ := 0.04
  let extra_minute_cost : ℝ := 0.15
  let free_hours : ℝ := 40
  let texts_sent : ℝ := 200
  let hours_talked : ℝ := 42
  calculate_total_cost base_cost text_cost extra_minute_cost free_hours texts_sent hours_talked = 56 := by
  sorry


end NUMINAMATH_CALUDE_tim_cell_phone_cost_l109_10946


namespace NUMINAMATH_CALUDE_division_problem_l109_10926

theorem division_problem : (100 : ℚ) / ((5 / 2) * 3) = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l109_10926


namespace NUMINAMATH_CALUDE_ellipse_properties_line_through_focus_l109_10977

/-- Ellipse C defined by the equation x²/2 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Point on ellipse C -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Line passing through a point with slope k -/
def line (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

theorem ellipse_properties :
  let a := Real.sqrt 2
  let b := 1
  let c := 1
  let e := c / a
  let left_focus := (-1, 0)
  (∀ x y, ellipse_C x y → x^2 / (a^2) + y^2 / (b^2) = 1) ∧
  (2 * a = 2 * Real.sqrt 2) ∧
  (2 * b = 2) ∧
  (e = Real.sqrt 2 / 2) ∧
  (left_focus.1 = -c ∧ left_focus.2 = 0) :=
by sorry

theorem line_through_focus (k : ℝ) :
  let left_focus := (-1, 0)
  ∃ A B : PointOnEllipse,
    line k left_focus.1 left_focus.2 A.x A.y ∧
    line k left_focus.1 left_focus.2 B.x B.y ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = (8 * Real.sqrt 2 / 7)^2 →
    k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_line_through_focus_l109_10977


namespace NUMINAMATH_CALUDE_min_mozart_and_bach_not_beethoven_l109_10905

def total_surveyed : ℕ := 150
def mozart_fans : ℕ := 120
def bach_fans : ℕ := 105
def beethoven_fans : ℕ := 45

theorem min_mozart_and_bach_not_beethoven :
  ∃ (mozart_set bach_set beethoven_set : Finset (Fin total_surveyed)),
    mozart_set.card = mozart_fans ∧
    bach_set.card = bach_fans ∧
    beethoven_set.card = beethoven_fans ∧
    ((mozart_set ∩ bach_set) \ beethoven_set).card ≥ 75 ∧
    ∀ (m b be : Finset (Fin total_surveyed)),
      m.card = mozart_fans →
      b.card = bach_fans →
      be.card = beethoven_fans →
      ((m ∩ b) \ be).card ≥ 75 :=
by sorry

end NUMINAMATH_CALUDE_min_mozart_and_bach_not_beethoven_l109_10905


namespace NUMINAMATH_CALUDE_solution_is_3x_l109_10968

-- Define the interval
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the integral equation
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x ∈ I, φ x = x + ∫ t in I, x * t * φ t

-- State the theorem
theorem solution_is_3x :
  ∃ φ : ℝ → ℝ, integral_equation φ ∧ (∀ x ∈ I, φ x = 3 * x) :=
sorry

end NUMINAMATH_CALUDE_solution_is_3x_l109_10968


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l109_10995

/-- Calculates the final length of Isabella's hair after a haircut -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem stating that Isabella's hair length after the cut is 9 inches -/
theorem isabellas_hair_length :
  hair_length_after_cut 18 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l109_10995


namespace NUMINAMATH_CALUDE_gcd_2023_2052_l109_10928

theorem gcd_2023_2052 : Nat.gcd 2023 2052 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2023_2052_l109_10928


namespace NUMINAMATH_CALUDE_peggy_final_doll_count_l109_10907

/-- Calculates the final number of dolls Peggy has -/
def peggy_dolls (initial : ℕ) (grandmother_gift : ℕ) : ℕ :=
  initial + grandmother_gift + (grandmother_gift / 2)

/-- Theorem stating that Peggy's final doll count is 51 -/
theorem peggy_final_doll_count :
  peggy_dolls 6 30 = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_final_doll_count_l109_10907


namespace NUMINAMATH_CALUDE_absolute_value_equation_l109_10920

theorem absolute_value_equation (x z : ℝ) 
  (h : |2*x - Real.log z| = 2*x + Real.log z) : x * (z - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l109_10920


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l109_10912

/-- Given a geometric sequence {a_n} where the 5th term is equal to the constant term
    in the expansion of (x + 1/x)^4, prove that a_3 * a_7 = 36 -/
theorem geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 5 = 6) →  -- 5th term is equal to the constant term in (x + 1/x)^4
  a 3 * a 7 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l109_10912


namespace NUMINAMATH_CALUDE_sum_three_numbers_l109_10954

theorem sum_three_numbers (a b c N : ℚ) : 
  a + b + c = 84 ∧ 
  a - 7 = N ∧ 
  b + 7 = N ∧ 
  c / 7 = N → 
  N = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l109_10954


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l109_10949

/-- Given two rectangles of equal area, where one rectangle has dimensions 5 inches by 24 inches,
    and the other rectangle has a length of 3 inches, prove that the width of the second rectangle
    is 40 inches. -/
theorem equal_area_rectangles_width (area : ℝ) (width : ℝ) :
  area = 5 * 24 →  -- Carol's rectangle area
  area = 3 * width →  -- Jordan's rectangle area
  width = 40 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l109_10949


namespace NUMINAMATH_CALUDE_intersection_probability_correct_l109_10942

/-- Given a positive integer n, this function calculates the probability that
    the intersection of two randomly selected non-empty subsets from {1, 2, ..., n}
    is not empty. -/
def intersection_probability (n : ℕ+) : ℚ :=
  (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2

/-- Theorem stating that the probability of non-empty intersection of two randomly
    selected non-empty subsets from {1, 2, ..., n} is given by the function
    intersection_probability. -/
theorem intersection_probability_correct (n : ℕ+) :
  intersection_probability n =
    (4^n.val - 3^n.val : ℚ) / (2^n.val - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_probability_correct_l109_10942


namespace NUMINAMATH_CALUDE_math_only_students_l109_10945

theorem math_only_students (total : ℕ) (math : ℕ) (foreign : ℕ) 
  (h1 : total = 93) 
  (h2 : math = 70) 
  (h3 : foreign = 54) : 
  math - (math + foreign - total) = 39 := by
  sorry

end NUMINAMATH_CALUDE_math_only_students_l109_10945


namespace NUMINAMATH_CALUDE_original_group_size_l109_10983

/-- Given a group of men working on a task, this theorem proves that the original number of men is 42, based on the conditions provided. -/
theorem original_group_size (total_days : ℕ) (remaining_days : ℕ) (absent_men : ℕ) : 
  (total_days = 17) → (remaining_days = 21) → (absent_men = 8) →
  ∃ (original_size : ℕ), 
    (original_size > absent_men) ∧ 
    (1 : ℚ) / (total_days * original_size) = (1 : ℚ) / (remaining_days * (original_size - absent_men)) ∧
    original_size = 42 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l109_10983


namespace NUMINAMATH_CALUDE_production_days_calculation_l109_10932

/-- Proves that given the conditions of the production problem, n must equal 9 -/
theorem production_days_calculation (n : ℕ) 
  (h1 : (n : ℝ) * 50 / n = 50)  -- Average for n days is 50
  (h2 : ((n : ℝ) * 50 + 90) / (n + 1) = 54)  -- New average for n+1 days is 54
  : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l109_10932


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l109_10973

/-- Represents a trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  -- Length of the top base
  AB : ℝ
  -- Distance from point D to point N on the bottom base
  DN : ℝ
  -- Radius of the inscribed circle
  r : ℝ

/-- The area of a trapezoid with an inscribed circle -/
def trapezoidArea (t : InscribedCircleTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 27 -/
theorem specific_trapezoid_area :
  ∀ (t : InscribedCircleTrapezoid),
    t.AB = 12 ∧ t.DN = 1 ∧ t.r = 2 →
    trapezoidArea t = 27 :=
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l109_10973


namespace NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l109_10974

/-- The probability that all 3 girls select the same colored marble from a bag with 3 white and 3 black marbles -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 6
  let white_marbles : ℕ := 3
  let black_marbles : ℕ := 3
  let girls : ℕ := 3
  let prob_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  let prob_all_black : ℚ := (black_marbles / total_marbles) * ((black_marbles - 1) / (total_marbles - 1)) * ((black_marbles - 2) / (total_marbles - 2))
  prob_all_white + prob_all_black

theorem same_color_marble_probability_is_one_twentieth : same_color_marble_probability = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_is_one_twentieth_l109_10974


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_l109_10937

theorem smiths_bakery_pies (mcgees_pies : ℕ) (smiths_pies : ℕ) : 
  mcgees_pies = 16 → 
  smiths_pies = 4 * mcgees_pies + 6 → 
  smiths_pies = 70 := by
sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_l109_10937


namespace NUMINAMATH_CALUDE_projective_transformation_uniqueness_l109_10952

/-- A projective transformation on a straight line -/
structure ProjectiveTransformation :=
  (f : ℝ → ℝ)

/-- The property that a projective transformation preserves cross-ratio -/
def PreservesCrossRatio (t : ProjectiveTransformation) : Prop :=
  ∀ a b c d : ℝ, (a - c) * (b - d) / ((b - c) * (a - d)) = 
    (t.f a - t.f c) * (t.f b - t.f d) / ((t.f b - t.f c) * (t.f a - t.f d))

/-- Two projective transformations are equal if they agree on three distinct points -/
theorem projective_transformation_uniqueness 
  (t₁ t₂ : ProjectiveTransformation)
  (h₁ : PreservesCrossRatio t₁)
  (h₂ : PreservesCrossRatio t₂)
  (a b c : ℝ)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (eq_a : t₁.f a = t₂.f a)
  (eq_b : t₁.f b = t₂.f b)
  (eq_c : t₁.f c = t₂.f c) :
  ∀ x : ℝ, t₁.f x = t₂.f x :=
sorry

end NUMINAMATH_CALUDE_projective_transformation_uniqueness_l109_10952


namespace NUMINAMATH_CALUDE_not_octal_7857_l109_10923

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end NUMINAMATH_CALUDE_not_octal_7857_l109_10923


namespace NUMINAMATH_CALUDE_system_solution_l109_10901

theorem system_solution : 
  ∀ x y : ℝ, (y^2 + x*y = 15 ∧ x^2 + x*y = 10) ↔ ((x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l109_10901


namespace NUMINAMATH_CALUDE_baking_contest_votes_l109_10906

/-- The number of votes for the witch cake -/
def witch_votes : ℕ := 7

/-- The number of votes for the unicorn cake -/
def unicorn_votes : ℕ := 3 * witch_votes

/-- The number of votes for the dragon cake -/
def dragon_votes : ℕ := witch_votes + 25

/-- The total number of votes cast -/
def total_votes : ℕ := witch_votes + unicorn_votes + dragon_votes

theorem baking_contest_votes : total_votes = 60 := by
  sorry

end NUMINAMATH_CALUDE_baking_contest_votes_l109_10906


namespace NUMINAMATH_CALUDE_prob_2_to_4_value_l109_10978

/-- The probability distribution of a random variable ξ -/
def P (k : ℕ) : ℚ := 1 / 2^k

/-- The probability that 2 < ξ ≤ 4 -/
def prob_2_to_4 : ℚ := P 3 + P 4

theorem prob_2_to_4_value : prob_2_to_4 = 3/16 := by sorry

end NUMINAMATH_CALUDE_prob_2_to_4_value_l109_10978


namespace NUMINAMATH_CALUDE_remainder_three_power_800_mod_17_l109_10913

theorem remainder_three_power_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_800_mod_17_l109_10913


namespace NUMINAMATH_CALUDE_relay_team_selection_l109_10916

-- Define the number of sprinters
def total_sprinters : ℕ := 6

-- Define the number of sprinters to be selected
def selected_sprinters : ℕ := 4

-- Define a function to calculate the number of ways to select and arrange sprinters
def relay_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

-- Define a function to calculate the number of ways with restrictions
def restricted_arrangements (total : ℕ) (selected : ℕ) : ℕ :=
  sorry

theorem relay_team_selection :
  restricted_arrangements total_sprinters selected_sprinters = 252 :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l109_10916


namespace NUMINAMATH_CALUDE_statement_S_holds_for_options_l109_10933

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def options : List ℕ := [90, 99, 108, 117]

theorem statement_S_holds_for_options : ∀ n ∈ options, 
  (sum_of_digits n) % 9 = 0 → n % 3 = 0 := by sorry

end NUMINAMATH_CALUDE_statement_S_holds_for_options_l109_10933


namespace NUMINAMATH_CALUDE_contact_lenses_sales_l109_10915

theorem contact_lenses_sales (soft_price hard_price : ℕ) 
  (soft_hard_difference total_sales : ℕ) : 
  soft_price = 150 →
  hard_price = 85 →
  soft_hard_difference = 5 →
  total_sales = 1455 →
  ∃ (soft hard : ℕ), 
    soft = hard + soft_hard_difference ∧
    soft_price * soft + hard_price * hard = total_sales ∧
    soft + hard = 11 :=
by sorry

end NUMINAMATH_CALUDE_contact_lenses_sales_l109_10915


namespace NUMINAMATH_CALUDE_division_of_powers_l109_10956

theorem division_of_powers (x : ℝ) (h : x ≠ 0) : (-6 * x^3) / (-2 * x^2) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_division_of_powers_l109_10956


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l109_10940

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) := by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, Real.exp x - x - 1 > 0) ↔ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l109_10940


namespace NUMINAMATH_CALUDE_sum_inequality_l109_10904

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) (h3 : c * d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l109_10904


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l109_10914

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_right (reflect_x center) 5) = (8, 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l109_10914


namespace NUMINAMATH_CALUDE_expression_evaluation_l109_10975

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -2) :
  -a - b^2 + a*b = -16 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l109_10975


namespace NUMINAMATH_CALUDE_clock_hand_speed_ratio_l109_10969

/-- Represents the number of degrees in a full rotation of a clock face. -/
def clock_degrees : ℕ := 360

/-- Represents the number of minutes it takes for the minute hand to complete a full rotation. -/
def minute_hand_period : ℕ := 60

/-- Represents the number of hours it takes for the hour hand to complete a full rotation. -/
def hour_hand_period : ℕ := 12

/-- Theorem stating that the ratio of the speeds of the hour hand to the minute hand is 2:24. -/
theorem clock_hand_speed_ratio :
  (clock_degrees / (hour_hand_period * minute_hand_period) : ℚ) / 
  (clock_degrees / minute_hand_period : ℚ) = 2 / 24 := by
  sorry

end NUMINAMATH_CALUDE_clock_hand_speed_ratio_l109_10969


namespace NUMINAMATH_CALUDE_alexs_score_l109_10931

theorem alexs_score (total_students : ℕ) (initial_students : ℕ) (initial_avg : ℕ) (final_avg : ℕ) :
  total_students = 20 →
  initial_students = 19 →
  initial_avg = 76 →
  final_avg = 78 →
  (initial_students * initial_avg + (total_students - initial_students) * x) / total_students = final_avg →
  x = 116 :=
by sorry

end NUMINAMATH_CALUDE_alexs_score_l109_10931


namespace NUMINAMATH_CALUDE_adjacent_negative_product_l109_10948

def a (n : ℕ) : ℤ := 2 * n - 17

theorem adjacent_negative_product :
  ∀ n : ℕ, (a n * a (n + 1) < 0) ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_adjacent_negative_product_l109_10948


namespace NUMINAMATH_CALUDE_division_by_240_property_l109_10989

-- Define a function to check if a number has at least two digits
def hasAtLeastTwoDigits (n : ℕ) : Prop := n ≥ 10

-- Define the theorem
theorem division_by_240_property (a b : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hta : hasAtLeastTwoDigits a) (htb : hasAtLeastTwoDigits b) 
  (hab : a > b) : 
  (∃ k : ℕ, a^4 - b^4 = 240 * k) ∧ 
  (∀ m : ℕ, m > 240 → ¬(∀ x y : ℕ, Nat.Prime x → Nat.Prime y → hasAtLeastTwoDigits x → hasAtLeastTwoDigits y → x > y → ∃ l : ℕ, x^4 - y^4 = m * l)) :=
by sorry


end NUMINAMATH_CALUDE_division_by_240_property_l109_10989


namespace NUMINAMATH_CALUDE_sin_cos_alpha_l109_10953

theorem sin_cos_alpha (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_alpha_l109_10953


namespace NUMINAMATH_CALUDE_stock_price_theorem_l109_10944

def stock_price_evolution (initial_price : ℝ) (year1_change : ℝ) (year2_change : ℝ) (year3_change : ℝ) : ℝ :=
  let price_after_year1 := initial_price * (1 + year1_change)
  let price_after_year2 := price_after_year1 * (1 + year2_change)
  let price_after_year3 := price_after_year2 * (1 + year3_change)
  price_after_year3

theorem stock_price_theorem :
  stock_price_evolution 150 0.5 (-0.3) 0.2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_theorem_l109_10944


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l109_10939

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) < -6452.25 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achieved_l109_10939


namespace NUMINAMATH_CALUDE_max_books_borrowed_l109_10938

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) (two_books : Nat) (avg_books : Nat) :
  total_students = 30 →
  zero_books = 5 →
  one_book = 12 →
  two_books = 8 →
  avg_books = 2 →
  ∃ (max_books : Nat), max_books = 20 ∧ 
    ∀ (student_books : Nat), student_books ≤ max_books ∧
    (total_students * avg_books = 
      zero_books * 0 + one_book * 1 + two_books * 2 + 
      (total_students - zero_books - one_book - two_books - 1) * 3 + max_books) :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l109_10938


namespace NUMINAMATH_CALUDE_ellipse_equation_l109_10917

/-- An ellipse with one focus at (1,0) and eccentricity 1/2 has the standard equation x²/4 + y²/3 = 1 -/
theorem ellipse_equation (F : ℝ × ℝ) (e : ℝ) (h1 : F = (1, 0)) (h2 : e = 1/2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l109_10917


namespace NUMINAMATH_CALUDE_salary_change_percentage_l109_10996

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l109_10996


namespace NUMINAMATH_CALUDE_point_outside_circle_l109_10967

theorem point_outside_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0 → (a - x)^2 + (2 - y)^2 > 0) ↔ 
  (2 < a ∧ a < 9/4) :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l109_10967


namespace NUMINAMATH_CALUDE_special_function_omega_value_l109_10925

/-- A function f with the properties described in the problem -/
structure SpecialFunction (ω : ℝ) where
  f : ℝ → ℝ
  eq : ∀ x, f x = 3 * Real.sin (ω * x + π / 3)
  positive_ω : ω > 0
  equal_values : f (π / 6) = f (π / 3)
  min_no_max : ∃ x₀ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x₀ ≤ f x
             ∧ ¬∃ x₁ ∈ Set.Ioo (π / 6) (π / 3), ∀ x ∈ Set.Ioo (π / 6) (π / 3), f x ≤ f x₁

/-- The main theorem stating that ω must be 14/3 -/
theorem special_function_omega_value {ω : ℝ} (sf : SpecialFunction ω) : ω = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_omega_value_l109_10925


namespace NUMINAMATH_CALUDE_cubic_function_properties_l109_10911

def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties (b c d : ℝ) :
  f b c d 0 = 2 ∧ 
  (∀ x, (6:ℝ)*x - f b c d (-1) + 7 = 0 ↔ x = -1) →
  f b c d (-1) = 1 ∧
  ∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l109_10911


namespace NUMINAMATH_CALUDE_remaining_shoes_l109_10943

theorem remaining_shoes (large medium small sold : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_sold : sold = 83) :
  large + medium + small - sold = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_shoes_l109_10943


namespace NUMINAMATH_CALUDE_total_cost_of_shirts_proof_total_cost_of_shirts_l109_10955

/-- The total cost of two shirts, where the first shirt costs $6 more than the second,
    and the first shirt costs $15, is $24. -/
theorem total_cost_of_shirts : ℕ → Prop :=
  fun n => n = 24 ∧ ∃ (cost1 cost2 : ℕ),
    cost1 = 15 ∧
    cost1 = cost2 + 6 ∧
    n = cost1 + cost2

/-- Proof of the theorem -/
theorem proof_total_cost_of_shirts : total_cost_of_shirts 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_shirts_proof_total_cost_of_shirts_l109_10955


namespace NUMINAMATH_CALUDE_apple_bag_cost_proof_l109_10999

/-- The cost of a bag of dozen apples -/
def apple_bag_cost : ℝ := 14

theorem apple_bag_cost_proof :
  let kiwi_cost : ℝ := 10
  let banana_cost : ℝ := 5
  let initial_money : ℝ := 50
  let subway_fare : ℝ := 3.5
  let max_apples : ℕ := 24
  apple_bag_cost = (initial_money - (kiwi_cost + banana_cost) - 2 * subway_fare) / (max_apples / 12) :=
by
  sorry

#check apple_bag_cost_proof

end NUMINAMATH_CALUDE_apple_bag_cost_proof_l109_10999


namespace NUMINAMATH_CALUDE_inscribed_squares_segment_product_l109_10980

theorem inscribed_squares_segment_product (c d : ℝ) : 
  (∃ (small_square_area large_square_area : ℝ),
    small_square_area = 9 ∧ 
    large_square_area = 18 ∧ 
    c + d = (large_square_area).sqrt ∧ 
    c^2 + d^2 = large_square_area) → 
  c * d = 0 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_segment_product_l109_10980


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_one_half_open_l109_10919

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set N
def N : Set ℝ := {x : ℝ | x < 1}

-- Theorem statement
theorem M_intersect_N_equals_zero_one_half_open : M ∩ N = Set.Icc 0 1 ∩ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_one_half_open_l109_10919


namespace NUMINAMATH_CALUDE_unique_digit_product_l109_10951

theorem unique_digit_product : ∃! (x y z : ℕ), 
  (x < 10 ∧ y < 10 ∧ z < 10) ∧ 
  (10 ≤ 10 * x + y) ∧ (10 * x + y ≤ 99) ∧
  (1 ≤ z) ∧
  (x * (10 * x + y) = 111 * z) := by
sorry

end NUMINAMATH_CALUDE_unique_digit_product_l109_10951


namespace NUMINAMATH_CALUDE_min_value_inequality_l109_10934

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l109_10934


namespace NUMINAMATH_CALUDE_power_of_25_equals_power_of_5_l109_10921

theorem power_of_25_equals_power_of_5 : (25 : ℕ) ^ 5 = 5 ^ 10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_25_equals_power_of_5_l109_10921


namespace NUMINAMATH_CALUDE_f_is_quadratic_l109_10922

/-- A quadratic equation in terms of x is a polynomial equation of degree 2 in x. -/
def IsQuadraticInX (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation x(x-1) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 1)

/-- Theorem stating that f is a quadratic equation in terms of x -/
theorem f_is_quadratic : IsQuadraticInX f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l109_10922


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l109_10971

theorem collinear_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![m, 2*m - 1]
  (∃ (k : ℝ), b = k • a) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l109_10971


namespace NUMINAMATH_CALUDE_crease_length_l109_10997

/-- Given a rectangular piece of paper 8 inches wide, when folded such that one corner
    touches the opposite side and forms an angle θ at the corner where the crease starts,
    the length L of the crease is equal to 8 cos(θ). -/
theorem crease_length (θ : Real) (L : Real) : L = 8 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_crease_length_l109_10997


namespace NUMINAMATH_CALUDE_spending_problem_l109_10970

theorem spending_problem (initial_amount : ℚ) : 
  (initial_amount * (5/7) * (10/13) * (4/5) * (8/11) = 5400) → 
  initial_amount = 16890 := by
  sorry

end NUMINAMATH_CALUDE_spending_problem_l109_10970


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l109_10992

theorem reciprocal_of_negative_fraction :
  ((-1 : ℚ) / 2023)⁻¹ = -2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l109_10992


namespace NUMINAMATH_CALUDE_base_12_addition_l109_10958

/-- Addition in base 12 --/
def base_12_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 12 --/
def to_base_12 (n : ℕ) : ℕ := sorry

/-- Conversion from base 12 to base 10 --/
def from_base_12 (n : ℕ) : ℕ := sorry

theorem base_12_addition :
  base_12_add (from_base_12 528) (from_base_12 274) = to_base_12 940 :=
sorry

end NUMINAMATH_CALUDE_base_12_addition_l109_10958


namespace NUMINAMATH_CALUDE_vector_magnitude_l109_10984

theorem vector_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-1, x)
  (2 • a - b) • b = 0 → ‖a‖ = 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l109_10984


namespace NUMINAMATH_CALUDE_initial_pencils_equals_sum_l109_10979

/-- The number of pencils Ken had initially -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of pencils Ken gave to Nilo -/
def pencils_to_nilo : ℕ := pencils_to_manny + 10

/-- The number of pencils Ken kept for himself -/
def pencils_kept : ℕ := 20

/-- Theorem stating that the initial number of pencils is equal to the sum of
    pencils given to Manny, Nilo, and kept by Ken -/
theorem initial_pencils_equals_sum :
  initial_pencils = pencils_to_manny + pencils_to_nilo + pencils_kept :=
by sorry

end NUMINAMATH_CALUDE_initial_pencils_equals_sum_l109_10979


namespace NUMINAMATH_CALUDE_election_vote_difference_l109_10985

theorem election_vote_difference (total_votes : ℕ) (invalid_votes : ℕ) (losing_percentage : ℚ) :
  total_votes = 12600 →
  invalid_votes = 100 →
  losing_percentage = 30 / 100 →
  ∃ (valid_votes winning_votes losing_votes : ℕ),
    valid_votes = total_votes - invalid_votes ∧
    losing_votes = (losing_percentage * valid_votes).floor ∧
    winning_votes = valid_votes - losing_votes ∧
    winning_votes - losing_votes = 5000 := by
  sorry

#check election_vote_difference

end NUMINAMATH_CALUDE_election_vote_difference_l109_10985


namespace NUMINAMATH_CALUDE_number_times_power_of_five_l109_10900

theorem number_times_power_of_five (x : ℝ) : x * (5^4) = 75625 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_number_times_power_of_five_l109_10900


namespace NUMINAMATH_CALUDE_school_total_students_l109_10962

/-- The total number of students in a school with a given number of grades and students per grade -/
def total_students (num_grades : ℕ) (students_per_grade : ℕ) : ℕ :=
  num_grades * students_per_grade

/-- Theorem stating that the total number of students in a school with 304 grades and 75 students per grade is 22800 -/
theorem school_total_students :
  total_students 304 75 = 22800 := by
  sorry

end NUMINAMATH_CALUDE_school_total_students_l109_10962


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_group_l109_10960

/-- Systematic sampling function -/
def systematic_sample (class_size : ℕ) (sample_size : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun group => start + (group - 1) * (class_size / sample_size)

theorem systematic_sampling_fourth_group 
  (class_size : ℕ) 
  (sample_size : ℕ) 
  (second_group_number : ℕ) :
  class_size = 72 →
  sample_size = 6 →
  second_group_number = 16 →
  systematic_sample class_size sample_size second_group_number 4 = 40 :=
by
  sorry

#check systematic_sampling_fourth_group

end NUMINAMATH_CALUDE_systematic_sampling_fourth_group_l109_10960


namespace NUMINAMATH_CALUDE_sticks_difference_l109_10982

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14)
  (h2 : left = 4) :
  picked_up - left = 10 := by
  sorry

end NUMINAMATH_CALUDE_sticks_difference_l109_10982


namespace NUMINAMATH_CALUDE_packets_to_fill_gunny_bag_l109_10998

/-- Represents the weight of a packet in ounces -/
def packet_weight : ℕ := 16 * 16 + 4

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- Conversion rate from tons to pounds -/
def tons_to_pounds : ℕ := 2500

/-- Conversion rate from pounds to ounces -/
def pounds_to_ounces : ℕ := 16

/-- Theorem stating the number of packets needed to fill the gunny bag -/
theorem packets_to_fill_gunny_bag : 
  (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight = 2000 := by
  sorry

#eval (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight

end NUMINAMATH_CALUDE_packets_to_fill_gunny_bag_l109_10998


namespace NUMINAMATH_CALUDE_apartment_number_exists_unique_l109_10963

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def contains_digit (n d : ℕ) : Prop :=
  ∃ k m, n = 100 * k + 10 * m + d ∧ 0 ≤ k ∧ k < 10 ∧ 0 ≤ m ∧ m < 10

theorem apartment_number_exists_unique :
  ∃! n : ℕ, is_three_digit n ∧
            n % 11 = 0 ∧
            n % 2 = 0 ∧
            n % 5 = 0 ∧
            ¬ contains_digit n 7 :=
sorry

end NUMINAMATH_CALUDE_apartment_number_exists_unique_l109_10963


namespace NUMINAMATH_CALUDE_isabel_bouquets_l109_10961

/-- Given the total number of flowers, flowers per bouquet, and wilted flowers,
    calculate the maximum number of full bouquets that can be made. -/
def max_bouquets (total : ℕ) (per_bouquet : ℕ) (wilted : ℕ) : ℕ :=
  (total - wilted) / per_bouquet

/-- Theorem stating that given 132 total flowers, 11 flowers per bouquet,
    and 16 wilted flowers, the maximum number of full bouquets is 10. -/
theorem isabel_bouquets :
  max_bouquets 132 11 16 = 10 := by
  sorry

end NUMINAMATH_CALUDE_isabel_bouquets_l109_10961


namespace NUMINAMATH_CALUDE_abc_sum_in_base_7_l109_10972

theorem abc_sum_in_base_7 : ∃ (A B C : Nat), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧  -- digits less than 7
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧  -- non-zero digits
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧  -- distinct digits
  Nat.Prime A ∧            -- A is prime
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧  -- ABC₇ + BC₇ = ACA₇
  A + B + C = 13           -- sum is 16₇ in base 7, which is 13 in base 10
  := by sorry

end NUMINAMATH_CALUDE_abc_sum_in_base_7_l109_10972


namespace NUMINAMATH_CALUDE_sets_equality_independent_of_order_l109_10959

theorem sets_equality_independent_of_order (A B : Set ℕ) : 
  (∀ x, x ∈ A ↔ x ∈ B) → A = B :=
by sorry

end NUMINAMATH_CALUDE_sets_equality_independent_of_order_l109_10959


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l109_10988

theorem max_leftover_grapes :
  ∀ n : ℕ, ∃ q r : ℕ, n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l109_10988


namespace NUMINAMATH_CALUDE_line_extraction_theorem_l109_10910

-- Define a structure for a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if two lines are parallel
def areParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Function to check if a line intersects all given lines
def intersectsAllLines (l : Line) (l1 l2 l3 l4 : Line) : Prop :=
  ∃ p1 p2 p3 p4 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    pointOnLine p4 l ∧ pointOnLine p4 l4

-- Function to check if segments are in given ratios
def segmentsInRatio (l : Line) (l1 l2 l3 : Line) (r1 r2 : ℝ) : Prop :=
  ∃ p1 p2 p3 : Point,
    pointOnLine p1 l ∧ pointOnLine p1 l1 ∧
    pointOnLine p2 l ∧ pointOnLine p2 l2 ∧
    pointOnLine p3 l ∧ pointOnLine p3 l3 ∧
    (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = r1 * ((p3.x - p2.x)^2 + (p3.y - p2.y)^2) ∧
    (p3.x - p2.x)^2 + (p3.y - p2.y)^2 = r2 * ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Theorem statement
theorem line_extraction_theorem (l1 l2 l3 l4 : Line) (r1 r2 : ℝ) :
  (∃ l : Line, intersectsAllLines l l1 l2 l3 l4 ∧ segmentsInRatio l l1 l2 l3 r1 r2) ∨
  (∃ m : Line, segmentsInRatio m l1 l2 l3 r1 r2 ∧ (areParallel m l4 ∨ m = l4)) :=
sorry

end NUMINAMATH_CALUDE_line_extraction_theorem_l109_10910


namespace NUMINAMATH_CALUDE_solution_characterization_l109_10903

/-- Represents a 3-digit integer abc --/
structure ThreeDigitInt where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a < 10
  h3 : b < 10
  h4 : c < 10

/-- Converts a ThreeDigitInt to its numerical value --/
def toInt (n : ThreeDigitInt) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Checks if a ThreeDigitInt satisfies the given equation --/
def satisfiesEquation (n : ThreeDigitInt) : Prop :=
  n.b * (10 * n.a + n.c) = n.c * (10 * n.a + n.b) + 10

/-- The set of all ThreeDigitInt that satisfy the equation --/
def solutionSet : Set ThreeDigitInt :=
  {n : ThreeDigitInt | satisfiesEquation n}

/-- The theorem to be proved --/
theorem solution_characterization :
  solutionSet = {
    ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩,
    ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩
  } := by sorry

#eval toInt ⟨1, 1, 2, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 2, 3, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 3, 4, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 4, 5, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 5, 6, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 6, 7, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 7, 8, by norm_num, by norm_num, by norm_num, by norm_num⟩
#eval toInt ⟨1, 8, 9, by norm_num, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_solution_characterization_l109_10903


namespace NUMINAMATH_CALUDE_max_checkers_8x8_l109_10927

/-- Represents a chess board -/
structure Board :=
  (size : Nat)

/-- Represents a configuration of checkers on a board -/
structure CheckerConfiguration :=
  (board : Board)
  (numCheckers : Nat)

/-- Predicate to check if a configuration is valid (all checkers under attack) -/
def isValidConfiguration (config : CheckerConfiguration) : Prop := sorry

/-- The maximum number of checkers that can be placed on a board -/
def maxCheckers (b : Board) : Nat := sorry

/-- Theorem stating the maximum number of checkers on an 8x8 board -/
theorem max_checkers_8x8 :
  ∃ (config : CheckerConfiguration),
    config.board.size = 8 ∧
    isValidConfiguration config ∧
    config.numCheckers = maxCheckers config.board ∧
    config.numCheckers = 32 :=
  sorry

end NUMINAMATH_CALUDE_max_checkers_8x8_l109_10927


namespace NUMINAMATH_CALUDE_integer_fraction_l109_10918

theorem integer_fraction (a : ℕ+) : 
  (↑(2 * a.val + 8) / ↑(a.val + 1) : ℚ).isInt ↔ a.val = 1 ∨ a.val = 2 ∨ a.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_l109_10918


namespace NUMINAMATH_CALUDE_min_value_expression_l109_10976

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a * b + 3 = b) :
  (1 / a + 2 * b) ≥ 8 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l109_10976


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l109_10981

def I : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (I \ M) ∩ (I \ N) = ∅ :=
by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l109_10981


namespace NUMINAMATH_CALUDE_ultramen_defeat_monster_l109_10929

theorem ultramen_defeat_monster (monster_health : ℕ) (ultraman1_rate : ℕ) (ultraman2_rate : ℕ) :
  monster_health = 100 →
  ultraman1_rate = 12 →
  ultraman2_rate = 8 →
  (monster_health : ℚ) / (ultraman1_rate + ultraman2_rate : ℚ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ultramen_defeat_monster_l109_10929
