import Mathlib

namespace NUMINAMATH_CALUDE_linear_function_decreases_iff_positive_slope_l2850_285076

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The value of a linear function at a given x -/
def LinearFunction.value (f : LinearFunction) (x : ℝ) : ℝ :=
  f.slope * x + f.intercept

/-- A linear function decreases as x decreases iff its slope is positive -/
theorem linear_function_decreases_iff_positive_slope (f : LinearFunction) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f.value x₁ < f.value x₂) ↔ f.slope > 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreases_iff_positive_slope_l2850_285076


namespace NUMINAMATH_CALUDE_mixture_percentage_l2850_285092

theorem mixture_percentage (solution1 solution2 : ℝ) 
  (percent1 percent2 : ℝ) (h1 : solution1 = 6) 
  (h2 : solution2 = 4) (h3 : percent1 = 0.2) 
  (h4 : percent2 = 0.6) : 
  (percent1 * solution1 + percent2 * solution2) / (solution1 + solution2) = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_mixture_percentage_l2850_285092


namespace NUMINAMATH_CALUDE_final_number_not_zero_l2850_285014

/-- Represents the operation of replacing two numbers with their sum or difference -/
inductive Operation
  | Sum : ℕ → ℕ → Operation
  | Difference : ℕ → ℕ → Operation

/-- The type representing the state of the blackboard -/
def Blackboard := List ℕ

/-- Applies an operation to the blackboard -/
def applyOperation (board : Blackboard) (op : Operation) : Blackboard :=
  match op with
  | Operation.Sum a b => sorry
  | Operation.Difference a b => sorry

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial blackboard -/
def applyOperations (initialBoard : Blackboard) (ops : OperationSequence) : Blackboard :=
  ops.foldl applyOperation initialBoard

/-- The initial state of the blackboard -/
def initialBoard : Blackboard := List.range 1974

theorem final_number_not_zero (ops : OperationSequence) :
  (applyOperations initialBoard ops).length = 1 →
  (applyOperations initialBoard ops).head? ≠ some 0 := by
  sorry

end NUMINAMATH_CALUDE_final_number_not_zero_l2850_285014


namespace NUMINAMATH_CALUDE_circle_properties_l2850_285061

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 5

-- Define the points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Define the line x - y = 0
def center_line (x y : ℝ) : Prop := x = y

-- Define the line x + 2y + 4 = 0
def distance_line (x y : ℝ) : Prop := x + 2*y + 4 = 0

-- Main theorem
theorem circle_properties :
  -- The circle passes through A and B
  circle_C point_A.1 point_A.2 ∧ circle_C point_B.1 point_B.2 ∧
  -- The center is on the line x - y = 0
  ∃ x y, circle_C x y ∧ center_line x y ∧
  -- Maximum and minimum distances
  (∀ x y, circle_C x y →
    (∃ d_max, d_max = (12/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≤ d_max) ∧
    (∃ d_min, d_min = (2/5)*Real.sqrt 5 ∧
      ∀ d, (∃ x' y', distance_line x' y' ∧ d = Real.sqrt ((x - x')^2 + (y - y')^2)) → d ≥ d_min)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l2850_285061


namespace NUMINAMATH_CALUDE_no_special_eight_digit_number_l2850_285023

theorem no_special_eight_digit_number : ¬∃ N : ℕ,
  (10000000 ≤ N ∧ N < 100000000) ∧
  (∀ i : Fin 8, 
    let digit := (N / (10 ^ (7 - i.val))) % 10
    digit ≠ 0 ∧
    N % digit = i.val + 1) :=
sorry

end NUMINAMATH_CALUDE_no_special_eight_digit_number_l2850_285023


namespace NUMINAMATH_CALUDE_x_in_terms_of_y_and_z_l2850_285018

theorem x_in_terms_of_y_and_z (x y z : ℝ) :
  1 / (x + y) + 1 / (x - y) = z / (x - y) → x = z / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_in_terms_of_y_and_z_l2850_285018


namespace NUMINAMATH_CALUDE_rotate90_clockwise_correct_rotation_result_l2850_285028

/-- Rotate a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate90_clockwise_correct (x y : ℝ) :
  rotate90Clockwise (x, y) = (y, -x) := by sorry

/-- The original point A -/
def A : ℝ × ℝ := (2, 3)

/-- The rotated point B -/
def B : ℝ × ℝ := rotate90Clockwise A

theorem rotation_result :
  B = (3, -2) := by sorry

end NUMINAMATH_CALUDE_rotate90_clockwise_correct_rotation_result_l2850_285028


namespace NUMINAMATH_CALUDE_sallys_cards_l2850_285071

/-- The number of Pokemon cards Sally had initially -/
def initial_cards : ℕ := 27

/-- The number of cards Dan gave to Sally -/
def dans_cards : ℕ := 41

/-- The number of cards Sally bought -/
def bought_cards : ℕ := 20

/-- The total number of cards Sally has now -/
def total_cards : ℕ := 88

/-- Theorem stating that the initial number of cards plus the acquired cards equals the total cards -/
theorem sallys_cards : initial_cards + dans_cards + bought_cards = total_cards := by
  sorry

end NUMINAMATH_CALUDE_sallys_cards_l2850_285071


namespace NUMINAMATH_CALUDE_money_share_difference_l2850_285064

theorem money_share_difference (total : ℝ) (moses_percent : ℝ) (rachel_percent : ℝ) 
  (h1 : total = 80)
  (h2 : moses_percent = 0.35)
  (h3 : rachel_percent = 0.20) : 
  moses_percent * total - (total - (moses_percent * total + rachel_percent * total)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_share_difference_l2850_285064


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l2850_285089

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: A ∩ C ≠ ∅ if and only if a < 8
theorem intersection_A_C_nonempty (a : ℝ) : A ∩ C a ≠ ∅ ↔ a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l2850_285089


namespace NUMINAMATH_CALUDE_line_equation_l2850_285003

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point
def given_point : Point := { x := 3, y := -2 }

-- Theorem stating the line equation
theorem line_equation : 
  ∃ (l1 l2 : Line), 
    (point_on_line given_point l1 ∧ equal_intercepts l1) ∧
    (point_on_line given_point l2 ∧ equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = 3 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l2850_285003


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l2850_285020

-- 1. Contrapositive
theorem contrapositive_equivalence (p q : ℝ) :
  (p^2 + q^2 = 2 → p + q ≤ 2) ↔ (p + q > 2 → p^2 + q^2 ≠ 2) := by sorry

-- 2. Vector dot product
theorem dot_product_not_sufficient (a b c : ℝ × ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  (a.1 * b.1 + a.2 * b.2 = b.1 * c.1 + b.2 * c.2) → (a = c → False) := by sorry

-- 3. Non-periodicity of exponential function
theorem exp_not_periodic (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ¬∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, a^x = a^(x + T) := by sorry

-- 4. Negation of existential proposition
theorem negation_existential :
  (¬∃ x : ℝ, x^2 - 3*x + 2 ≥ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_dot_product_not_sufficient_exp_not_periodic_negation_existential_l2850_285020


namespace NUMINAMATH_CALUDE_max_a_value_l2850_285038

theorem max_a_value (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 3 * c)
  (h3 : c < 4 * d)
  (h4 : b + d = 200) :
  a ≤ 449 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 449 ∧ 
    a' < 3 * b' ∧ 
    b' < 3 * c' ∧ 
    c' < 4 * d' ∧ 
    b' + d' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2850_285038


namespace NUMINAMATH_CALUDE_men_on_first_road_calculation_l2850_285017

/-- The number of men who worked on the first road -/
def men_on_first_road : ℕ := 30

/-- The length of the first road in kilometers -/
def first_road_length : ℕ := 1

/-- The number of days spent working on the first road -/
def days_on_first_road : ℕ := 12

/-- The number of hours worked per day on the first road -/
def hours_per_day_first_road : ℕ := 8

/-- The number of men working on the second road -/
def men_on_second_road : ℕ := 20

/-- The number of days spent working on the second road -/
def days_on_second_road : ℕ := 32

/-- The number of hours worked per day on the second road -/
def hours_per_day_second_road : ℕ := 9

/-- The length of the second road in kilometers -/
def second_road_length : ℕ := 2

theorem men_on_first_road_calculation :
  men_on_first_road * days_on_first_road * hours_per_day_first_road =
  (men_on_second_road * days_on_second_road * hours_per_day_second_road) / 2 :=
by sorry

end NUMINAMATH_CALUDE_men_on_first_road_calculation_l2850_285017


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2850_285027

theorem compound_interest_rate (P : ℝ) (t : ℕ) (CI : ℝ) (r : ℝ) : 
  P = 4500 →
  t = 2 →
  CI = 945.0000000000009 →
  (P + CI) = P * (1 + r) ^ t →
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2850_285027


namespace NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_sin_36_l2850_285035

theorem cos_24_cos_36_minus_sin_24_sin_36 :
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.sin (36 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_24_cos_36_minus_sin_24_sin_36_l2850_285035


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l2850_285091

theorem chinese_chess_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.6) 
  (h_not_lose : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l2850_285091


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_all_ones_sequence_l2850_285058

/-- Represents a number in the sequence 11, 111, 1111, ... -/
def allOnesNumber (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

/-- Theorem: There are no perfect squares in the sequence of numbers
    consisting of only the digit 1, starting from 11 -/
theorem no_perfect_squares_in_all_ones_sequence :
  ∀ n : ℕ, n ≥ 2 → ¬ isPerfectSquare (allOnesNumber n) :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_all_ones_sequence_l2850_285058


namespace NUMINAMATH_CALUDE_alloy_mixture_theorem_l2850_285049

/-- Represents an alloy with given ratios of gold, silver, and copper -/
structure Alloy where
  gold : ℚ
  silver : ℚ
  copper : ℚ

/-- The total ratio of an alloy -/
def Alloy.total (a : Alloy) : ℚ := a.gold + a.silver + a.copper

/-- Create an alloy from integer ratios -/
def Alloy.fromRatios (g s c : ℕ) : Alloy :=
  let t : ℚ := (g + s + c : ℚ)
  { gold := g / t, silver := s / t, copper := c / t }

theorem alloy_mixture_theorem (x y z : ℚ) :
  let a1 := Alloy.fromRatios 1 3 5
  let a2 := Alloy.fromRatios 3 5 1
  let a3 := Alloy.fromRatios 5 1 3
  let total_mass : ℚ := 351
  let desired_ratio := Alloy.fromRatios 7 9 11
  x = 195 ∧ y = 78 ∧ z = 78 →
  x + y + z = total_mass ∧
  (x * a1.gold + y * a2.gold + z * a3.gold) / total_mass = desired_ratio.gold ∧
  (x * a1.silver + y * a2.silver + z * a3.silver) / total_mass = desired_ratio.silver ∧
  (x * a1.copper + y * a2.copper + z * a3.copper) / total_mass = desired_ratio.copper := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_theorem_l2850_285049


namespace NUMINAMATH_CALUDE_area_ratio_is_one_seventh_l2850_285067

/-- Given a triangle XYZ with sides XY, YZ, XZ and points P on XY and Q on XZ,
    this function calculates the ratio of the area of triangle XPQ to the area of quadrilateral PQYZ -/
def areaRatio (XY YZ XZ XP XQ : ℝ) : ℝ :=
  -- Define the ratio calculation here
  sorry

/-- Theorem stating that for the given triangle and points, the area ratio is 1/7 -/
theorem area_ratio_is_one_seventh :
  areaRatio 24 52 60 12 20 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_seventh_l2850_285067


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l2850_285021

theorem fraction_product_equals_one : 
  (36 : ℚ) / 34 * 26 / 48 * 136 / 78 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l2850_285021


namespace NUMINAMATH_CALUDE_valid_three_digit_count_l2850_285011

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 90

/-- The total count of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The count of three-digit numbers with exactly two different non-adjacent digits -/
def excluded_count : ℕ := 810

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count :
  valid_count = total_three_digit - excluded_count :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_l2850_285011


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2850_285070

theorem necessary_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b ∨ c = 0) ∧
  (∃ a b c : ℝ, a * c^2 > b * c^2 ∧ a ≤ b) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2850_285070


namespace NUMINAMATH_CALUDE_equation_solution_l2850_285084

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (27 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2850_285084


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2850_285077

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (k : ℝ), (2 - 7*I) * (a + b*I) = k*I) : a/b = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2850_285077


namespace NUMINAMATH_CALUDE_system_solution_l2850_285004

theorem system_solution (x y a : ℝ) : 
  x - 2*y = a - 6 →
  2*x + 5*y = 2*a →
  x + y = 9 →
  a = 11 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2850_285004


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2850_285081

theorem quadratic_equation_result (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) : 
  (14 * y - 5)^2 = 333 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2850_285081


namespace NUMINAMATH_CALUDE_sum_not_zero_l2850_285047

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) : 
  a + b + c + d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_not_zero_l2850_285047


namespace NUMINAMATH_CALUDE_bug_triangle_probability_l2850_285079

/-- Probability of the bug being at the starting vertex after n moves -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => (1 - P n) / 2

/-- The bug's movement on an equilateral triangle -/
theorem bug_triangle_probability :
  P 12 = 683 / 2048 :=
by sorry

end NUMINAMATH_CALUDE_bug_triangle_probability_l2850_285079


namespace NUMINAMATH_CALUDE_plaid_shirts_count_l2850_285078

/-- Prove that the number of plaid shirts is 3 -/
theorem plaid_shirts_count (total_shirts : ℕ) (total_pants : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ) : ℕ :=
  by
  have total_items : ℕ := total_shirts + total_pants
  have plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
  have plaid_shirts : ℕ := plaid_or_purple - purple_pants
  exact plaid_shirts

#check plaid_shirts_count 5 24 5 21

end NUMINAMATH_CALUDE_plaid_shirts_count_l2850_285078


namespace NUMINAMATH_CALUDE_total_marks_math_physics_l2850_285001

/-- Proves that the total marks in mathematics and physics is 60 -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 40 →
  M + P = 60 := by
sorry

end NUMINAMATH_CALUDE_total_marks_math_physics_l2850_285001


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2850_285072

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 3 * x₁ - 4 = 0) ∧ (3 * x₂^2 - 3 * x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2850_285072


namespace NUMINAMATH_CALUDE_triangle_side_constraint_l2850_285075

theorem triangle_side_constraint (a : ℝ) : 
  (0 < a) → (0 < 2) → (0 < 6) → 
  (2 + 6 > a) → (6 + a > 2) → (2 + a > 6) → 
  (4 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_constraint_l2850_285075


namespace NUMINAMATH_CALUDE_flower_combination_l2850_285029

theorem flower_combination : Nat.choose 10 6 = 210 := by
  sorry

end NUMINAMATH_CALUDE_flower_combination_l2850_285029


namespace NUMINAMATH_CALUDE_doll_factory_operation_time_l2850_285024

/-- Calculate the total machine operation time for dolls and accessories -/
theorem doll_factory_operation_time :
  let num_dolls : ℕ := 12000
  let shoes_per_doll : ℕ := 2
  let bags_per_doll : ℕ := 3
  let cosmetics_per_doll : ℕ := 1
  let hats_per_doll : ℕ := 5
  let doll_production_time : ℕ := 45
  let accessory_production_time : ℕ := 10

  let total_accessories : ℕ := num_dolls * (shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll)
  let doll_time : ℕ := num_dolls * doll_production_time
  let accessory_time : ℕ := total_accessories * accessory_production_time
  let total_time : ℕ := doll_time + accessory_time

  total_time = 1860000 := by
  sorry

end NUMINAMATH_CALUDE_doll_factory_operation_time_l2850_285024


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2850_285012

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  sphere_radius : ℝ
  tangent_to_top : Bool
  tangent_to_bottom : Bool
  tangent_to_lateral : Bool

/-- The theorem stating the radius of the sphere in a specific truncated cone configuration -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottom_radius = 12)
  (h2 : cone.top_radius = 3)
  (h3 : cone.tangent_to_top = true)
  (h4 : cone.tangent_to_bottom = true)
  (h5 : cone.tangent_to_lateral = true) :
  cone.sphere_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2850_285012


namespace NUMINAMATH_CALUDE_quadratic_properties_l2850_285037

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate that checks if x is in the solution set (2, 3) -/
def inSolutionSet (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- The quadratic function is positive in the interval (2, 3) -/
def isPositiveInInterval (f : QuadraticFunction) : Prop :=
  ∀ x, inSolutionSet x → f.a * x^2 + f.b * x + f.c > 0

theorem quadratic_properties (f : QuadraticFunction) 
  (h : isPositiveInInterval f) : 
  f.a < 0 ∧ f.b * f.c < 0 ∧ f.b + f.c = f.a ∧ f.a - f.b + f.c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2850_285037


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2850_285046

/-- Given a geometric sequence {aₙ} where all terms are positive,
    prove that a₅a₇a₉ = 12 when a₂a₄a₆ = 6 and a₈a₁₀a₁₂ = 24 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2850_285046


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l2850_285053

theorem right_triangle_sin_A (A B C : Real) (h1 : 3 * Real.sin A = 2 * Real.cos A) 
  (h2 : Real.cos B = 0) : Real.sin A = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l2850_285053


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_correct_l2850_285016

/-- The number of times Billy rode the ferris wheel -/
def ferris_wheel_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_car_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

/-- Theorem stating that the number of ferris wheel rides is correct -/
theorem ferris_wheel_rides_correct : 
  ferris_wheel_rides * cost_per_ride + bumper_car_rides * cost_per_ride = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_correct_l2850_285016


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2850_285030

theorem sum_of_roots_quadratic (x : ℝ) : 
  let a : ℝ := 3
  let b : ℝ := -12
  let c : ℝ := 12
  let sum_of_roots := -b / a
  (3 * x^2 - 12 * x + 12 = 0) → sum_of_roots = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2850_285030


namespace NUMINAMATH_CALUDE_circle_points_l2850_285074

theorem circle_points (π : ℝ) (h : π > 0) : 
  let radii : List ℝ := [1.5, 2, 3.5, 4.5, 5.5]
  let circumference (r : ℝ) := 2 * π * r
  let area (r : ℝ) := π * r^2
  let points := radii.map (λ r => (circumference r, area r))
  points = [(3*π, 2.25*π), (4*π, 4*π), (7*π, 12.25*π), (9*π, 20.25*π), (11*π, 30.25*π)] := by
  sorry

end NUMINAMATH_CALUDE_circle_points_l2850_285074


namespace NUMINAMATH_CALUDE_gcd_lcm_product_180_l2850_285065

def count_gcd_values (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ),
    (∀ a b : ℕ, (Nat.gcd a b) * (Nat.lcm a b) = n →
      (Nat.gcd a b) ∈ S) ∧
    S.card = 8

theorem gcd_lcm_product_180 :
  count_gcd_values 180 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_180_l2850_285065


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l2850_285096

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * Real.log x) x = Real.log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l2850_285096


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2850_285050

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 2 → y = 2 * 2 + 3) : f 2 + (deriv f) 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2850_285050


namespace NUMINAMATH_CALUDE_car_speed_calculation_l2850_285097

theorem car_speed_calculation (D : ℝ) (h_D_pos : D > 0) : ∃ v : ℝ,
  (D / ((0.8 * D / 80) + (0.2 * D / v)) = 50) → v = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_calculation_l2850_285097


namespace NUMINAMATH_CALUDE_parsley_rows_juvy_parsley_rows_l2850_285005

/-- Calculates the number of rows planted with parsley in Juvy's garden. -/
theorem parsley_rows (total_rows : Nat) (plants_per_row : Nat) (rosemary_rows : Nat) (chives_count : Nat) : Nat :=
  let remaining_rows := total_rows - rosemary_rows
  let chives_rows := chives_count / plants_per_row
  remaining_rows - chives_rows

/-- Proves that Juvy plants parsley in 3 rows given the garden's conditions. -/
theorem juvy_parsley_rows : parsley_rows 20 10 2 150 = 3 := by
  sorry

end NUMINAMATH_CALUDE_parsley_rows_juvy_parsley_rows_l2850_285005


namespace NUMINAMATH_CALUDE_g_three_properties_l2850_285051

/-- A function satisfying the given condition for all real x and y -/
def special_function (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = x + y + 1

/-- The theorem stating the properties of g(3) -/
theorem g_three_properties (g : ℝ → ℝ) (h : special_function g) :
  (∃ a b : ℝ, (∀ x : ℝ, g 3 = x → (x = a ∨ x = b)) ∧ a + b = 0) :=
sorry

end NUMINAMATH_CALUDE_g_three_properties_l2850_285051


namespace NUMINAMATH_CALUDE_base_number_proof_l2850_285069

theorem base_number_proof (x y : ℝ) (h1 : x ^ y = 3 ^ 16) (h2 : y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2850_285069


namespace NUMINAMATH_CALUDE_quadratic_function_k_l2850_285009

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem quadratic_function_k (a b c k : ℤ) : 
  g a b c (-1) = 0 → 
  (30 < g a b c 5 ∧ g a b c 5 < 40) → 
  (120 < g a b c 7 ∧ g a b c 7 < 130) → 
  (2000 * k < g a b c 50 ∧ g a b c 50 < 2000 * (k + 1)) → 
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_l2850_285009


namespace NUMINAMATH_CALUDE_cricket_overs_played_l2850_285068

/-- Proves that the number of overs played initially in a cricket game is 10, given the specified conditions --/
theorem cricket_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) : 
  target = 242 ∧ initial_rate = 3.2 ∧ required_rate = 5.25 →
  ∃ x : ℝ, x = 10 ∧ target - initial_rate * x = required_rate * (50 - x) := by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_played_l2850_285068


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2850_285060

theorem rational_equation_solution : ∃ x : ℚ, 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) ∧ 
  x = 55/13 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2850_285060


namespace NUMINAMATH_CALUDE_problem_statement_l2850_285095

theorem problem_statement (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2850_285095


namespace NUMINAMATH_CALUDE_smallest_number_divisible_after_increase_l2850_285057

theorem smallest_number_divisible_after_increase : ∃ (k : ℕ), 
  (∀ (n : ℕ), n < 3153 → ¬∃ (m : ℕ), (n + m) % 18 = 0 ∧ (n + m) % 70 = 0 ∧ (n + m) % 25 = 0 ∧ (n + m) % 21 = 0) ∧
  (∃ (m : ℕ), (3153 + m) % 18 = 0 ∧ (3153 + m) % 70 = 0 ∧ (3153 + m) % 25 = 0 ∧ (3153 + m) % 21 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_after_increase_l2850_285057


namespace NUMINAMATH_CALUDE_proportion_solution_l2850_285042

/-- Given a proportion x : 10 :: 8 : 0.6, prove that x = 400/3 -/
theorem proportion_solution (x : ℚ) : (x / 10 = 8 / (3/5)) → x = 400/3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2850_285042


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l2850_285099

/-- Edward's lawn mowing earnings problem -/
theorem lawn_mowing_earnings 
  (total_earnings summer_earnings spring_earnings supplies_cost end_amount : ℕ)
  (h1 : total_earnings = summer_earnings + spring_earnings)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5)
  (h4 : end_amount = 24)
  (h5 : total_earnings = end_amount + supplies_cost) :
  spring_earnings = 2 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l2850_285099


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2850_285000

theorem gcd_digits_bound (a b : ℕ) : 
  100000 ≤ a ∧ a < 1000000 ∧ 
  100000 ≤ b ∧ b < 1000000 ∧ 
  1000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10000000000 →
  Nat.gcd a b < 1000 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2850_285000


namespace NUMINAMATH_CALUDE_reader_count_l2850_285054

/-- The number of readers who read science fiction -/
def science_fiction_readers : ℕ := 120

/-- The number of readers who read literary works -/
def literary_works_readers : ℕ := 90

/-- The number of readers who read both science fiction and literary works -/
def both_genres_readers : ℕ := 60

/-- The total number of readers in the group -/
def total_readers : ℕ := science_fiction_readers + literary_works_readers - both_genres_readers

theorem reader_count : total_readers = 150 := by
  sorry

end NUMINAMATH_CALUDE_reader_count_l2850_285054


namespace NUMINAMATH_CALUDE_painted_cube_equality_l2850_285062

/-- Represents a cube with edge length n, painted with alternating colors on adjacent faces. -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of unit cubes with exactly one black face in a painted cube. -/
def black_face_count (cube : PaintedCube) : ℕ :=
  3 * (cube.n - 2)^2

/-- The number of unpainted unit cubes in a painted cube. -/
def unpainted_count (cube : PaintedCube) : ℕ :=
  (cube.n - 2)^3

/-- Theorem stating that the number of unit cubes with exactly one black face
    equals the number of unpainted unit cubes if and only if n = 5. -/
theorem painted_cube_equality (cube : PaintedCube) :
  black_face_count cube = unpainted_count cube ↔ cube.n = 5 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_equality_l2850_285062


namespace NUMINAMATH_CALUDE_day_250_is_tuesday_l2850_285026

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def dayOfWeek (n : Nat) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem day_250_is_tuesday (h : dayOfWeek 35 = DayOfWeek.Wednesday) :
  dayOfWeek 250 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_day_250_is_tuesday_l2850_285026


namespace NUMINAMATH_CALUDE_hundredth_group_sum_divided_by_100_l2850_285025

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth group -/
def last_number (n : ℕ) : ℕ := 2 * sum_of_naturals n

/-- The first number in the nth group -/
def first_number (n : ℕ) : ℕ := last_number (n - 1) - 2 * (n - 1)

/-- The sum of numbers in the nth group -/
def group_sum (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem hundredth_group_sum_divided_by_100 :
  group_sum 100 / 100 = 10001 := by sorry

end NUMINAMATH_CALUDE_hundredth_group_sum_divided_by_100_l2850_285025


namespace NUMINAMATH_CALUDE_triangle_inequality_max_l2850_285087

theorem triangle_inequality_max (a b c x y z : ℝ) 
  (triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (positive : 0 < x ∧ 0 < y ∧ 0 < z) 
  (sum_one : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ 
    (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_max_l2850_285087


namespace NUMINAMATH_CALUDE_greatest_b_value_l2850_285033

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ 7) ∧ 
  (7^2 - 12*7 + 35 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2850_285033


namespace NUMINAMATH_CALUDE_ratio_change_l2850_285048

theorem ratio_change (x : ℤ) : 
  (4 * x = 16) →  -- The larger integer is 16, and it's 4 times the smaller integer
  ((x + 12) / (4 * x) = 1) -- The new ratio after adding 12 to the smaller integer is 1:1
:= by sorry

end NUMINAMATH_CALUDE_ratio_change_l2850_285048


namespace NUMINAMATH_CALUDE_reflection_of_line_l2850_285082

/-- Given a line with equation 2x + 3y - 5 = 0, its reflection about the line y = x
    is the line with equation 3x + 2y - 5 = 0 -/
theorem reflection_of_line :
  let original_line : ℝ → ℝ → Prop := λ x y ↦ 2*x + 3*y - 5 = 0
  let reflection_axis : ℝ → ℝ → Prop := λ x y ↦ y = x
  let reflected_line : ℝ → ℝ → Prop := λ x y ↦ 3*x + 2*y - 5 = 0
  ∀ (x y : ℝ), original_line x y ↔ reflected_line y x :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_line_l2850_285082


namespace NUMINAMATH_CALUDE_ethanol_in_tank_l2850_285055

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem stating that the total ethanol in the given scenario is 30 gallons -/
theorem ethanol_in_tank : 
  total_ethanol 204 66 0.12 0.16 = 30 := by
  sorry

#eval total_ethanol 204 66 0.12 0.16

end NUMINAMATH_CALUDE_ethanol_in_tank_l2850_285055


namespace NUMINAMATH_CALUDE_student_average_score_l2850_285085

/-- Given a student's scores in physics, chemistry, and mathematics, prove that the average of all three subjects is 60. -/
theorem student_average_score (P C M : ℝ) : 
  P = 140 →                -- Physics score
  (P + M) / 2 = 90 →       -- Average of physics and mathematics
  (P + C) / 2 = 70 →       -- Average of physics and chemistry
  (P + C + M) / 3 = 60 :=  -- Average of all three subjects
by
  sorry


end NUMINAMATH_CALUDE_student_average_score_l2850_285085


namespace NUMINAMATH_CALUDE_sequence_general_term_l2850_285073

theorem sequence_general_term (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ k, S k = k^2 + k) →
  (a 1 = S 1) →
  (∀ k ≥ 2, a k = S k - S (k - 1)) →
  ∀ k, a k = 2 * k :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2850_285073


namespace NUMINAMATH_CALUDE_tims_dimes_count_l2850_285043

/-- Represents the number of coins of each type --/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  halfDollars : ℕ

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : ℚ :=
  0.05 * c.nickels + 0.10 * c.dimes + 0.50 * c.halfDollars

/-- Represents Tim's earnings from shining shoes and tips --/
structure TimsEarnings where
  shoeShining : CoinCount
  tipJar : CoinCount

/-- The main theorem to prove --/
theorem tims_dimes_count 
  (earnings : TimsEarnings)
  (h1 : earnings.shoeShining.nickels = 3)
  (h2 : earnings.tipJar.dimes = 7)
  (h3 : earnings.tipJar.halfDollars = 9)
  (h4 : coinValue earnings.shoeShining + coinValue earnings.tipJar = 6.65) :
  earnings.shoeShining.dimes = 13 :=
by sorry

end NUMINAMATH_CALUDE_tims_dimes_count_l2850_285043


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2850_285008

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2850_285008


namespace NUMINAMATH_CALUDE_probability_of_C_l2850_285032

/-- A board game spinner with six regions -/
structure Spinner :=
  (probA : ℚ)
  (probB : ℚ)
  (probC : ℚ)
  (probD : ℚ)
  (probE : ℚ)
  (probF : ℚ)

/-- The conditions of the spinner -/
def spinnerConditions (s : Spinner) : Prop :=
  s.probA = 2/9 ∧
  s.probB = 1/6 ∧
  s.probC = s.probD ∧
  s.probC = s.probE ∧
  s.probF = 2 * s.probC ∧
  s.probA + s.probB + s.probC + s.probD + s.probE + s.probF = 1

/-- The theorem stating the probability of region C -/
theorem probability_of_C (s : Spinner) (h : spinnerConditions s) : s.probC = 11/90 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_C_l2850_285032


namespace NUMINAMATH_CALUDE_second_year_increase_is_fifteen_percent_l2850_285045

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  ((final_population : ℚ) / population_after_first_year - 1) * 100

/-- Theorem stating that given the specific conditions of the problem,
    the second year increase is 15%. -/
theorem second_year_increase_is_fifteen_percent :
  second_year_increase 800 (25 / 100) 1150 = 15 := by
  sorry

#eval second_year_increase 800 (25 / 100) 1150

end NUMINAMATH_CALUDE_second_year_increase_is_fifteen_percent_l2850_285045


namespace NUMINAMATH_CALUDE_bianca_candy_problem_l2850_285007

/-- Bianca's Halloween candy problem -/
theorem bianca_candy_problem (initial_candy : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h1 : initial_candy = 32)
  (h2 : piles = 4)
  (h3 : pieces_per_pile = 5) :
  initial_candy - (piles * pieces_per_pile) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bianca_candy_problem_l2850_285007


namespace NUMINAMATH_CALUDE_D_is_empty_l2850_285010

-- Define the set D
def D : Set ℝ := {x : ℝ | x^2 + 2 = 0}

-- Theorem stating that D is an empty set
theorem D_is_empty : D = ∅ := by sorry

end NUMINAMATH_CALUDE_D_is_empty_l2850_285010


namespace NUMINAMATH_CALUDE_max_train_collection_l2850_285086

/-- The number of trains Max receives each year --/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains --/
def collection_years : ℕ := 5

/-- The total number of trains Max has after the collection period --/
def initial_trains : ℕ := trains_per_year * collection_years

/-- The factor by which Max's train collection is multiplied at the end --/
def doubling_factor : ℕ := 2

/-- The final number of trains Max has --/
def final_trains : ℕ := initial_trains * doubling_factor

theorem max_train_collection :
  final_trains = 30 :=
sorry

end NUMINAMATH_CALUDE_max_train_collection_l2850_285086


namespace NUMINAMATH_CALUDE_insurance_slogan_equivalence_l2850_285059

-- Define the universe of people
variable (Person : Type)

-- Define predicates
variable (happy : Person → Prop)
variable (has_it : Person → Prop)

-- Theorem stating the logical equivalence
theorem insurance_slogan_equivalence :
  (∀ p : Person, happy p → has_it p) ↔ (∀ p : Person, ¬has_it p → ¬happy p) :=
sorry

end NUMINAMATH_CALUDE_insurance_slogan_equivalence_l2850_285059


namespace NUMINAMATH_CALUDE_dozens_of_eggs_l2850_285083

def eggs_bought : ℕ := 72
def eggs_per_dozen : ℕ := 12

theorem dozens_of_eggs : eggs_bought / eggs_per_dozen = 6 := by
  sorry

end NUMINAMATH_CALUDE_dozens_of_eggs_l2850_285083


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2850_285040

/-- Given four intersecting squares with sides 12, 9, 7, and 3,
    the difference between the sum of the areas of the largest and third largest squares
    and the sum of the areas of the second largest and smallest squares is 103. -/
theorem intersecting_squares_area_difference : 
  let a := 12 -- side length of the largest square
  let b := 9  -- side length of the second largest square
  let c := 7  -- side length of the third largest square
  let d := 3  -- side length of the smallest square
  (a ^ 2 + c ^ 2) - (b ^ 2 + d ^ 2) = 103 := by
sorry

#eval (12 ^ 2 + 7 ^ 2) - (9 ^ 2 + 3 ^ 2) -- This should evaluate to 103

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2850_285040


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l2850_285098

/-- The area of a parallelogram with base 12 cm and height 10 cm is 120 cm². -/
theorem parallelogram_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 12 ∧ height = 10 → area = base * height → area = 120

#check parallelogram_area

-- Proof
theorem parallelogram_area_proof : parallelogram_area 12 10 120 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l2850_285098


namespace NUMINAMATH_CALUDE_system_solution_l2850_285031

theorem system_solution (a b c x y z : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (x * y + y * z + z * x = a^2 - x^2) ∧ 
  (x * y + y * z + z * x = b^2 - y^2) ∧ 
  (x * y + y * z + z * x = c^2 - z^2) →
  ((x = (|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = (|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = (-|a*b/c| + |a*c/b| + |b*c/a|) / 2) ∨
   (x = -(|a*b/c| + |a*c/b| - |b*c/a|) / 2 ∧
    y = -(|a*b/c| - |a*c/b| + |b*c/a|) / 2 ∧
    z = -(-|a*b/c| + |a*c/b| + |b*c/a|) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2850_285031


namespace NUMINAMATH_CALUDE_polynomial_multiplication_identity_l2850_285013

theorem polynomial_multiplication_identity (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_identity_l2850_285013


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l2850_285088

-- Define the triangle ABC inscribed in a unit circle
def Triangle (A B C : ℝ) := True

-- Define the area of a triangle
def area (a b c : ℝ) : ℝ := sorry

-- Define the sine function
noncomputable def sin (θ : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_ratio 
  (A B C : ℝ) 
  (h : Triangle A B C) :
  area (sin A) (sin B) (sin C) = (1/4 : ℝ) * area (2 * sin A) (2 * sin B) (2 * sin C) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l2850_285088


namespace NUMINAMATH_CALUDE_exactly_five_false_propositions_l2850_285090

-- Define the geometric objects
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry
def Angle : Type := sorry

-- Define geometric relations
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def coplanar (l1 l2 l3 : Line) : Prop := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def commonPoint (pl1 pl2 : Plane) (p : Point) : Prop := sorry
def sidesParallel (a1 a2 : Angle) : Prop := sorry

-- Define the propositions
def prop1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def prop2 : Prop := ∀ l1 l2 l3 : Line, intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 → coplanar l1 l2 l3
def prop3 : Prop := ∀ p1 p2 p3 p4 : Point, (∃ pl : Plane, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl)) → ¬collinear p1 p2 p3 ∧ ¬collinear p1 p2 p4 ∧ ¬collinear p1 p3 p4 ∧ ¬collinear p2 p3 p4
def prop4 : Prop := ∀ pl1 pl2 : Plane, (∃ p1 p2 p3 : Point, commonPoint pl1 pl2 p1 ∧ commonPoint pl1 pl2 p2 ∧ commonPoint pl1 pl2 p3) → pl1 = pl2
def prop5 : Prop := ∃ α β : Plane, ∃! p : Point, commonPoint α β p
def prop6 : Prop := ∀ a1 a2 : Angle, sidesParallel a1 a2 → a1 = a2

-- Theorem statement
theorem exactly_five_false_propositions : 
  ¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4 ∧ ¬prop5 ∧ ¬prop6 := by sorry

end NUMINAMATH_CALUDE_exactly_five_false_propositions_l2850_285090


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2850_285039

variable (i : ℂ)
variable (z : ℂ)

theorem complex_equation_solution (hi : i * i = -1) (hz : (1 + i) / z = 1 - i) : z = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2850_285039


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2850_285019

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (n * (n + 1) * (n + 2) = 336) → (n + (n + 1) + (n + 2) = 21) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2850_285019


namespace NUMINAMATH_CALUDE_bill_donut_order_ways_l2850_285015

/-- The number of ways to distribute identical items into distinct groups -/
def distribute_items (items : ℕ) (groups : ℕ) : ℕ :=
  Nat.choose (items + groups - 1) (groups - 1)

/-- The number of ways Bill can fulfill his donut order -/
theorem bill_donut_order_ways : distribute_items 3 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bill_donut_order_ways_l2850_285015


namespace NUMINAMATH_CALUDE_unique_orthocenter_line_l2850_285041

/-- The ellipse with equation x^2/2 + y^2 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- The upper vertex of the ellipse -/
def B : ℝ × ℝ := (0, 1)

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- A line that intersects the ellipse -/
def line_intersects_ellipse (m b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b

/-- F is the orthocenter of triangle BMN -/
def F_is_orthocenter (M N : ℝ × ℝ) : Prop :=
  let (xm, ym) := M
  let (xn, yn) := N
  (1 - xn) * xm - yn * (ym - 1) = 0 ∧
  (1 - xm) * xn - ym * (yn - 1) = 0

theorem unique_orthocenter_line :
  ∃! m b : ℝ, 
    line_intersects_ellipse m b ∧
    (∀ M N : ℝ × ℝ, 
      ellipse M.1 M.2 → ellipse N.1 N.2 → 
      M.2 = m * M.1 + b → N.2 = m * N.1 + b →
      F_is_orthocenter M N) ∧
    m = 1 ∧ b = -4/3 :=
sorry

end NUMINAMATH_CALUDE_unique_orthocenter_line_l2850_285041


namespace NUMINAMATH_CALUDE_not_all_six_multiples_have_prime_neighbor_l2850_285044

theorem not_all_six_multiples_have_prime_neighbor :
  ∃ n : ℕ, 6 ∣ n ∧ ¬(Nat.Prime (n - 1) ∨ Nat.Prime (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_six_multiples_have_prime_neighbor_l2850_285044


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2850_285034

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * p * (1 - p) * (1 - p) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2850_285034


namespace NUMINAMATH_CALUDE_nectar_water_percentage_l2850_285056

/-- The ratio of flower-nectar to honey produced -/
def nectarToHoneyRatio : ℝ := 1.6

/-- The percentage of water in the produced honey -/
def honeyWaterPercentage : ℝ := 20

/-- The percentage of water in flower-nectar -/
def nectarWaterPercentage : ℝ := 50

theorem nectar_water_percentage :
  nectarWaterPercentage = 100 * (nectarToHoneyRatio - (1 - honeyWaterPercentage / 100)) / nectarToHoneyRatio :=
by sorry

end NUMINAMATH_CALUDE_nectar_water_percentage_l2850_285056


namespace NUMINAMATH_CALUDE_student_allowance_proof_l2850_285063

/-- The student's weekly allowance in dollars -/
def weekly_allowance : ℝ := 4.50

theorem student_allowance_proof :
  ∃ (arcade_spent toy_store_spent : ℝ),
    arcade_spent = (3/5) * weekly_allowance ∧
    toy_store_spent = (1/3) * (weekly_allowance - arcade_spent) ∧
    weekly_allowance - arcade_spent - toy_store_spent = 1.20 :=
by sorry

end NUMINAMATH_CALUDE_student_allowance_proof_l2850_285063


namespace NUMINAMATH_CALUDE_athlete_A_one_win_one_loss_l2850_285093

/-- The probability of athlete A winning against athlete B -/
def prob_A_wins_B : ℝ := 0.8

/-- The probability of athlete A winning against athlete C -/
def prob_A_wins_C : ℝ := 0.7

/-- The probability of athlete A achieving one win and one loss -/
def prob_one_win_one_loss : ℝ := prob_A_wins_B * (1 - prob_A_wins_C) + (1 - prob_A_wins_B) * prob_A_wins_C

theorem athlete_A_one_win_one_loss : prob_one_win_one_loss = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_athlete_A_one_win_one_loss_l2850_285093


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l2850_285066

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180
  all_positive : ∀ i, 0 < angles i

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) :
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l2850_285066


namespace NUMINAMATH_CALUDE_tomatoes_left_l2850_285002

/-- Theorem: Given a farmer with 97 tomatoes who picks 83 tomatoes, the number of tomatoes left is equal to 14. -/
theorem tomatoes_left (total : ℕ) (picked : ℕ) (h1 : total = 97) (h2 : picked = 83) :
  total - picked = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l2850_285002


namespace NUMINAMATH_CALUDE_primes_between_50_and_60_l2850_285036

def count_primes_in_range (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

theorem primes_between_50_and_60 : count_primes_in_range 50 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_primes_between_50_and_60_l2850_285036


namespace NUMINAMATH_CALUDE_product_of_roots_l2850_285080

theorem product_of_roots (x : ℝ) : 
  (x^3 - 9*x^2 + 27*x - 64 = 0) → 
  (∃ p q r : ℝ, x^3 - 9*x^2 + 27*x - 64 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 64) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2850_285080


namespace NUMINAMATH_CALUDE_new_year_gift_exchange_l2850_285006

/-- Represents a group of friends exchanging gifts -/
structure GiftExchange where
  num_friends : Nat
  num_exchanges : Nat

/-- Predicate to check if the number of friends receiving 4 gifts is valid -/
def valid_four_gift_recipients (ge : GiftExchange) (n : Nat) : Prop :=
  n = 2 ∨ n = 4

/-- Theorem stating that in the given scenario, the number of friends receiving 4 gifts is either 2 or 4 -/
theorem new_year_gift_exchange (ge : GiftExchange) 
  (h1 : ge.num_friends = 6)
  (h2 : ge.num_exchanges = 13) :
  ∃ n : Nat, valid_four_gift_recipients ge n := by
  sorry

end NUMINAMATH_CALUDE_new_year_gift_exchange_l2850_285006


namespace NUMINAMATH_CALUDE_inequality_proof_l2850_285052

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (hd : 0 < d ∧ d < 1) : 
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2850_285052


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2850_285022

/-- Given a rectangle ACDE with AC = 48 and AE = 30, where point B divides AC in ratio 1:3
    and point F divides AE in ratio 2:3, the area of quadrilateral ABDF is equal to 468. -/
theorem area_of_quadrilateral (AC AE : ℝ) (B F : ℝ) : 
  AC = 48 → 
  AE = 30 → 
  B / AC = 1 / 4 → 
  F / AE = 2 / 5 → 
  (AC * AE) - (3 * AC * AE / 4) - (3 * AC * AE / 5) = 468 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2850_285022


namespace NUMINAMATH_CALUDE_taxi_overtakes_bus_l2850_285094

theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_delay : ℝ) (speed_difference : ℝ)
  (h1 : taxi_speed = 60)
  (h2 : bus_delay = 3)
  (h3 : speed_difference = 30) :
  let bus_speed := taxi_speed - speed_difference
  let overtake_time := (bus_speed * bus_delay) / (taxi_speed - bus_speed)
  overtake_time = 3 := by
sorry

end NUMINAMATH_CALUDE_taxi_overtakes_bus_l2850_285094
