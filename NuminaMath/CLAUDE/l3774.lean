import Mathlib

namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_eight_and_n_plus_three_l3774_377467

theorem gcd_n_cube_plus_eight_and_n_plus_three (n : ℕ) (h : n > 27) : 
  Nat.gcd (n^3 + 8) (n + 3) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_eight_and_n_plus_three_l3774_377467


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3774_377408

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a * b = 2) 
  (h2 : a - b = 3) : 
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3774_377408


namespace NUMINAMATH_CALUDE_simplify_expression_l3774_377425

theorem simplify_expression (x y z : ℝ) :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 30 * x - 10 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3774_377425


namespace NUMINAMATH_CALUDE_max_sum_product_sqrt_max_value_quarter_equality_condition_l3774_377456

theorem max_sum_product_sqrt (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  ∀ y1 y2 y3 y4 : ℝ, 
    y1 ≥ 0 → y2 ≥ 0 → y3 ≥ 0 → y4 ≥ 0 → 
    y1 + y2 + y3 + y4 = 1 → 
    sum_prod ≥ (y1 + y2) * Real.sqrt (y1 * y2) + 
               (y1 + y3) * Real.sqrt (y1 * y3) + 
               (y1 + y4) * Real.sqrt (y1 * y4) + 
               (y2 + y3) * Real.sqrt (y2 * y3) + 
               (y2 + y4) * Real.sqrt (y2 * y4) + 
               (y3 + y4) * Real.sqrt (y3 * y4) :=
by sorry

theorem max_value_quarter (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod ≤ 3/4 :=
by sorry

theorem equality_condition (x1 x2 x3 x4 : ℝ) 
  (non_neg : x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0) 
  (sum_constraint : x1 + x2 + x3 + x4 = 1) :
  let sum_prod := (x1 + x2) * Real.sqrt (x1 * x2) + 
                  (x1 + x3) * Real.sqrt (x1 * x3) + 
                  (x1 + x4) * Real.sqrt (x1 * x4) + 
                  (x2 + x3) * Real.sqrt (x2 * x3) + 
                  (x2 + x4) * Real.sqrt (x2 * x4) + 
                  (x3 + x4) * Real.sqrt (x3 * x4)
  sum_prod = 3/4 ↔ x1 = 1/4 ∧ x2 = 1/4 ∧ x3 = 1/4 ∧ x4 = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_product_sqrt_max_value_quarter_equality_condition_l3774_377456


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3774_377449

/-- Given a line l: 4x + 5y - 8 = 0 and a point A (3, 2), 
    the perpendicular line through A has the equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let l : Set (ℝ × ℝ) := {(x, y) | 4 * x + 5 * y - 8 = 0}
  let A : ℝ × ℝ := (3, 2)
  let perpendicular_line : Set (ℝ × ℝ) := {(x, y) | 4 * y - 5 * x + 7 = 0}
  (∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l ∧ p ≠ q →
    (A.1 - p.1) * (q.1 - p.1) + (A.2 - p.2) * (q.2 - p.2) = 0) ∧
  A ∈ perpendicular_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3774_377449


namespace NUMINAMATH_CALUDE_circular_table_seating_l3774_377493

theorem circular_table_seating (n : ℕ) (a : Fin (2*n) → Fin (2*n)) 
  (h_perm : Function.Bijective a) :
  ∃ i j : Fin (2*n), i ≠ j ∧ 
    (a i - a j : ℤ) % (2*n) = (i - j : ℤ) % (2*n) ∨
    (a i - a j : ℤ) % (2*n) = (i - j - 2*n : ℤ) % (2*n) :=
by sorry

end NUMINAMATH_CALUDE_circular_table_seating_l3774_377493


namespace NUMINAMATH_CALUDE_min_sum_dimensions_of_box_l3774_377479

/-- Given a rectangular box with positive integer dimensions and volume 2310,
    the minimum possible sum of its three dimensions is 42. -/
theorem min_sum_dimensions_of_box (l w h : ℕ+) : 
  l * w * h = 2310 → l.val + w.val + h.val ≥ 42 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_of_box_l3774_377479


namespace NUMINAMATH_CALUDE_valleyball_club_members_l3774_377499

/-- The cost of a pair of knee pads in dollars -/
def knee_pad_cost : ℕ := 6

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := knee_pad_cost + 7

/-- The cost of a wristband in dollars -/
def wristband_cost : ℕ := jersey_cost + 3

/-- The total cost for one member's equipment (indoor and outdoor sets) -/
def member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)

/-- The total cost for all members' equipment -/
def total_cost : ℕ := 4080

/-- The number of members in the Valleyball Volleyball Club -/
def club_members : ℕ := total_cost / member_cost

theorem valleyball_club_members : club_members = 58 := by
  sorry

end NUMINAMATH_CALUDE_valleyball_club_members_l3774_377499


namespace NUMINAMATH_CALUDE_pie_remainder_pie_problem_l3774_377492

theorem pie_remainder (carlos_portion : Real) (maria_fraction : Real) : Real :=
  let remaining_after_carlos := 1 - carlos_portion
  let maria_portion := maria_fraction * remaining_after_carlos
  let final_remainder := remaining_after_carlos - maria_portion
  
  final_remainder

theorem pie_problem :
  pie_remainder 0.6 0.25 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_pie_remainder_pie_problem_l3774_377492


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3774_377428

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3774_377428


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3774_377424

/-- The imaginary part of (1+2i) / (1-i)² is 1/2 -/
theorem imaginary_part_of_z : Complex.im ((1 + 2*Complex.I) / (1 - Complex.I)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3774_377424


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l3774_377401

theorem prime_equation_solutions (p : ℕ) :
  (Prime p ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p) ↔ p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l3774_377401


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l3774_377437

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two :
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l3774_377437


namespace NUMINAMATH_CALUDE_ratio_w_y_l3774_377432

-- Define the variables
variable (w x y z : ℚ)

-- Define the given ratios
def ratio_w_x : w / x = 5 / 4 := by sorry
def ratio_y_z : y / z = 4 / 3 := by sorry
def ratio_z_x : z / x = 1 / 8 := by sorry

-- Theorem to prove
theorem ratio_w_y (hw : w / x = 5 / 4) (hy : y / z = 4 / 3) (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ratio_w_y_l3774_377432


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3774_377439

theorem right_triangle_hypotenuse (a b c : ℝ) :
  -- Right triangle condition
  c^2 = a^2 + b^2 →
  -- Area condition
  (1/2) * a * b = 48 →
  -- Geometric mean condition
  (a * b)^(1/2) = 8 →
  -- Conclusion: hypotenuse length
  c = 4 * (13 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3774_377439


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3774_377416

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : 
  fourth_quadrant (2, -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3774_377416


namespace NUMINAMATH_CALUDE_two_thousand_thirteenth_underlined_pair_l3774_377491

/-- The sequence of n values where n and 3^n have the same units digit -/
def underlined_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => underlined_sequence n + 2

/-- The nth pair in the sequence of underlined pairs -/
def nth_underlined_pair (n : ℕ) : ℕ × ℕ :=
  let m := underlined_sequence (n - 1)
  (m, 3^m)

theorem two_thousand_thirteenth_underlined_pair :
  nth_underlined_pair 2013 = (4025, 3^4025) := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_thirteenth_underlined_pair_l3774_377491


namespace NUMINAMATH_CALUDE_expression_range_l3774_377494

/-- The quadratic equation in terms of x with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*(m-2)*x + m^2 + 4

/-- Predicate to check if the quadratic equation has two real roots -/
def has_two_real_roots (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

/-- The expression we want to find the range of -/
def expression (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂

theorem expression_range :
  ∀ m : ℝ, has_two_real_roots m →
    (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ 
      expression x₁ x₂ ≥ 4 ∧ 
      ∀ ε > 0, ∃ m' : ℝ, has_two_real_roots m' ∧ 
        ∃ y₁ y₂ : ℝ, quadratic m' y₁ = 0 ∧ quadratic m' y₂ = 0 ∧ 
          expression y₁ y₂ < 4 + ε) :=
sorry

end NUMINAMATH_CALUDE_expression_range_l3774_377494


namespace NUMINAMATH_CALUDE_sum_of_roots_gt_two_l3774_377402

noncomputable def f (x : ℝ) : ℝ := (x^2 - x) / Real.exp x

theorem sum_of_roots_gt_two (x₁ x₂ : ℝ) 
  (h₁ : f x₁ = (Real.log x₁ + 1) / Real.exp x₁)
  (h₂ : f x₂ = (Real.log x₂ + 1) / Real.exp x₂) :
  x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_gt_two_l3774_377402


namespace NUMINAMATH_CALUDE_max_no_draw_in_our_tournament_l3774_377462

/-- Represents a tic-tac-toe tournament --/
structure Tournament where
  participants : Nat
  total_points : Nat
  win_points : Nat
  draw_points : Nat

/-- The maximum number of participants who could have played without a draw --/
def max_no_draw (t : Tournament) : Nat :=
  sorry

/-- Our specific tournament --/
def our_tournament : Tournament :=
  { participants := 16
  , total_points := 550
  , win_points := 5
  , draw_points := 2 }

theorem max_no_draw_in_our_tournament :
  max_no_draw our_tournament = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_no_draw_in_our_tournament_l3774_377462


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3774_377458

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  Real.cos (Real.arccos (1/2)) = d / (2 * e) →
  Real.cos (Real.arccos (3/5)) = e / (2 * f) →
  Real.cos (Real.arccos (-1/8)) = f / (2 * d) →
  d + e + f ≥ 33 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3774_377458


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3774_377430

theorem inequality_and_equality_condition (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3774_377430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3774_377473

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The common difference of an arithmetic sequence. -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  common_difference a = 7 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3774_377473


namespace NUMINAMATH_CALUDE_value_of_a_l3774_377471

def U (a : ℝ) : Set ℝ := {2, 4, 3 - a^2}
def P (a : ℝ) : Set ℝ := {2, a^2 - a + 2}

theorem value_of_a : 
  ∃ (a : ℝ), (U a).diff (P a) = {-1} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l3774_377471


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l3774_377400

/-- The function f parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x_max x_min : ℝ), ∀ (x : ℝ), f a x ≤ f a x_max ∧ f a x_min ≤ f a x

/-- The main theorem -/
theorem f_max_min_implies_a_range :
  ∀ a : ℝ, has_max_and_min a → a > 2 ∨ a < -1 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l3774_377400


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l3774_377421

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l3774_377421


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_11_l3774_377451

theorem binomial_coefficient_20_11 :
  (Nat.choose 18 9 = 48620) →
  (Nat.choose 18 8 = 43758) →
  (Nat.choose 20 11 = 168168) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_11_l3774_377451


namespace NUMINAMATH_CALUDE_fifth_term_value_l3774_377422

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem fifth_term_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = 6 →
  a 3 + a 5 + a 7 = 78 →
  a 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_value_l3774_377422


namespace NUMINAMATH_CALUDE_team_ate_63_slices_l3774_377441

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Represents the number of pizzas of each size -/
structure PizzaCounts where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (counts : PizzaCounts) : Nat :=
  slices.extraLarge * counts.extraLarge +
  slices.large * counts.large +
  slices.medium * counts.medium

/-- Theorem stating that the team ate 63 slices of pizza -/
theorem team_ate_63_slices 
  (slices : PizzaSlices)
  (counts : PizzaCounts)
  (h1 : slices.extraLarge = 16)
  (h2 : slices.large = 12)
  (h3 : slices.medium = 8)
  (h4 : counts.extraLarge = 3)
  (h5 : counts.large = 2)
  (h6 : counts.medium = 1)
  (h7 : totalSlices slices counts - 17 = 63) :
  63 = totalSlices slices counts - 17 := by
  sorry

#eval totalSlices ⟨16, 12, 8⟩ ⟨3, 2, 1⟩ - 17

end NUMINAMATH_CALUDE_team_ate_63_slices_l3774_377441


namespace NUMINAMATH_CALUDE_elliot_average_speed_l3774_377472

/-- Calculates the average speed given initial and final odometer readings and time spent riding. -/
def average_speed (initial_reading final_reading : ℕ) (hours : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / hours

/-- Theorem stating that the average speed for the given conditions is 30 miles per hour. -/
theorem elliot_average_speed :
  average_speed 2002 2332 11 = 30 := by
  sorry

end NUMINAMATH_CALUDE_elliot_average_speed_l3774_377472


namespace NUMINAMATH_CALUDE_invitation_ways_l3774_377443

-- Define the total number of classmates
def total_classmates : ℕ := 10

-- Define the number of classmates to invite
def invited_classmates : ℕ := 6

-- Define the number of classmates excluding A and B
def remaining_classmates : ℕ := total_classmates - 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem invitation_ways : 
  combination remaining_classmates (invited_classmates - 2) + 
  combination remaining_classmates invited_classmates = 98 := by
sorry

end NUMINAMATH_CALUDE_invitation_ways_l3774_377443


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l3774_377423

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l3774_377423


namespace NUMINAMATH_CALUDE_salary_increase_proof_l3774_377415

theorem salary_increase_proof (S : ℝ) (P : ℝ) : 
  S > 0 →
  0.06 * S > 0 →
  0.10 * (S * (1 + P / 100)) = 1.8333333333333331 * (0.06 * S) →
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l3774_377415


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3774_377454

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  (x + 2) / 2 ≥ (2 * x + m) / 3 + 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ 8

-- Theorem statement
theorem inequality_solution_implies_m_value :
  (∀ x, inequality x m ↔ solution_set x) → 2^m = (1 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l3774_377454


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3774_377463

theorem profit_percentage_calculation (original_cost selling_price : ℝ) :
  original_cost = 3000 →
  selling_price = 3450 →
  (selling_price - original_cost) / original_cost * 100 = 15 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3774_377463


namespace NUMINAMATH_CALUDE_simplify_expression_l3774_377411

theorem simplify_expression (a : ℝ) : 5*a^2 - (a^2 - 2*(a^2 - 3*a)) = 6*a^2 - 6*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3774_377411


namespace NUMINAMATH_CALUDE_max_multiplication_table_sum_l3774_377497

theorem max_multiplication_table_sum (numbers : Finset ℕ) : 
  numbers = {2, 3, 5, 7, 11, 17} →
  (∃ (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = numbers),
    ∀ (x y z u v w : ℕ) (h' : {x, y, z, u, v, w} = numbers),
      a * d + a * e + a * f + b * d + b * e + b * f + c * d + c * e + c * f ≥ 
      x * u + x * v + x * w + y * u + y * v + y * w + z * u + z * v + z * w) →
  (∃ (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = numbers),
    a * d + a * e + a * f + b * d + b * e + b * f + c * d + c * e + c * f = 450) := by
  sorry

end NUMINAMATH_CALUDE_max_multiplication_table_sum_l3774_377497


namespace NUMINAMATH_CALUDE_xyz_values_l3774_377466

theorem xyz_values (x y z : ℝ) 
  (eq1 : x * y - 5 * y = 20)
  (eq2 : y * z - 5 * z = 20)
  (eq3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := by
sorry

end NUMINAMATH_CALUDE_xyz_values_l3774_377466


namespace NUMINAMATH_CALUDE_math_score_proof_l3774_377417

def science : ℕ := 65
def social_studies : ℕ := 82
def english : ℕ := 47
def biology : ℕ := 85
def average : ℕ := 71
def total_subjects : ℕ := 5

theorem math_score_proof :
  ∃ (math : ℕ), 
    (science + social_studies + english + biology + math) / total_subjects = average ∧
    math = 76 := by
  sorry

end NUMINAMATH_CALUDE_math_score_proof_l3774_377417


namespace NUMINAMATH_CALUDE_ten_boys_handshakes_l3774_377477

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem: When 10 boys each shake hands once with every other boy, there are 45 handshakes -/
theorem ten_boys_handshakes : handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_boys_handshakes_l3774_377477


namespace NUMINAMATH_CALUDE_debby_dvd_sale_l3774_377433

theorem debby_dvd_sale (original : ℕ) (left : ℕ) (sold : ℕ) : 
  original = 13 → left = 7 → sold = original - left → sold = 6 := by sorry

end NUMINAMATH_CALUDE_debby_dvd_sale_l3774_377433


namespace NUMINAMATH_CALUDE_solve_equation_l3774_377409

theorem solve_equation (x : ℝ) (h : Real.sqrt (3 / x + 1) = 5 / 3) : x = 27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3774_377409


namespace NUMINAMATH_CALUDE_baking_contest_votes_l3774_377403

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

end NUMINAMATH_CALUDE_baking_contest_votes_l3774_377403


namespace NUMINAMATH_CALUDE_derivative_y_l3774_377468

noncomputable def y (x : ℝ) : ℝ := (Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4)

theorem derivative_y (x : ℝ) (h : Real.cos (2 * x) ≠ 0) :
  deriv y x = -(Real.cos (2 * x)) ^ ((Real.log (Real.cos (2 * x))) / 4) * 
               Real.tan (2 * x) * 
               Real.log (Real.cos (2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_l3774_377468


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3774_377474

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2100 - 1) (2^2000 - 1) = 2^100 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3774_377474


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3774_377496

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) : 
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3774_377496


namespace NUMINAMATH_CALUDE_smallest_m_is_16_l3774_377405

/-- The set T of complex numbers -/
def T : Set ℂ :=
  {z : ℂ | ∃ (u v : ℝ), z = u + v * Complex.I ∧ Real.sqrt 3 / 3 ≤ u ∧ u ≤ Real.sqrt 3 / 2}

/-- The property P(n) that should hold for all n ≥ m -/
def P (n : ℕ) : Prop :=
  ∃ z ∈ T, z ^ (2 * n) = 1

/-- The theorem stating that 16 is the smallest positive integer m satisfying the condition -/
theorem smallest_m_is_16 :
  (∀ n ≥ 16, P n) ∧ ∀ m < 16, ¬(∀ n ≥ m, P n) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_16_l3774_377405


namespace NUMINAMATH_CALUDE_sin_even_function_phi_l3774_377483

theorem sin_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_even_function_phi_l3774_377483


namespace NUMINAMATH_CALUDE_problem_solution_l3774_377461

theorem problem_solution : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3774_377461


namespace NUMINAMATH_CALUDE_cost_price_is_1000_l3774_377418

/-- The cost price of a toy, given the selling conditions -/
def cost_price_of_toy (total_sold : ℕ) (selling_price : ℕ) (gain_in_toys : ℕ) : ℕ :=
  selling_price / (total_sold + gain_in_toys)

/-- Theorem stating the cost price of a toy under the given conditions -/
theorem cost_price_is_1000 :
  cost_price_of_toy 18 21000 3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_1000_l3774_377418


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l3774_377482

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d :=
by sorry

theorem fraction_multiplication (a b c : ℚ) (n : ℕ) :
  (a * b * c : ℚ) * n = (n : ℚ) / ((1 / a) * (1 / b) * (1 / c)) :=
by sorry

theorem main_theorem : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 7 : ℚ) * 126 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_fraction_multiplication_main_theorem_l3774_377482


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3774_377480

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  let c := Real.sqrt (a^2 + b^2)
  let area := (1/2) * a * b
  area = 35 := by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3774_377480


namespace NUMINAMATH_CALUDE_polynomial_equality_l3774_377459

theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) : 
  (4 * x^5 + 3 * x^3 - 2 * x + 5 + g x = 7 * x^3 - 4 * x^2 + x + 2) → 
  (g x = -4 * x^5 + 4 * x^3 - 4 * x^2 + 3 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3774_377459


namespace NUMINAMATH_CALUDE_georges_walk_l3774_377478

/-- Given that George walks 1 mile to school at 3 mph normally, prove that
    if he walks the first 1/2 mile at 2 mph, he must run the last 1/2 mile
    at 6 mph to arrive at the same time. -/
theorem georges_walk (normal_distance : Real) (normal_speed : Real) 
  (first_half_distance : Real) (first_half_speed : Real) 
  (second_half_distance : Real) (second_half_speed : Real) :
  normal_distance = 1 ∧ 
  normal_speed = 3 ∧ 
  first_half_distance = 1/2 ∧ 
  first_half_speed = 2 ∧ 
  second_half_distance = 1/2 ∧
  normal_distance / normal_speed = 
    first_half_distance / first_half_speed + second_half_distance / second_half_speed →
  second_half_speed = 6 := by
  sorry

#check georges_walk

end NUMINAMATH_CALUDE_georges_walk_l3774_377478


namespace NUMINAMATH_CALUDE_nuts_problem_l3774_377452

theorem nuts_problem (x y : ℕ) : 
  (70 ≤ x + y ∧ x + y ≤ 80) ∧ 
  (3 * x + 5 * y + x = 20 * x + 20) →
  x = 36 ∧ y = 41 :=
sorry

end NUMINAMATH_CALUDE_nuts_problem_l3774_377452


namespace NUMINAMATH_CALUDE_cars_lifted_is_six_l3774_377410

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars being lifted -/
def cars_lifted : ℕ := 6

/-- The number of trucks being lifted -/
def trucks_lifted : ℕ := 3

/-- Theorem stating that the number of cars being lifted is 6 -/
theorem cars_lifted_is_six : cars_lifted = 6 := by sorry

end NUMINAMATH_CALUDE_cars_lifted_is_six_l3774_377410


namespace NUMINAMATH_CALUDE_base_five_product_131_21_l3774_377435

/-- Represents a number in base 5 --/
def BaseFive : Type := List Nat

/-- Converts a base 5 number to its decimal representation --/
def to_decimal (n : BaseFive) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base 5 representation --/
def to_base_five (n : Nat) : BaseFive :=
  sorry

/-- Multiplies two base 5 numbers --/
def base_five_mul (a b : BaseFive) : BaseFive :=
  to_base_five (to_decimal a * to_decimal b)

theorem base_five_product_131_21 :
  base_five_mul [1, 3, 1] [1, 2] = [1, 5, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_base_five_product_131_21_l3774_377435


namespace NUMINAMATH_CALUDE_power_function_decreasing_condition_l3774_377438

/-- A function f is a power function if it's of the form f(x) = x^a for some real a -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞), x < y implies f(x) > f(y) -/
def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_condition (m : ℝ) : 
  (is_power_function (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1)) ∧ 
   is_decreasing_on_positive_reals (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1))) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_condition_l3774_377438


namespace NUMINAMATH_CALUDE_constant_value_l3774_377469

theorem constant_value (t : ℝ) (C : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t + C
  (x = y → t = 0.5) → C = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l3774_377469


namespace NUMINAMATH_CALUDE_probability_even_sum_l3774_377427

def set_A : Finset ℕ := {3, 4, 5, 8}
def set_B : Finset ℕ := {6, 7, 9}

def is_sum_even (a b : ℕ) : Bool :=
  (a + b) % 2 = 0

def count_even_sums : ℕ :=
  (set_A.card * set_B.card).div 2

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_A.card * set_B.card) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l3774_377427


namespace NUMINAMATH_CALUDE_odd_function_property_l3774_377446

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3774_377446


namespace NUMINAMATH_CALUDE_total_legs_is_108_l3774_377429

-- Define the number of each animal
def num_birds : ℕ := 3
def num_dogs : ℕ := 5
def num_snakes : ℕ := 4
def num_spiders : ℕ := 1
def num_horses : ℕ := 2
def num_rabbits : ℕ := 6
def num_octopuses : ℕ := 3
def num_ants : ℕ := 7

-- Define the number of legs for each animal type
def legs_bird : ℕ := 2
def legs_dog : ℕ := 4
def legs_snake : ℕ := 0
def legs_spider : ℕ := 8
def legs_horse : ℕ := 4
def legs_rabbit : ℕ := 4
def legs_octopus : ℕ := 0
def legs_ant : ℕ := 6

-- Theorem to prove
theorem total_legs_is_108 : 
  num_birds * legs_bird + 
  num_dogs * legs_dog + 
  num_snakes * legs_snake + 
  num_spiders * legs_spider + 
  num_horses * legs_horse + 
  num_rabbits * legs_rabbit + 
  num_octopuses * legs_octopus + 
  num_ants * legs_ant = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_108_l3774_377429


namespace NUMINAMATH_CALUDE_father_twice_son_age_l3774_377436

/-- Represents the age difference between father and son when the father's age becomes more than twice the son's age -/
def AgeDifference : ℕ → Prop :=
  λ x => ∃ (y : ℕ), (27 + x = 2 * (((27 - 3) / 3) + x) + y) ∧ y > 0

/-- Theorem stating that it takes 11 years for the father's age to be more than twice the son's age -/
theorem father_twice_son_age : AgeDifference 11 := by
  sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l3774_377436


namespace NUMINAMATH_CALUDE_smallest_difference_l3774_377498

def Digits : Finset ℕ := {1, 3, 5, 7, 8}

def IsValidArrangement (a b : ℕ) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  (a / 100 = 1 ∨ a / 100 = 8) ∧
  Finset.card (Finset.filter (λ d => d ∈ Digits) {a / 100, (a / 10) % 10, a % 10, b / 10, b % 10}) = 5

def Difference (a b : ℕ) : ℕ := a - b

theorem smallest_difference :
  ∃ (a b : ℕ), IsValidArrangement a b ∧
    Difference a b = 48 ∧
    ∀ (x y : ℕ), IsValidArrangement x y → Difference x y ≥ 48 :=
  sorry

end NUMINAMATH_CALUDE_smallest_difference_l3774_377498


namespace NUMINAMATH_CALUDE_brady_earnings_correct_l3774_377490

/-- Calculates the total earnings for Brady's transcription work -/
def brady_earnings (basic_cards : ℕ) (gourmet_cards : ℕ) : ℚ :=
  let basic_rate : ℚ := 70 / 100
  let gourmet_rate : ℚ := 90 / 100
  let basic_earnings := basic_rate * basic_cards
  let gourmet_earnings := gourmet_rate * gourmet_cards
  let card_earnings := basic_earnings + gourmet_earnings
  let total_cards := basic_cards + gourmet_cards
  let bonus_count := total_cards / 100
  let bonus_base := 10
  let bonus_increment := 5
  let bonus_total := bonus_count * bonus_base + (bonus_count * (bonus_count - 1) / 2) * bonus_increment
  card_earnings + bonus_total

theorem brady_earnings_correct :
  brady_earnings 120 80 = 181 := by
  sorry

end NUMINAMATH_CALUDE_brady_earnings_correct_l3774_377490


namespace NUMINAMATH_CALUDE_section_B_seats_l3774_377455

-- Define the number of seats in the different subsections of Section A
def seats_subsection_1 : ℕ := 60
def seats_subsection_2 : ℕ := 80
def num_subsection_2 : ℕ := 3

-- Define the total number of seats in Section A
def total_seats_A : ℕ := seats_subsection_1 + seats_subsection_2 * num_subsection_2

-- Define the number of seats in Section B
def seats_B : ℕ := 3 * total_seats_A + 20

-- Theorem statement
theorem section_B_seats : seats_B = 920 := by sorry

end NUMINAMATH_CALUDE_section_B_seats_l3774_377455


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3774_377476

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line given its equation
def slope_of_line (f : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- Theorem: The slope of a line parallel to 3x - 6y = 12 is 1/2
theorem parallel_line_slope :
  slope_of_line line_equation = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3774_377476


namespace NUMINAMATH_CALUDE_last_digit_of_3_to_2010_l3774_377431

theorem last_digit_of_3_to_2010 (h : ∀ n : ℕ, 
  (3^n % 10) = (3^(n % 4) % 10)) : 
  3^2010 % 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_last_digit_of_3_to_2010_l3774_377431


namespace NUMINAMATH_CALUDE_homework_time_theorem_l3774_377445

/-- The total time left for homework completion --/
def total_time (jacob_time greg_time patrick_time : ℕ) : ℕ :=
  jacob_time + greg_time + patrick_time

/-- Theorem stating the total time left for homework completion --/
theorem homework_time_theorem (jacob_time greg_time patrick_time : ℕ) 
  (h1 : jacob_time = 18)
  (h2 : greg_time = jacob_time - 6)
  (h3 : patrick_time = 2 * greg_time - 4) :
  total_time jacob_time greg_time patrick_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_theorem_l3774_377445


namespace NUMINAMATH_CALUDE_log_75843_bounds_l3774_377434

theorem log_75843_bounds : ∃ (c d : ℤ), (c : ℝ) < Real.log 75843 / Real.log 10 ∧ 
  Real.log 75843 / Real.log 10 < (d : ℝ) ∧ c = 4 ∧ d = 5 ∧ c + d = 9 := by
  sorry

#check log_75843_bounds

end NUMINAMATH_CALUDE_log_75843_bounds_l3774_377434


namespace NUMINAMATH_CALUDE_exists_identical_triangles_l3774_377460

-- Define a triangle type
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (hypotenuse : ℝ)

-- Define a function to represent a cut operation
def cut (t : Triangle) : (Triangle × Triangle) := sorry

-- Define a function to check if two triangles are identical
def are_identical (t1 t2 : Triangle) : Prop := sorry

-- Define the initial set of triangles
def initial_triangles : Finset Triangle := sorry

-- Define the set of triangles after n cuts
def triangles_after_cuts (n : ℕ) : Finset Triangle := sorry

-- The main theorem
theorem exists_identical_triangles (n : ℕ) :
  ∃ t1 t2 : Triangle, t1 ∈ triangles_after_cuts n ∧ t2 ∈ triangles_after_cuts n ∧ t1 ≠ t2 ∧ are_identical t1 t2 :=
sorry

end NUMINAMATH_CALUDE_exists_identical_triangles_l3774_377460


namespace NUMINAMATH_CALUDE_cos_5pi_4_plus_x_l3774_377487

theorem cos_5pi_4_plus_x (x : ℝ) (h : Real.sin (π/4 - x) = -1/5) : 
  Real.cos (5*π/4 + x) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_5pi_4_plus_x_l3774_377487


namespace NUMINAMATH_CALUDE_solve_for_m_l3774_377406

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (m : ℝ) : Prop :=
  (1 - m * i) / (i^3) = 1 + i

-- Theorem statement
theorem solve_for_m :
  ∃ (m : ℝ), equation m ∧ m = 1 :=
sorry

end NUMINAMATH_CALUDE_solve_for_m_l3774_377406


namespace NUMINAMATH_CALUDE_max_k_value_l3774_377413

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ 1 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = 1^2 * ((x^2 / y^2) + (y^2 / x^2)) + 1 * ((x / y) + (y / x)) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3774_377413


namespace NUMINAMATH_CALUDE_number_975_in_column_B_l3774_377481

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- Determines if a given row number is odd --/
def isOddRow (n : ℕ) : Bool :=
  n % 2 = 1

/-- Calculates the column for a given number in the arrangement --/
def columnForNumber (n : ℕ) : Column :=
  let adjustedN := n - 1
  let rowNumber := (adjustedN / 6) + 1
  let positionInRow := adjustedN % 6
  if isOddRow rowNumber then
    match positionInRow with
    | 0 => Column.A
    | 1 => Column.B
    | 2 => Column.C
    | 3 => Column.D
    | 4 => Column.E
    | _ => Column.F
  else
    match positionInRow with
    | 0 => Column.F
    | 1 => Column.E
    | 2 => Column.D
    | 3 => Column.C
    | 4 => Column.B
    | _ => Column.A

/-- Theorem: The integer 975 is in column B in the given arrangement --/
theorem number_975_in_column_B : columnForNumber 975 = Column.B := by
  sorry

end NUMINAMATH_CALUDE_number_975_in_column_B_l3774_377481


namespace NUMINAMATH_CALUDE_fence_painting_fraction_l3774_377465

theorem fence_painting_fraction (total_time minutes : ℚ) (fraction : ℚ) :
  total_time = 60 →
  minutes = 15 →
  fraction = minutes / total_time →
  fraction = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fence_painting_fraction_l3774_377465


namespace NUMINAMATH_CALUDE_cousins_distribution_l3774_377420

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of rooms available -/
def num_rooms : ℕ := 4

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The theorem stating that there are 76 ways to distribute the cousins into the rooms -/
theorem cousins_distribution : distribute num_cousins num_rooms = 76 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l3774_377420


namespace NUMINAMATH_CALUDE_consecutive_integers_product_2720_sum_103_l3774_377412

theorem consecutive_integers_product_2720_sum_103 :
  ∀ n : ℕ, n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_2720_sum_103_l3774_377412


namespace NUMINAMATH_CALUDE_divisible_by_21_with_sqrt_between_30_and_30_5_l3774_377407

theorem divisible_by_21_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (21 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_21_with_sqrt_between_30_and_30_5_l3774_377407


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimizing_x_value_l3774_377485

/-- The quadratic function f(x) = x^2 - 10x + 24 attains its minimum value when x = 5. -/
theorem quadratic_minimum : ∀ x : ℝ, (x^2 - 10*x + 24) ≥ (5^2 - 10*5 + 24) := by
  sorry

/-- The value of x that minimizes the quadratic function f(x) = x^2 - 10x + 24 is 5. -/
theorem minimizing_x_value : ∃! x : ℝ, ∀ y : ℝ, (x^2 - 10*x + 24) ≤ (y^2 - 10*y + 24) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimizing_x_value_l3774_377485


namespace NUMINAMATH_CALUDE_vector_sum_proof_l3774_377426

/-- Given two 2D vectors a and b, prove that 3a + 4b equals the specified result -/
theorem vector_sum_proof (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-3, 4) → (3 • a + 4 • b : ℝ × ℝ) = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l3774_377426


namespace NUMINAMATH_CALUDE_three_numbers_sum_divisible_by_three_l3774_377404

def set_of_numbers : Finset ℕ := Finset.range 20

theorem three_numbers_sum_divisible_by_three (set_of_numbers : Finset ℕ) :
  (Finset.filter (fun s : Finset ℕ => s.card = 3 ∧ 
    (s.sum id) % 3 = 0 ∧ 
    s ⊆ set_of_numbers) (Finset.powerset set_of_numbers)).card = 384 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_divisible_by_three_l3774_377404


namespace NUMINAMATH_CALUDE_car_dealership_monthly_payment_l3774_377453

/-- Calculates the total monthly payment for employees in a car dealership --/
theorem car_dealership_monthly_payment 
  (fiona_hours : ℕ) 
  (john_hours : ℕ) 
  (jeremy_hours : ℕ) 
  (hourly_rate : ℕ) 
  (h1 : fiona_hours = 40)
  (h2 : john_hours = 30)
  (h3 : jeremy_hours = 25)
  (h4 : hourly_rate = 20)
  : (fiona_hours + john_hours + jeremy_hours) * hourly_rate * 4 = 7600 := by
  sorry


end NUMINAMATH_CALUDE_car_dealership_monthly_payment_l3774_377453


namespace NUMINAMATH_CALUDE_largest_whole_number_times_eight_less_than_150_l3774_377464

theorem largest_whole_number_times_eight_less_than_150 :
  ∃ y : ℕ, y = 18 ∧ 8 * y < 150 ∧ ∀ z : ℕ, z > y → 8 * z ≥ 150 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_times_eight_less_than_150_l3774_377464


namespace NUMINAMATH_CALUDE_maximize_product_l3774_377489

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^6 * y^3 ≤ (100/3)^6 * (50/3)^3 ∧
  x^6 * y^3 = (100/3)^6 * (50/3)^3 ↔ x = 100/3 ∧ y = 50/3 :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l3774_377489


namespace NUMINAMATH_CALUDE_half_of_eighteen_is_nine_l3774_377442

theorem half_of_eighteen_is_nine : (18 : ℝ) / 2 = 9 := by sorry

end NUMINAMATH_CALUDE_half_of_eighteen_is_nine_l3774_377442


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3774_377495

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 4016.25 →
  rate = 9 / 100 →
  time = 5 →
  principal = 8925 →
  interest = principal * rate * time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3774_377495


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3774_377419

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- A line with a given slope -/
structure Line where
  slope : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.1^2 / (a^2) + p.2^2 / (b^2) = 1

/-- The equation of a line -/
def line_equation (m b : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => p.2 = m * p.1 + b

theorem ellipse_intersection_theorem (C : Ellipse) (l : Line) :
  C.center = (0, 0) ∧ C.left_focus = (-Real.sqrt 3, 0) ∧ C.right_vertex = (2, 0) ∧ l.slope = 1/2 →
  (∃ a b : ℝ, standard_equation a b = standard_equation 2 1) ∧
  (∃ chord_length : ℝ, chord_length ≤ Real.sqrt 10 ∧
    ∀ other_length : ℝ, other_length ≤ chord_length) ∧
  (∃ b : ℝ, line_equation (1/2) b = line_equation (1/2) 0 →
    ∀ other_b : ℝ, ∃ length : ℝ, length ≤ Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3774_377419


namespace NUMINAMATH_CALUDE_ngon_area_division_l3774_377457

/-- Represents a convex n-gon -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Condition for convexity

/-- Represents a point on the boundary of the n-gon -/
structure BoundaryPoint (n : ℕ) (polygon : ConvexNGon n) where
  point : ℝ × ℝ
  on_boundary : sorry -- Condition for being on the boundary
  not_vertex : sorry -- Condition for not being a vertex

/-- Predicate to check if a line divides the polygon's area in half -/
def divides_area_in_half (n : ℕ) (polygon : ConvexNGon n) (a b : ℝ × ℝ) : Prop := sorry

/-- The number of sides on which the boundary points lie -/
def sides_with_points (n : ℕ) (polygon : ConvexNGon n) (points : Fin n → BoundaryPoint n polygon) : ℕ := sorry

theorem ngon_area_division (n : ℕ) (polygon : ConvexNGon n) 
  (points : Fin n → BoundaryPoint n polygon)
  (h_divide : ∀ i : Fin n, divides_area_in_half n polygon (polygon.vertices i) (points i).point) :
  (3 ≤ sides_with_points n polygon points) ∧ 
  (sides_with_points n polygon points ≤ if n % 2 = 0 then n - 1 else n) := sorry

end NUMINAMATH_CALUDE_ngon_area_division_l3774_377457


namespace NUMINAMATH_CALUDE_x_squared_over_x_fourth_plus_x_squared_plus_one_l3774_377414

theorem x_squared_over_x_fourth_plus_x_squared_plus_one (x : ℝ) 
  (h1 : x^2 - 3*x - 1 = 0) (h2 : x ≠ 0) : x^2 / (x^4 + x^2 + 1) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_over_x_fourth_plus_x_squared_plus_one_l3774_377414


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l3774_377448

/-- The value of p for which the focus of the parabola y^2 = -2px coincides with the left focus of the ellipse (x^2/16) + (y^2/12) = 1 -/
theorem parabola_ellipse_focus_coincide : ∃ p : ℝ,
  (∀ x y : ℝ, y^2 = -2*p*x → (x^2/16 + y^2/12 = 1 → x = -2)) →
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincide_l3774_377448


namespace NUMINAMATH_CALUDE_gasoline_spending_increase_l3774_377475

theorem gasoline_spending_increase (P Q : ℝ) (P_increase : ℝ) (Q_decrease : ℝ) :
  P > 0 ∧ Q > 0 ∧ P_increase = 0.25 ∧ Q_decrease = 0.16 →
  (1 + 0.05) * (P * Q) = (P * (1 + P_increase)) * (Q * (1 - Q_decrease)) :=
by sorry

end NUMINAMATH_CALUDE_gasoline_spending_increase_l3774_377475


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l3774_377450

/-- The force required to loosen a nut with a wrench -/
def force_to_loosen (handle_length : ℝ) (force : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * handle_length = k

theorem wrench_force_calculation 
  (h₁ : force_to_loosen 12 480) 
  (h₂ : force_to_loosen 18 f) : 
  f = 320 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_calculation_l3774_377450


namespace NUMINAMATH_CALUDE_tan_four_thirds_pi_l3774_377484

theorem tan_four_thirds_pi : Real.tan (4 * π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_thirds_pi_l3774_377484


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3774_377447

theorem trigonometric_identities :
  let a := Real.sqrt 2 / 2 * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180))
  let b := Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2
  let c := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let d := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  (a = 1/2) ∧ 
  (c = 1/2) ∧ 
  (b ≠ 1/2) ∧ 
  (d ≠ 1/2) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3774_377447


namespace NUMINAMATH_CALUDE_cross_in_square_l3774_377444

/-- Given a square with side length S containing a cross made of two large squares
    (each with side length S/2) and two small squares (each with side length S/4),
    if the total area of the cross is 810 cm², then S = 36 cm. -/
theorem cross_in_square (S : ℝ) : 
  (2 * (S/2)^2 + 2 * (S/4)^2 = 810) → S = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l3774_377444


namespace NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3774_377440

-- Define the total number of workers
def total_workers : ℕ := 14

-- Define the number of workers to be chosen
def chosen_workers : ℕ := 2

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem probability_james_and_david_chosen :
  (1 : ℚ) / (combination total_workers chosen_workers) = 1 / 91 :=
sorry

end NUMINAMATH_CALUDE_probability_james_and_david_chosen_l3774_377440


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l3774_377486

-- Definition of rational numbers
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Definition of irrational numbers
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- Theorem stating that √2 is irrational
theorem sqrt_two_irrational : IsIrrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l3774_377486


namespace NUMINAMATH_CALUDE_unique_triple_l3774_377470

/-- A function that checks if a number is divisible by any prime less than 2014 -/
def not_divisible_by_small_primes (n : ℕ) : Prop :=
  ∀ p, p < 2014 → Nat.Prime p → ¬(p ∣ n)

/-- The main theorem statement -/
theorem unique_triple : 
  ∃! (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ n : ℕ, n > 0 → not_divisible_by_small_primes n → 
      (n + c) ∣ (a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_triple_l3774_377470


namespace NUMINAMATH_CALUDE_smallest_absolute_value_at_0_l3774_377488

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- The property that a polynomial P satisfies P(-10) = 145 and P(9) = 164 -/
def SatisfiesConditions (P : IntPolynomial) : Prop :=
  P (-10) = 145 ∧ P 9 = 164

/-- The smallest possible absolute value of P(0) for polynomials satisfying the conditions -/
def SmallestAbsoluteValueAt0 : ℕ := 25

theorem smallest_absolute_value_at_0 :
  ∀ P : IntPolynomial,
  SatisfiesConditions P →
  ∀ n : ℕ,
  n < SmallestAbsoluteValueAt0 →
  ¬(|P 0| = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_at_0_l3774_377488
