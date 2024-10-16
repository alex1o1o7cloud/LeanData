import Mathlib

namespace NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_b_range_for_decreasing_l2680_268033

/-- A quadratic function f(x) = ax^2 + bx -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem decreasing_interval_of_quadratic (a b : ℝ) :
  (f_derivative a b 3 = 24) →  -- Tangent at x=3 is parallel to 24x-y+1=0
  (f_derivative a b 1 = 0) →   -- Extreme value at x=1
  ∀ x > 1, f_derivative a b x < 0 := by sorry

theorem b_range_for_decreasing (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f_derivative 1 b x ≤ 0) →
  b ≤ -2 := by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_b_range_for_decreasing_l2680_268033


namespace NUMINAMATH_CALUDE_abc_right_triangle_l2680_268035

/-- Parabola defined by y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- Point A -/
def A : ℝ × ℝ := (1, 2)

/-- Point P -/
def P : ℝ × ℝ := (5, -2)

/-- B and C are on the parabola -/
def on_parabola (B C : ℝ × ℝ) : Prop := parabola B ∧ parabola C

/-- Line BC passes through P -/
def line_through_P (B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B.1 + t * (C.1 - B.1) = P.1 ∧ B.2 + t * (C.2 - B.2) = P.2

/-- Triangle ABC is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem abc_right_triangle (B C : ℝ × ℝ) :
  on_parabola B C → line_through_P B C → is_right_triangle A B C := by sorry

end NUMINAMATH_CALUDE_abc_right_triangle_l2680_268035


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2680_268095

theorem complex_number_in_third_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := -5 * i / (2 + 3 * i)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2680_268095


namespace NUMINAMATH_CALUDE_lily_milk_problem_l2680_268020

theorem lily_milk_problem (initial_milk : ℚ) (milk_given : ℚ) (milk_left : ℚ) : 
  initial_milk = 5 → milk_given = 18/7 → milk_left = initial_milk - milk_given → milk_left = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_lily_milk_problem_l2680_268020


namespace NUMINAMATH_CALUDE_titan_high_school_contest_l2680_268091

theorem titan_high_school_contest (f s : ℕ) 
  (h1 : f > 0) (h2 : s > 0)
  (h3 : (f / 3 : ℚ) = (s / 2 : ℚ)) : s = 3 * f := by
  sorry

end NUMINAMATH_CALUDE_titan_high_school_contest_l2680_268091


namespace NUMINAMATH_CALUDE_phillips_remaining_money_l2680_268086

/-- Calculates the remaining money after purchases --/
def remaining_money (initial : ℕ) (orange_cost apple_cost candy_cost : ℕ) : ℕ :=
  initial - (orange_cost + apple_cost + candy_cost)

/-- Theorem stating that given the specific amounts, the remaining money is $50 --/
theorem phillips_remaining_money :
  remaining_money 95 14 25 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_phillips_remaining_money_l2680_268086


namespace NUMINAMATH_CALUDE_digit_cube_equals_square_l2680_268031

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem digit_cube_equals_square (n : Nat) : 
  n ∈ Finset.range 1000 → (n^2 = (sum_of_digits n)^3 ↔ n = 1 ∨ n = 27) := by
  sorry

end NUMINAMATH_CALUDE_digit_cube_equals_square_l2680_268031


namespace NUMINAMATH_CALUDE_inverse_of_M_l2680_268004

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℝ := B * A

theorem inverse_of_M : 
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] :=
sorry

end NUMINAMATH_CALUDE_inverse_of_M_l2680_268004


namespace NUMINAMATH_CALUDE_prob_product_odd_eight_rolls_l2680_268023

-- Define a standard die
def StandardDie : Type := Fin 6

-- Define the property of being an odd number
def isOdd (n : Nat) : Prop := n % 2 = 1

-- Define the probability of rolling an odd number on a standard die
def probOddRoll : ℚ := 1 / 2

-- Define the number of rolls
def numRolls : Nat := 8

-- Theorem statement
theorem prob_product_odd_eight_rolls :
  (probOddRoll ^ numRolls : ℚ) = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_odd_eight_rolls_l2680_268023


namespace NUMINAMATH_CALUDE_count_counterexamples_l2680_268094

def sum_of_digits (n : ℕ) : ℕ := sorry

def has_no_zero_digit (n : ℕ) : Prop := sorry

def counterexample (n : ℕ) : Prop :=
  sum_of_digits n = 5 ∧ has_no_zero_digit n ∧ ¬ Nat.Prime n

theorem count_counterexamples : 
  ∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ counterexample n :=
sorry

end NUMINAMATH_CALUDE_count_counterexamples_l2680_268094


namespace NUMINAMATH_CALUDE_constant_term_proof_l2680_268016

/-- Given an equation (ax + w)(cx + d) = 6x^2 + x - 12, where a, w, c, and d are real numbers
    whose absolute values sum to 12, prove that the constant term in the expanded form is -12. -/
theorem constant_term_proof (a w c d : ℝ) 
    (eq : ∀ x, (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
    (sum_abs : |a| + |w| + |c| + |d| = 12) :
    w * d = -12 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_proof_l2680_268016


namespace NUMINAMATH_CALUDE_limit_f_at_infinity_l2680_268044

noncomputable def f (x : ℝ) := (x - Real.sin x) / (x + Real.sin x)

theorem limit_f_at_infinity :
  ∀ ε > 0, ∃ N : ℝ, ∀ x ≥ N, |f x - 1| < ε :=
by
  sorry

/- Assumptions:
   1. x is a real number (implied by the use of ℝ)
   2. sin x is bounded between -1 and 1 (this is a property of sine in Mathlib)
-/

end NUMINAMATH_CALUDE_limit_f_at_infinity_l2680_268044


namespace NUMINAMATH_CALUDE_third_number_proof_l2680_268053

theorem third_number_proof (sum : ℝ) (a b c : ℝ) (h : sum = a + b + c + 0.217) :
  sum - a - b - c = 0.217 :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l2680_268053


namespace NUMINAMATH_CALUDE_unique_perfect_cube_divisibility_l2680_268071

theorem unique_perfect_cube_divisibility : ∃! X : ℕ+, 
  (∃ Y : ℕ+, X = Y^3) ∧ 
  X = (555 * 465)^2 * (555 - 465)^3 + (555 - 465)^4 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_cube_divisibility_l2680_268071


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2680_268036

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2680_268036


namespace NUMINAMATH_CALUDE_intersection_theorem_l2680_268010

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 2) ≤ 0}

-- Define set B
def B : Set ℝ := {x | |x - 1| < 2}

-- Define the complement of A with respect to ℝ
def not_A : Set ℝ := {x | x ∉ A}

-- Define the intersection of B and the complement of A
def B_intersect_not_A : Set ℝ := B ∩ not_A

-- Theorem statement
theorem intersection_theorem : B_intersect_not_A = {x | 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2680_268010


namespace NUMINAMATH_CALUDE_initial_pens_l2680_268017

theorem initial_pens (initial : ℕ) (mike_gives : ℕ) (cindy_doubles : ℕ → ℕ) (sharon_takes : ℕ) (final : ℕ) : 
  mike_gives = 22 →
  cindy_doubles = (· * 2) →
  sharon_takes = 19 →
  final = 39 →
  cindy_doubles (initial + mike_gives) - sharon_takes = final →
  initial = 7 := by
sorry

end NUMINAMATH_CALUDE_initial_pens_l2680_268017


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2680_268088

theorem geometric_sequence_sum (a₁ r : ℝ) (n : ℕ) : 
  a₁ = 4 → r = 2 → n = 4 → 
  (a₁ * (1 - r^n)) / (1 - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2680_268088


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2680_268027

-- Define the sets A, B, and E
def A : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def B : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

-- Theorem for the first part of the problem
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
sorry

-- Theorem for the second part of the problem
theorem E_subset_B_implies_a_geq_neg_one (a : ℝ) :
  E a ⊆ B → a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_E_subset_B_implies_a_geq_neg_one_l2680_268027


namespace NUMINAMATH_CALUDE_lowest_score_problem_l2680_268042

theorem lowest_score_problem (scores : Finset ℕ) (highest lowest : ℕ) :
  Finset.card scores = 15 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 100 →
  (Finset.sum scores id) / 15 = 85 →
  ((Finset.sum scores id) - highest - lowest) / 13 = 86 →
  lowest = 57 := by
  sorry

end NUMINAMATH_CALUDE_lowest_score_problem_l2680_268042


namespace NUMINAMATH_CALUDE_range_of_k_is_real_l2680_268002

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an increasing function
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_k_is_real (h : IsIncreasing f) : 
  ∀ k : ℝ, ∃ x : ℝ, f x = k :=
sorry

end NUMINAMATH_CALUDE_range_of_k_is_real_l2680_268002


namespace NUMINAMATH_CALUDE_max_safe_caffeine_value_l2680_268047

/-- The maximum safe amount of caffeine one can consume per day -/
def max_safe_caffeine : ℕ := sorry

/-- The amount of caffeine in one energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumes -/
def drinks_consumed : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume (in mg) -/
def additional_safe_caffeine : ℕ := 20

/-- Theorem stating the maximum safe amount of caffeine one can consume per day -/
theorem max_safe_caffeine_value : 
  max_safe_caffeine = caffeine_per_drink * drinks_consumed + additional_safe_caffeine := by
  sorry

end NUMINAMATH_CALUDE_max_safe_caffeine_value_l2680_268047


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2680_268019

/-- Fixed circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- Fixed line L -/
def L (x : ℝ) : Prop := x = 1

/-- Moving circle P -/
def P (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- P is externally tangent to C -/
def externally_tangent (x y r : ℝ) : Prop :=
  (x + 2 - 1)^2 + y^2 = (r + 1)^2

/-- P is tangent to L -/
def tangent_to_L (x y r : ℝ) : Prop := x - r = 1

/-- Trajectory of the center of P -/
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

theorem moving_circle_trajectory :
  ∀ x y r : ℝ,
  C x y ∧ L 1 ∧ P x y r ∧ externally_tangent x y r ∧ tangent_to_L x y r →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2680_268019


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2680_268093

/-- A hyperbola with foci on the x-axis and asymptotic lines y = ±√3x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (b / a = Real.sqrt 3) → 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2680_268093


namespace NUMINAMATH_CALUDE_sphere_volume_for_maximized_tetrahedron_l2680_268061

theorem sphere_volume_for_maximized_tetrahedron (r : ℝ) (h : r = (3 * Real.sqrt 3) / 2) :
  (4 / 3) * Real.pi * r^3 = (27 * Real.sqrt 3 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_for_maximized_tetrahedron_l2680_268061


namespace NUMINAMATH_CALUDE_max_value_of_z_l2680_268056

theorem max_value_of_z (x y k : ℝ) (h1 : x + 2*y - 1 ≥ 0) (h2 : x - y ≥ 0) 
  (h3 : 0 ≤ x) (h4 : x ≤ k) (h5 : ∃ (x_min y_min : ℝ), x_min + k*y_min = -2 ∧ 
  x_min + 2*y_min - 1 ≥ 0 ∧ x_min - y_min ≥ 0 ∧ 0 ≤ x_min ∧ x_min ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≥ -2) :
  ∃ (x_max y_max : ℝ), x_max + k*y_max = 20 ∧ 
  x_max + 2*y_max - 1 ≥ 0 ∧ x_max - y_max ≥ 0 ∧ 0 ≤ x_max ∧ x_max ≤ k ∧
  ∀ (x' y' : ℝ), x' + 2*y' - 1 ≥ 0 → x' - y' ≥ 0 → 0 ≤ x' → x' ≤ k → x' + k*y' ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2680_268056


namespace NUMINAMATH_CALUDE_find_divisor_l2680_268032

theorem find_divisor (N : ℝ) (D : ℝ) (h1 : N = 95) (h2 : N / D + 23 = 42) : D = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2680_268032


namespace NUMINAMATH_CALUDE_book_completion_time_l2680_268026

/-- Calculates the number of weeks needed to complete a book given the writing schedule and book length -/
theorem book_completion_time (writing_hours_per_day : ℕ) (pages_per_hour : ℕ) (total_pages : ℕ) :
  writing_hours_per_day = 3 →
  pages_per_hour = 5 →
  total_pages = 735 →
  (total_pages / (writing_hours_per_day * pages_per_hour) + 6) / 7 = 7 :=
by
  sorry

#check book_completion_time

end NUMINAMATH_CALUDE_book_completion_time_l2680_268026


namespace NUMINAMATH_CALUDE_min_ab_in_triangle_l2680_268030

theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1/2) * a * b * Real.sin C = (Real.sqrt 3 / 2) * c →
  a * b ≥ 12 := by
sorry

end NUMINAMATH_CALUDE_min_ab_in_triangle_l2680_268030


namespace NUMINAMATH_CALUDE_initial_average_height_l2680_268082

theorem initial_average_height (n : ℕ) (wrong_height correct_height actual_average : ℝ) 
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 179) :
  (n * actual_average + (wrong_height - correct_height)) / n = 181 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_height_l2680_268082


namespace NUMINAMATH_CALUDE_squares_in_50th_ring_l2680_268049

/-- The number of squares in the nth ring of a square pattern -/
def squares_in_ring (n : ℕ) : ℕ := 4 * n + 4

/-- The number of squares in the 50th ring is 204 -/
theorem squares_in_50th_ring : squares_in_ring 50 = 204 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_50th_ring_l2680_268049


namespace NUMINAMATH_CALUDE_scalper_ticket_percentage_l2680_268066

theorem scalper_ticket_percentage :
  let normal_price : ℝ := 50
  let website_tickets : ℕ := 2
  let scalper_tickets : ℕ := 2
  let discounted_tickets : ℕ := 1
  let discounted_percentage : ℝ := 60
  let total_paid : ℝ := 360
  let scalper_discount : ℝ := 10

  ∃ P : ℝ,
    website_tickets * normal_price +
    scalper_tickets * (P / 100 * normal_price) - scalper_discount +
    discounted_tickets * (discounted_percentage / 100 * normal_price) = total_paid ∧
    P = 480 :=
by sorry

end NUMINAMATH_CALUDE_scalper_ticket_percentage_l2680_268066


namespace NUMINAMATH_CALUDE_combination_problem_l2680_268015

theorem combination_problem (n : ℕ) (h : Nat.choose n 13 = Nat.choose n 7) :
  Nat.choose n 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_combination_problem_l2680_268015


namespace NUMINAMATH_CALUDE_f_intersects_y_axis_at_zero_one_l2680_268034

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem f_intersects_y_axis_at_zero_one : f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_y_axis_at_zero_one_l2680_268034


namespace NUMINAMATH_CALUDE_afternoon_shells_l2680_268059

theorem afternoon_shells (morning_shells : ℕ) (total_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : total_shells = 616) : 
  total_shells - morning_shells = 324 := by
sorry

end NUMINAMATH_CALUDE_afternoon_shells_l2680_268059


namespace NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2680_268060

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r ≠ 0 ∧ ∀ n, b n = b₁ * r^(n - 1)

/-- The main theorem -/
theorem arithmetic_geometric_relation
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_relation : a 1 - (a 7)^2 + a 13 = 0)
  (h_equal : b 7 = a 7) :
  b 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_relation_l2680_268060


namespace NUMINAMATH_CALUDE_first_group_size_is_20_l2680_268069

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the water fountain built by the first group -/
def first_fountain_length : ℝ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 7

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the water fountain built by the second group -/
def second_fountain_length : ℝ := 42

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- The theorem stating that the first group size is 20 men -/
theorem first_group_size_is_20 :
  first_group_size = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_20_l2680_268069


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l2680_268085

/-- The number of ways to distribute n indistinguishable balls into 2 distinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ := n + 1

/-- Theorem: There are 7 ways to distribute 6 indistinguishable balls into 2 distinguishable boxes -/
theorem six_balls_two_boxes : distribute_balls 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l2680_268085


namespace NUMINAMATH_CALUDE_functional_equation_implies_constant_l2680_268080

theorem functional_equation_implies_constant (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_eq : ∀ x y : ℝ, f (x + 2*y) = 2 * f x * f y) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_constant_l2680_268080


namespace NUMINAMATH_CALUDE_jumping_probabilities_l2680_268083

/-- Probability of an athlete successfully jumping over a 2-meter high bar -/
structure Athlete where
  success_prob : ℝ
  success_prob_nonneg : 0 ≤ success_prob
  success_prob_le_one : success_prob ≤ 1

/-- The problem setup with two athletes A and B -/
def problem_setup (A B : Athlete) : Prop :=
  A.success_prob = 0.7 ∧ B.success_prob = 0.6

/-- The probability that A succeeds on the third attempt -/
def prob_A_third_attempt (A : Athlete) : ℝ :=
  (1 - A.success_prob) * (1 - A.success_prob) * A.success_prob

/-- The probability that at least one of A or B succeeds on the first attempt -/
def prob_at_least_one_first_attempt (A B : Athlete) : ℝ :=
  1 - (1 - A.success_prob) * (1 - B.success_prob)

/-- The probability that A succeeds exactly one more time than B in two attempts for each -/
def prob_A_one_more_than_B (A B : Athlete) : ℝ :=
  2 * A.success_prob * (1 - A.success_prob) * (1 - B.success_prob) * (1 - B.success_prob) +
  A.success_prob * A.success_prob * 2 * B.success_prob * (1 - B.success_prob)

theorem jumping_probabilities (A B : Athlete) 
  (h : problem_setup A B) : 
  prob_A_third_attempt A = 0.063 ∧
  prob_at_least_one_first_attempt A B = 0.88 ∧
  prob_A_one_more_than_B A B = 0.3024 := by
  sorry

end NUMINAMATH_CALUDE_jumping_probabilities_l2680_268083


namespace NUMINAMATH_CALUDE_dilation_determinant_l2680_268064

theorem dilation_determinant (D : Matrix (Fin 3) (Fin 3) ℝ) 
  (h1 : D = Matrix.diagonal (λ _ => (5 : ℝ))) 
  (h2 : ∀ (i j : Fin 3), i ≠ j → D i j = 0) : 
  Matrix.det D = 125 := by
  sorry

end NUMINAMATH_CALUDE_dilation_determinant_l2680_268064


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l2680_268081

theorem sum_of_quadratic_roots (x : ℝ) : 
  x^2 - 17*x + 54 = 0 → ∃ r s : ℝ, r + s = 17 ∧ r * s = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l2680_268081


namespace NUMINAMATH_CALUDE_painted_cube_problem_l2680_268014

theorem painted_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 4 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_problem_l2680_268014


namespace NUMINAMATH_CALUDE_max_wins_l2680_268001

/-- 
Given that the ratio of Chloe's wins to Max's wins is 8:3, and Chloe won 24 times,
prove that Max won 9 times.
-/
theorem max_wins (chloe_wins : ℕ) (max_wins : ℕ) 
  (h1 : chloe_wins = 24)
  (h2 : chloe_wins * 3 = max_wins * 8) : 
  max_wins = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_wins_l2680_268001


namespace NUMINAMATH_CALUDE_towel_shrinkage_l2680_268073

theorem towel_shrinkage (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let shrunk_length := 0.8 * L
  let shrunk_breadth := 0.9 * B
  let shrunk_area := shrunk_length * shrunk_breadth
  let cumulative_shrunk_area := 0.95 * shrunk_area
  let folded_area := 0.5 * cumulative_shrunk_area
  let percentage_change := (folded_area - original_area) / original_area * 100
  percentage_change = -65.8 := by
sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l2680_268073


namespace NUMINAMATH_CALUDE_range_of_g_l2680_268038

-- Define the function g(x)
def g (x : ℝ) : ℝ := 3 * (x + 5)

-- State the theorem
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ 1 ∧ g x = y) ↔ y ≠ 18 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2680_268038


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l2680_268028

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℝ) (num_days : ℕ) (h1 : total_earnings = 120) (h2 : num_days = 20) :
  total_earnings / num_days = 6 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l2680_268028


namespace NUMINAMATH_CALUDE_base8_digit_product_12345_l2680_268052

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Computes the product of a list of natural numbers --/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

/-- The product of the digits in the base 8 representation of 12345 (base 10) is 0 --/
theorem base8_digit_product_12345 :
  listProduct (toBase8 12345) = 0 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_12345_l2680_268052


namespace NUMINAMATH_CALUDE_vector_dot_product_l2680_268077

/-- Given two vectors a and b in ℝ² satisfying certain conditions, 
    their dot product is equal to -222/25 -/
theorem vector_dot_product (a b : ℝ × ℝ) 
    (h1 : a.1 + 2 * b.1 = 1 ∧ a.2 + 2 * b.2 = -3)
    (h2 : 2 * a.1 - b.1 = 1 ∧ 2 * a.2 - b.2 = 9) :
    a.1 * b.1 + a.2 * b.2 = -222 / 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2680_268077


namespace NUMINAMATH_CALUDE_total_muffins_after_baking_l2680_268013

def initial_muffins : ℕ := 35
def additional_muffins : ℕ := 48

theorem total_muffins_after_baking :
  initial_muffins + additional_muffins = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_muffins_after_baking_l2680_268013


namespace NUMINAMATH_CALUDE_prob_three_draws_equals_36_125_l2680_268090

/-- The probability of drawing exactly 3 balls to get two red balls -/
def prob_three_draws (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  2 * (p_red * p_white * p_red)

/-- The box contains 3 red balls and 2 white balls -/
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def total_balls : ℕ := red_balls + white_balls

theorem prob_three_draws_equals_36_125 :
  prob_three_draws total_balls red_balls white_balls = 36 / 125 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_draws_equals_36_125_l2680_268090


namespace NUMINAMATH_CALUDE_expand_expression_l2680_268074

theorem expand_expression (x : ℝ) : 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1) = 10 * x^3 - 15 * x^2 + 20 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2680_268074


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l2680_268000

theorem ball_hitting_ground_time :
  ∃ t : ℝ, t > 0 ∧ -10 * t^2 - 20 * t + 180 = 0 ∧ t = 3 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l2680_268000


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2680_268005

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 10 = 4 * x + 5 ∧ x = -15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2680_268005


namespace NUMINAMATH_CALUDE_evaluate_expression_l2680_268025

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = 8) :
  x^2 * y^3 * z = 27/128 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2680_268025


namespace NUMINAMATH_CALUDE_dot_product_specific_vectors_l2680_268070

/-- Given two vectors a and b in a 2D plane with specific magnitudes and angle between them,
    prove that the dot product of a and (a + b) is 12. -/
theorem dot_product_specific_vectors (a b : ℝ × ℝ) :
  ‖a‖ = 4 →
  ‖b‖ = Real.sqrt 2 →
  a • b = -4 →
  a • (a + b) = 12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_specific_vectors_l2680_268070


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2680_268079

/-- A quadratic function with specific properties -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 3.5 and p(0) = 2, p(20) = 2 -/
theorem quadratic_symmetry (A B C : ℝ) :
  (∀ x : ℝ, p A B C (3.5 + x) = p A B C (3.5 - x)) →  -- Axis of symmetry at x = 3.5
  p A B C 0 = 2 →                                     -- p(0) = 2
  p A B C 20 = 2 :=                                   -- Conclusion: p(20) = 2
by
  sorry


end NUMINAMATH_CALUDE_quadratic_symmetry_l2680_268079


namespace NUMINAMATH_CALUDE_charity_event_volunteers_l2680_268007

theorem charity_event_volunteers (n : ℕ) : 
  (n : ℚ) / 2 = (((n : ℚ) / 2 - 3) / n) * n → n / 2 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_charity_event_volunteers_l2680_268007


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l2680_268003

theorem seventh_root_unity_product (s : ℂ) (h1 : s^7 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) * (s^6 - 1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l2680_268003


namespace NUMINAMATH_CALUDE_remainder_of_645_l2680_268092

-- Define the set s
def s : Set ℕ := {n : ℕ | n > 0 ∧ ∃ k, n = 8 * k + 5}

-- Define the 81st element of s
def element_81 : ℕ := 645

-- Theorem statement
theorem remainder_of_645 : 
  element_81 ∈ s ∧ (∃ k : ℕ, element_81 = 8 * k + 5) :=
by sorry

end NUMINAMATH_CALUDE_remainder_of_645_l2680_268092


namespace NUMINAMATH_CALUDE_election_votes_l2680_268078

theorem election_votes (total_votes : ℕ) (winner_votes : ℕ) 
  (diff1 diff2 diff3 : ℕ) : 
  total_votes = 963 →
  winner_votes - diff1 + winner_votes - diff2 + winner_votes - diff3 + winner_votes = total_votes →
  diff1 = 53 →
  diff2 = 79 →
  diff3 = 105 →
  winner_votes = 300 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l2680_268078


namespace NUMINAMATH_CALUDE_compute_expression_l2680_268046

theorem compute_expression : 2 * ((3 + 7)^2 + (3^2 + 7^2)) = 316 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2680_268046


namespace NUMINAMATH_CALUDE_tower_surface_area_l2680_268084

/-- Represents a layer in the tower -/
structure Layer where
  cubes : ℕ
  exposed_top : ℕ
  exposed_sides : ℕ

/-- Represents the tower of cubes -/
def Tower : List Layer := [
  { cubes := 1, exposed_top := 1, exposed_sides := 5 },
  { cubes := 3, exposed_top := 3, exposed_sides := 8 },
  { cubes := 4, exposed_top := 4, exposed_sides := 6 },
  { cubes := 6, exposed_top := 6, exposed_sides := 0 }
]

/-- The total number of cubes in the tower -/
def total_cubes : ℕ := (Tower.map (·.cubes)).sum

/-- The exposed surface area of the tower -/
def exposed_surface_area : ℕ := 
  (Tower.map (·.exposed_top)).sum + (Tower.map (·.exposed_sides)).sum

theorem tower_surface_area : 
  total_cubes = 14 ∧ exposed_surface_area = 29 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_l2680_268084


namespace NUMINAMATH_CALUDE_definite_integral_abs_quadratic_l2680_268043

theorem definite_integral_abs_quadratic : ∫ x in (-2)..2, |x^2 - 2*x| = 8 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_abs_quadratic_l2680_268043


namespace NUMINAMATH_CALUDE_product_range_l2680_268055

theorem product_range (a b : ℝ) (g : ℝ → ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : g = fun x => 2^x) (h₄ : g a * g b = 2) : 
  0 < a * b ∧ a * b ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_product_range_l2680_268055


namespace NUMINAMATH_CALUDE_calculation_result_l2680_268057

theorem calculation_result : (101 * 2012 * 121) / 1111 / 503 = 44 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2680_268057


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2680_268039

theorem algebraic_expression_value : 
  ∀ (a b : ℝ), 
  (a * 1^3 + b * 1 + 2022 = 2020) → 
  (a * (-1)^3 + b * (-1) + 2023 = 2025) :=
by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2680_268039


namespace NUMINAMATH_CALUDE_f_2007_equals_neg_two_l2680_268098

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_around_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem f_2007_equals_neg_two
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_sym : symmetric_around_two f)
  (h_neg_three : f (-3) = -2) :
  f 2007 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_neg_two_l2680_268098


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l2680_268029

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l2680_268029


namespace NUMINAMATH_CALUDE_oplus_nested_equation_l2680_268063

def oplus (x y : ℝ) : ℝ := x^2 + 2*y

theorem oplus_nested_equation (a : ℝ) : oplus a (oplus a a) = 3*a^2 + 4*a := by
  sorry

end NUMINAMATH_CALUDE_oplus_nested_equation_l2680_268063


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2680_268062

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -33 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2680_268062


namespace NUMINAMATH_CALUDE_sport_participation_theorem_l2680_268037

/-- Represents the number of students who play various sports in a class -/
structure SportParticipation where
  total_students : ℕ
  basketball : ℕ
  cricket : ℕ
  baseball : ℕ
  basketball_cricket : ℕ
  cricket_baseball : ℕ
  basketball_baseball : ℕ
  all_three : ℕ

/-- Calculates the number of students who play at least one sport -/
def students_playing_at_least_one_sport (sp : SportParticipation) : ℕ :=
  sp.basketball + sp.cricket + sp.baseball - sp.basketball_cricket - sp.cricket_baseball - sp.basketball_baseball + sp.all_three

/-- Calculates the number of students who don't play any sport -/
def students_not_playing_any_sport (sp : SportParticipation) : ℕ :=
  sp.total_students - students_playing_at_least_one_sport sp

/-- Theorem stating the correct number of students playing at least one sport and not playing any sport -/
theorem sport_participation_theorem (sp : SportParticipation) 
  (h1 : sp.total_students = 40)
  (h2 : sp.basketball = 15)
  (h3 : sp.cricket = 20)
  (h4 : sp.baseball = 12)
  (h5 : sp.basketball_cricket = 5)
  (h6 : sp.cricket_baseball = 7)
  (h7 : sp.basketball_baseball = 3)
  (h8 : sp.all_three = 2) :
  students_playing_at_least_one_sport sp = 32 ∧ students_not_playing_any_sport sp = 8 := by
  sorry

end NUMINAMATH_CALUDE_sport_participation_theorem_l2680_268037


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2680_268097

theorem absolute_value_inequality (x : ℝ) :
  x ≠ 1 →
  (|(2 * x - 1) / (x - 1)| > 2) ↔ (x > 3/4 ∧ x < 1) ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2680_268097


namespace NUMINAMATH_CALUDE_parabola_c_value_l2680_268076

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value : 
  ∀ p : Parabola, 
  p.x_coord 3 = 5 → -- vertex at (5, 3)
  p.x_coord 1 = 7 → -- passes through (7, 1)
  p.c = 19/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2680_268076


namespace NUMINAMATH_CALUDE_negative_option_l2680_268089

theorem negative_option : ∃ (x : ℝ), x < 0 ∧ 
  x = -(-5)^2 ∧ 
  -(-5) ≥ 0 ∧ 
  |-5| ≥ 0 ∧ 
  (-5) * (-5) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_option_l2680_268089


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2680_268099

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ p q : ℝ, (k₁ * (p^2 - 2*p) + 3*p + 7 = 0 ∧ 
               k₂ * (q^2 - 2*q) + 3*q + 7 = 0) ∧
              (p / q + q / p = 6 / 7)) →
  k₁ / k₂ + k₂ / k₁ = 14 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2680_268099


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_four_l2680_268067

theorem same_solution_implies_k_equals_four (x k : ℝ) :
  (8 * x - k = 2 * (x + 1)) ∧ 
  (2 * (2 * x - 3) = 1 - 3 * x) ∧ 
  (∃ x, (8 * x - k = 2 * (x + 1)) ∧ (2 * (2 * x - 3) = 1 - 3 * x)) →
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_four_l2680_268067


namespace NUMINAMATH_CALUDE_second_half_tickets_calculation_l2680_268058

/-- Calculates the number of tickets sold in the second half of the season -/
def tickets_second_half (total_tickets first_half_tickets : ℕ) : ℕ :=
  total_tickets - first_half_tickets

/-- Theorem stating that the number of tickets sold in the second half
    is the difference between total tickets and first half tickets -/
theorem second_half_tickets_calculation 
  (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) :
  tickets_second_half total_tickets first_half_tickets = 5703 := by
  sorry

end NUMINAMATH_CALUDE_second_half_tickets_calculation_l2680_268058


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2680_268006

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) =
  13.5 / (x - 4) + (-27) / (x - 2) + (-15) / (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2680_268006


namespace NUMINAMATH_CALUDE_function_inequality_l2680_268040

theorem function_inequality (f : ℤ → ℤ) 
  (h1 : ∀ k : ℤ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (h2 : f 4 = 25) : 
  ∀ k : ℤ, k ≥ 4 → f k ≥ k^2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l2680_268040


namespace NUMINAMATH_CALUDE_centroid_perpendicular_triangle_area_l2680_268065

/-- Given a triangle ABC with sides a, b, c, and area S, prove that the area of the triangle 
    formed by the bases of perpendiculars dropped from the centroid to the sides of ABC 
    is equal to (4/9) * (a² + b² + c²) / (a² * b² * c²) * S³ -/
theorem centroid_perpendicular_triangle_area 
  (a b c : ℝ) 
  (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : S > 0) : 
  ∃ (S_new : ℝ), S_new = (4/9) * (a^2 + b^2 + c^2) / (a^2 * b^2 * c^2) * S^3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_perpendicular_triangle_area_l2680_268065


namespace NUMINAMATH_CALUDE_first_student_stickers_l2680_268075

/-- Given a sequence of gold sticker counts for students 2 to 6, 
    prove that the first student received 29 stickers. -/
theorem first_student_stickers 
  (second : ℕ) 
  (third : ℕ) 
  (fourth : ℕ) 
  (fifth : ℕ) 
  (sixth : ℕ) 
  (h1 : second = 35) 
  (h2 : third = 41) 
  (h3 : fourth = 47) 
  (h4 : fifth = 53) 
  (h5 : sixth = 59) : 
  second - 6 = 29 := by
  sorry

end NUMINAMATH_CALUDE_first_student_stickers_l2680_268075


namespace NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2680_268012

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0)

/-- Theorem: For a linear function y = kx + b that does not pass through the third quadrant,
    k is negative and b is non-negative -/
theorem linear_function_not_in_third_quadrant
  (f : LinearFunctionNotInThirdQuadrant) : f.k < 0 ∧ f.b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_third_quadrant_l2680_268012


namespace NUMINAMATH_CALUDE_percent_composition_l2680_268022

theorem percent_composition (z : ℝ) (hz : z ≠ 0) :
  (42 / 100) * z = (60 / 100) * ((70 / 100) * z) := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l2680_268022


namespace NUMINAMATH_CALUDE_may_fourth_is_sunday_l2680_268009

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  fridayCount : Nat
  fridayDatesSum : Nat

/-- Returns the day of the week for a given date in the month -/
def dayOfWeek (m : Month) (date : Nat) : DayOfWeek := sorry

theorem may_fourth_is_sunday (m : Month) 
  (h1 : m.fridayCount = 5) 
  (h2 : m.fridayDatesSum = 80) : 
  dayOfWeek m 4 = DayOfWeek.Sunday := by sorry

end NUMINAMATH_CALUDE_may_fourth_is_sunday_l2680_268009


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l2680_268051

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : Nat
  red : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least one of each color -/
def minDrawForAllColors (counts : BallCounts) : Nat :=
  counts.white + counts.red + counts.black - 2

/-- The minimum number of balls to draw to ensure 10 balls of one color -/
def minDrawForTenOfOneColor (counts : BallCounts) : Nat :=
  min counts.white counts.red + min counts.white counts.black + 
  min counts.red counts.black + 10 - 1

/-- Theorem stating the correct answers for the given ball counts -/
theorem ball_drawing_theorem (counts : BallCounts) 
  (h1 : counts.white = 5) (h2 : counts.red = 12) (h3 : counts.black = 20) : 
  minDrawForAllColors counts = 33 ∧ minDrawForTenOfOneColor counts = 24 := by
  sorry

#eval minDrawForAllColors ⟨5, 12, 20⟩
#eval minDrawForTenOfOneColor ⟨5, 12, 20⟩

end NUMINAMATH_CALUDE_ball_drawing_theorem_l2680_268051


namespace NUMINAMATH_CALUDE_sqrt_35_between_5_and_6_l2680_268096

theorem sqrt_35_between_5_and_6 : 5 < Real.sqrt 35 ∧ Real.sqrt 35 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_35_between_5_and_6_l2680_268096


namespace NUMINAMATH_CALUDE_max_dimes_possible_l2680_268087

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- The total amount in cents -/
def total_amount : ℕ := 550

/-- Theorem stating the maximum number of dimes possible -/
theorem max_dimes_possible (quarters nickels dimes : ℕ) 
  (h1 : quarters = nickels)
  (h2 : dimes ≥ 3 * quarters)
  (h3 : quarters * coin_value "quarter" + 
        nickels * coin_value "nickel" + 
        dimes * coin_value "dime" = total_amount) :
  dimes ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_dimes_possible_l2680_268087


namespace NUMINAMATH_CALUDE_book_distribution_l2680_268054

/-- The number of books -/
def num_books : ℕ := 15

/-- The number of exercise books -/
def num_exercise_books : ℕ := 26

/-- The number of students in the first scenario -/
def students_scenario1 : ℕ := (num_exercise_books / 2)

/-- The number of students in the second scenario -/
def students_scenario2 : ℕ := (num_books / 3)

theorem book_distribution :
  (students_scenario1 + 2 = num_books) ∧
  (2 * students_scenario1 = num_exercise_books) ∧
  (3 * students_scenario2 = num_books) ∧
  (5 * students_scenario2 + 1 = num_exercise_books) :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l2680_268054


namespace NUMINAMATH_CALUDE_sine_graph_shift_l2680_268024

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x - 2 * π / 3)) = 2 * Real.sin (2 * ((x + 2 * π / 3) - 2 * π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l2680_268024


namespace NUMINAMATH_CALUDE_surface_is_cone_l2680_268041

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The equation of the surface in spherical coordinates -/
def surface_equation (c : ℝ) (p : SphericalPoint) : Prop :=
  p.ρ = c * Real.sin p.φ

/-- Definition of a cone in spherical coordinates -/
def is_cone (S : Set SphericalPoint) : Prop :=
  ∃ c > 0, ∀ p ∈ S, surface_equation c p

theorem surface_is_cone (c : ℝ) (hc : c > 0) :
    is_cone {p : SphericalPoint | surface_equation c p} := by
  sorry

end NUMINAMATH_CALUDE_surface_is_cone_l2680_268041


namespace NUMINAMATH_CALUDE_correct_num_pregnant_dogs_l2680_268021

/-- The number of pregnant dogs Chuck has. -/
def num_pregnant_dogs : ℕ := 3

/-- The number of puppies each pregnant dog gives birth to. -/
def puppies_per_dog : ℕ := 4

/-- The number of shots each puppy needs. -/
def shots_per_puppy : ℕ := 2

/-- The cost of each shot in dollars. -/
def cost_per_shot : ℕ := 5

/-- The total cost of all shots in dollars. -/
def total_cost : ℕ := 120

/-- Theorem stating that the number of pregnant dogs is correct given the conditions. -/
theorem correct_num_pregnant_dogs :
  num_pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_num_pregnant_dogs_l2680_268021


namespace NUMINAMATH_CALUDE_corner_sum_6x12_board_l2680_268011

/-- Represents a rectangular board filled with consecutive numbers -/
structure NumberBoard where
  rows : Nat
  cols : Nat
  total_numbers : Nat

/-- Returns the number at a given position on the board -/
def NumberBoard.number_at (board : NumberBoard) (row : Nat) (col : Nat) : Nat :=
  (row - 1) * board.cols + col

/-- Theorem stating that the sum of corner numbers on a 6x12 board is 146 -/
theorem corner_sum_6x12_board :
  let board : NumberBoard := ⟨6, 12, 72⟩
  (board.number_at 1 1) + (board.number_at 1 12) +
  (board.number_at 6 1) + (board.number_at 6 12) = 146 := by
  sorry

#check corner_sum_6x12_board

end NUMINAMATH_CALUDE_corner_sum_6x12_board_l2680_268011


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l2680_268048

def num_white_balls : ℕ := 3
def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 5
def num_boxes : ℕ := 3

def distribute_balls : ℕ := (Nat.choose (num_boxes + num_white_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_red_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_yellow_balls - 1) (num_boxes - 1))

theorem ball_distribution_theorem : distribute_balls = 3150 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l2680_268048


namespace NUMINAMATH_CALUDE_school_population_l2680_268068

theorem school_population (G B D : ℕ) (h1 : G = 5467) (h2 : D = 1932) (h3 : B = G - D) :
  G + B = 9002 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2680_268068


namespace NUMINAMATH_CALUDE_base6_addition_l2680_268008

/-- Convert a number from base 6 to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Definition of the first number in base 6 --/
def num1 : List Nat := [2, 3, 5, 4]

/-- Definition of the second number in base 6 --/
def num2 : List Nat := [3, 5, 2, 4, 2]

/-- Theorem stating that the sum of the two numbers in base 6 equals the result --/
theorem base6_addition :
  base10ToBase6 (base6ToBase10 num1 + base6ToBase10 num2) = [5, 2, 2, 2, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base6_addition_l2680_268008


namespace NUMINAMATH_CALUDE_farmer_remaining_apples_l2680_268045

def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

theorem farmer_remaining_apples : initial_apples - apples_given_away = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_remaining_apples_l2680_268045


namespace NUMINAMATH_CALUDE_inequality_chain_l2680_268072

theorem inequality_chain (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l2680_268072


namespace NUMINAMATH_CALUDE_mold_growth_problem_l2680_268018

/-- Calculates the number of mold spores after a given time period -/
def mold_growth (initial_spores : ℕ) (doubling_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  initial_spores * 2^(elapsed_time / doubling_time)

/-- The mold growth problem -/
theorem mold_growth_problem :
  let initial_spores : ℕ := 50
  let doubling_time : ℕ := 10  -- in minutes
  let elapsed_time : ℕ := 70   -- time from 9:00 a.m. to 10:10 a.m. in minutes
  mold_growth initial_spores doubling_time elapsed_time = 6400 :=
by
  sorry

end NUMINAMATH_CALUDE_mold_growth_problem_l2680_268018


namespace NUMINAMATH_CALUDE_triangle_max_area_l2680_268050

theorem triangle_max_area (x y : ℝ) (h : x + y = 418) :
  ⌊(1/2 : ℝ) * x * y⌋ ≤ 21840 ∧ ∃ (x' y' : ℝ), x' + y' = 418 ∧ ⌊(1/2 : ℝ) * x' * y'⌋ = 21840 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2680_268050
