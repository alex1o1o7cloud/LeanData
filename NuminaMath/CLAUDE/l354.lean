import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l354_35436

theorem rectangle_triangle_equal_area (b h : ℝ) : 
  b > 0 → 
  h > 0 → 
  h ≤ 2 → 
  b * h = (1/2) * b * (1 - h/2) → 
  h = 2/5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l354_35436


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_cubed_l354_35447

theorem opposite_of_negative_two_cubed : -((-2)^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_cubed_l354_35447


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l354_35476

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_f_at_2 : 
  deriv f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l354_35476


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_k_l354_35469

/-- Given a quadratic equation x^2 + 3x + k = 0 where x = -3 is a root, 
    prove that the other root is 0 and k = 0. -/
theorem quadratic_equation_roots_and_k (k : ℝ) : 
  ((-3 : ℝ)^2 + 3*(-3) + k = 0) → 
  (∃ (r : ℝ), r ≠ -3 ∧ r^2 + 3*r + k = 0 ∧ r = 0) ∧ 
  (k = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_k_l354_35469


namespace NUMINAMATH_CALUDE_number_difference_l354_35448

theorem number_difference (x y : ℤ) (h1 : x > y) (h2 : x + y = 64) (h3 : y = 26) : x - y = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l354_35448


namespace NUMINAMATH_CALUDE_mask_distribution_arrangements_l354_35482

/-- The number of ways to distribute n distinct objects among k distinct people,
    where each person must receive at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := sorry

theorem mask_distribution_arrangements :
  (distribute 7 4) * (permutations 4) = 8400 := by sorry

end NUMINAMATH_CALUDE_mask_distribution_arrangements_l354_35482


namespace NUMINAMATH_CALUDE_rational_function_property_l354_35400

theorem rational_function_property (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = 2 * f (x / 2) + 3 * f (y / 3)) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_rational_function_property_l354_35400


namespace NUMINAMATH_CALUDE_same_remainder_problem_l354_35408

theorem same_remainder_problem (x : ℕ+) : 
  (∃ q r : ℕ, 100 = q * x + r ∧ r < x) ∧ 
  (∃ p r : ℕ, 197 = p * x + r ∧ r < x) → 
  (∃ r : ℕ, 100 % x = r ∧ 197 % x = r ∧ r = 3) := by
sorry

end NUMINAMATH_CALUDE_same_remainder_problem_l354_35408


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l354_35465

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (1, -2)

theorem perpendicular_vectors_sum_magnitude (x : ℝ) :
  let a := vector_a x
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- perpendicular condition
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_magnitude_l354_35465


namespace NUMINAMATH_CALUDE_mike_pens_l354_35449

theorem mike_pens (initial_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ) :
  initial_pens = 7 →
  sharon_pens = 19 →
  final_pens = 39 →
  ∃ M : ℕ, 2 * (initial_pens + M) - sharon_pens = final_pens ∧ M = 22 :=
by sorry

end NUMINAMATH_CALUDE_mike_pens_l354_35449


namespace NUMINAMATH_CALUDE_equation_solutions_l354_35485

theorem equation_solutions : 
  ∀ x : ℝ, (4 * (3 * x)^2 + 3 * x + 5 = 3 * (8 * x^2 + 3 * x + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l354_35485


namespace NUMINAMATH_CALUDE_cubic_expansion_simplification_l354_35464

theorem cubic_expansion_simplification :
  (30 + 5)^3 - (30^3 + 3*30^2*5 + 3*30*5^2 + 5^3 - 5^3) = 125 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_simplification_l354_35464


namespace NUMINAMATH_CALUDE_count_rectangles_l354_35495

/-- The number of rectangles with sides parallel to the axes in an n×n grid -/
def num_rectangles (n : ℕ) : ℕ :=
  n^2 * (n-1)^2 / 4

/-- Theorem stating the number of rectangles in an n×n grid -/
theorem count_rectangles (n : ℕ) (h : n > 0) :
  num_rectangles n = (n.choose 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_count_rectangles_l354_35495


namespace NUMINAMATH_CALUDE_football_cost_l354_35477

theorem football_cost (total_spent marbles_cost baseball_cost : ℚ)
  (h1 : total_spent = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_spent - marbles_cost - baseball_cost = 4.95 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_l354_35477


namespace NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l354_35433

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (k : ℕ+), (4 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = k) ↔ p ∈ ({5, 8, 18, 50} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l354_35433


namespace NUMINAMATH_CALUDE_largest_vertex_sum_l354_35445

/-- Represents a parabola passing through specific points -/
structure Parabola (P : ℤ) where
  a : ℤ
  b : ℤ
  c : ℤ
  pass_origin : a * 0 * 0 + b * 0 + c = 0
  pass_3P : a * (3 * P) * (3 * P) + b * (3 * P) + c = 0
  pass_3P_minus_1 : a * (3 * P - 1) * (3 * P - 1) + b * (3 * P - 1) + c = 45

/-- Calculates the sum of coordinates of the vertex of a parabola -/
def vertexSum (P : ℤ) (p : Parabola P) : ℚ :=
  3 * P / 2 - (p.a : ℚ) * (9 * P^2 : ℚ) / 4

/-- Theorem stating the largest possible vertex sum -/
theorem largest_vertex_sum :
  ∀ P : ℤ, P ≠ 0 → ∀ p : Parabola P, vertexSum P p ≤ 138 := by sorry

end NUMINAMATH_CALUDE_largest_vertex_sum_l354_35445


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l354_35414

/-- Calculates the number of overtime hours worked given the regular pay rate,
    regular hours limit, and total pay received. -/
def overtime_hours (regular_rate : ℚ) (regular_hours_limit : ℕ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours_limit
  let overtime_rate := 2 * regular_rate
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the specified conditions, the number of overtime hours is 12. -/
theorem overtime_hours_calculation :
  let regular_rate : ℚ := 3
  let regular_hours_limit : ℕ := 40
  let total_pay : ℚ := 192
  overtime_hours regular_rate regular_hours_limit total_pay = 12 := by
  sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l354_35414


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l354_35430

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors created by combining 5 scoops from 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 5 3

/-- Theorem: The number of ice cream flavors is 21 -/
theorem ice_cream_flavors_count : ice_cream_flavors = 21 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l354_35430


namespace NUMINAMATH_CALUDE_streetlight_problem_l354_35473

/-- The number of streetlights --/
def n : ℕ := 2020

/-- The number of lights to be turned off --/
def k : ℕ := 300

/-- The number of ways to select k non-adjacent positions from n-2 positions --/
def non_adjacent_selections (n k : ℕ) : ℕ := Nat.choose (n - k - 1) k

theorem streetlight_problem :
  non_adjacent_selections n k = Nat.choose 1710 300 :=
sorry

end NUMINAMATH_CALUDE_streetlight_problem_l354_35473


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l354_35419

theorem range_of_m_for_quadratic_inequality (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x > m) ∧ 
  (∃ x : ℝ, x > m ∧ x^2 - 3*x + 2 ≥ 0) → 
  m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_inequality_l354_35419


namespace NUMINAMATH_CALUDE_problem_solution_l354_35460

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem problem_solution :
  -- Part I
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → a = 2) ∧
  -- Part II
  (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ 1) ∧
  (∀ k : ℝ, (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ k) → k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l354_35460


namespace NUMINAMATH_CALUDE_all_fruits_fall_on_day_14_l354_35412

/-- The number of fruits on the tree initially -/
def initial_fruits : ℕ := 60

/-- The number of fruits that fall on day n according to the original pattern -/
def fruits_falling (n : ℕ) : ℕ := n

/-- The sum of fruits that have fallen up to day n according to the original pattern -/
def sum_fallen (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of fruits remaining on the tree after n days of the original pattern -/
def fruits_remaining (n : ℕ) : ℕ := max 0 (initial_fruits - sum_fallen n)

/-- The day when the original pattern stops -/
def pattern_stop_day : ℕ := 10

/-- The number of days needed to finish the remaining fruits after the original pattern stops -/
def additional_days : ℕ := fruits_remaining pattern_stop_day

/-- The total number of days needed for all fruits to fall -/
def total_days : ℕ := pattern_stop_day + additional_days - 1

theorem all_fruits_fall_on_day_14 : total_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_all_fruits_fall_on_day_14_l354_35412


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l354_35452

theorem average_of_six_numbers (sequence : Fin 6 → ℝ) 
  (h1 : (sequence 0 + sequence 1 + sequence 2 + sequence 3) / 4 = 25)
  (h2 : (sequence 3 + sequence 4 + sequence 5) / 3 = 35)
  (h3 : sequence 3 = 25) :
  (sequence 0 + sequence 1 + sequence 2 + sequence 3 + sequence 4 + sequence 5) / 6 = 30 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l354_35452


namespace NUMINAMATH_CALUDE_angle_rewrite_and_terminal_sides_l354_35406

theorem angle_rewrite_and_terminal_sides (α : Real) (h : α = 1200 * π / 180) :
  ∃ (β k : Real),
    α = β + 2 * k * π ∧
    0 ≤ β ∧ β < 2 * π ∧
    β = 2 * π / 3 ∧
    k = 3 ∧
    (2 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    (-4 * π / 3 ∈ Set.Icc (-2 * π) (2 * π)) ∧
    ∃ (m n : ℤ),
      2 * π / 3 = α + 2 * m * π ∧
      -4 * π / 3 = α + 2 * n * π :=
by sorry

end NUMINAMATH_CALUDE_angle_rewrite_and_terminal_sides_l354_35406


namespace NUMINAMATH_CALUDE_condition_satisfies_equation_l354_35426

theorem condition_satisfies_equation (x y z : ℤ) (h1 : x = y + 1) (h2 : z = y) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_condition_satisfies_equation_l354_35426


namespace NUMINAMATH_CALUDE_ice_cream_line_problem_l354_35410

def Line := Fin 5 → Fin 5

def is_valid_line (l : Line) : Prop :=
  (∀ i j, i ≠ j → l i ≠ l j) ∧
  (∃ i, l i = 0) ∧
  (∃ i, l i = 1) ∧
  (∃ i, l i = 2) ∧
  (∃ i, l i = 3) ∧
  (∃ i, l i = 4)

theorem ice_cream_line_problem (l : Line) 
  (h_valid : is_valid_line l)
  (h_A_first : ∃ i, l i = 0)
  (h_B_next_to_A : ∃ i j, l i = 0 ∧ l j = 1 ∧ (i.val + 1 = j.val ∨ j.val + 1 = i.val))
  (h_C_second_to_last : ∃ i, l i = 3)
  (h_D_last : ∃ i, l i = 4)
  (h_E_remaining : ∃ i, l i = 2) :
  ∃ i j, l i = 2 ∧ l j = 1 ∧ (i.val = j.val + 1 ∨ j.val = i.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_line_problem_l354_35410


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l354_35450

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l354_35450


namespace NUMINAMATH_CALUDE_lines_no_common_points_implies_a_equals_negative_one_l354_35404

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ := λ x => (a^2 - a) * x + 1 - a
  line2 : ℝ → ℝ := λ x => 2 * x - 1

/-- The property that two lines have no points in common -/
def NoCommonPoints (l : TwoLines) : Prop :=
  ∀ x : ℝ, l.line1 x ≠ l.line2 x

/-- The theorem statement -/
theorem lines_no_common_points_implies_a_equals_negative_one (l : TwoLines) :
  NoCommonPoints l → l.a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_no_common_points_implies_a_equals_negative_one_l354_35404


namespace NUMINAMATH_CALUDE_robot_competition_max_weight_l354_35472

/-- The weight of the standard robot in pounds -/
def standard_robot_weight : ℝ := 100

/-- The minimum additional weight above the standard robot weight -/
def min_additional_weight : ℝ := 5

/-- The minimum weight of a robot in the competition -/
def min_robot_weight : ℝ := standard_robot_weight + min_additional_weight

/-- The maximum weight multiplier relative to the minimum weight -/
def max_weight_multiplier : ℝ := 2

/-- The maximum weight of a robot in the competition -/
def max_robot_weight : ℝ := max_weight_multiplier * min_robot_weight

theorem robot_competition_max_weight :
  max_robot_weight = 210 := by sorry

end NUMINAMATH_CALUDE_robot_competition_max_weight_l354_35472


namespace NUMINAMATH_CALUDE_prob_three_primes_in_six_dice_l354_35446

-- Define a 12-sided die
def twelve_sided_die : Finset ℕ := Finset.range 12

-- Define prime numbers on a 12-sided die
def primes_on_die : Finset ℕ := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime on a single die
def prob_prime : ℚ := (primes_on_die.card : ℚ) / (twelve_sided_die.card : ℚ)

-- Define the probability of rolling a non-prime on a single die
def prob_non_prime : ℚ := 1 - prob_prime

-- Define the number of dice
def num_dice : ℕ := 6

-- Define the number of dice showing prime
def num_prime_dice : ℕ := 3

-- Theorem statement
theorem prob_three_primes_in_six_dice : 
  (Nat.choose num_dice num_prime_dice : ℚ) * 
  (prob_prime ^ num_prime_dice) * 
  (prob_non_prime ^ (num_dice - num_prime_dice)) = 857500 / 2985984 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_primes_in_six_dice_l354_35446


namespace NUMINAMATH_CALUDE_cats_and_fish_l354_35481

theorem cats_and_fish (c d : ℕ) : 
  (6 : ℕ) * (1 : ℕ) * (6 : ℕ) = (6 : ℕ) * (6 : ℕ) →  -- 6 cats eat 6 fish in 1 day
  c * d * (1 : ℕ) = (91 : ℕ) →                      -- c cats eat 91 fish in d days
  1 < c →                                           -- c is more than 1
  c < (10 : ℕ) →                                    -- c is less than 10
  c + d = (20 : ℕ) :=                               -- prove that c + d = 20
by sorry

end NUMINAMATH_CALUDE_cats_and_fish_l354_35481


namespace NUMINAMATH_CALUDE_max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l354_35437

theorem max_a_for_nonpositive_f (a : ℝ) : 
  (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) → a ≤ 1 :=
by sorry

theorem existence_of_m_for_a_eq_1 : 
  ∃ m : ℝ, m > 0 ∧ m^3 - m^2 + (1^2 - 2)*m + 1 ≤ 0 :=
by sorry

theorem max_a_equals_one : 
  (∃ a : ℝ, (∃ m : ℝ, m > 0 ∧ m^3 - a*m^2 + (a^2 - 2)*m + 1 ≤ 0) ∧ 
    ∀ b : ℝ, (∃ n : ℝ, n > 0 ∧ n^3 - b*n^2 + (b^2 - 2)*n + 1 ≤ 0) → b ≤ a) ∧
  (∃ m : ℝ, m > 0 ∧ m^3 - 1*m^2 + (1^2 - 2)*m + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_nonpositive_f_existence_of_m_for_a_eq_1_max_a_equals_one_l354_35437


namespace NUMINAMATH_CALUDE_largest_two_digit_number_from_3_and_6_l354_35421

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ 
  ((n / 10 = 3 ∧ n % 10 = 6) ∨ (n / 10 = 6 ∧ n % 10 = 3))

theorem largest_two_digit_number_from_3_and_6 :
  ∀ n : ℕ, is_valid_number n → n ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_from_3_and_6_l354_35421


namespace NUMINAMATH_CALUDE_quarter_capacity_at_6_l354_35405

/-- Represents the volume of water in the pool as a fraction of its full capacity -/
def PoolVolume := Fin 9 → ℚ

/-- The pool's volume doubles every hour -/
def doubles (v : PoolVolume) : Prop :=
  ∀ i : Fin 8, v (i + 1) = 2 * v i

/-- The pool is full after 8 hours -/
def full_at_8 (v : PoolVolume) : Prop :=
  v 8 = 1

/-- The main theorem: If the pool's volume doubles every hour and is full after 8 hours,
    then it was at one quarter capacity after 6 hours -/
theorem quarter_capacity_at_6 (v : PoolVolume) 
  (h1 : doubles v) (h2 : full_at_8 v) : v 6 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quarter_capacity_at_6_l354_35405


namespace NUMINAMATH_CALUDE_problem_statement_l354_35462

theorem problem_statement (x y : ℤ) (a b : ℤ) : 
  (∃ k₁ k₂ : ℤ, x - 5 = 7 * k₁ ∧ y + 7 = 7 * k₂) →
  (∃ k₃ : ℤ, x^2 + y^3 = 11 * k₃) →
  x = 7 * a + 5 →
  y = 7 * b - 7 →
  (y - x) / 13 = (7 * (b - a) - 12) / 13 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l354_35462


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l354_35471

/-- Given two hyperbolas with equations x^2/9 - y^2/16 = 1 and y^2/25 - x^2/M = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/25 - x^2/M = 1) →
  (∀ x y : ℝ, |y| = (4/3) * |x| ↔ |y| = (5/Real.sqrt M) * |x|) →
  M = 225/16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l354_35471


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l354_35429

theorem prime_squared_minus_one_divisible_by_24 (n : ℕ) 
  (h_prime : Nat.Prime n) (h_gt_3 : n > 3) : 
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l354_35429


namespace NUMINAMATH_CALUDE_pyramid_volume_l354_35491

/-- Given a pyramid with a rhombus base (diagonals d₁ and d₂, where d₁ > d₂) and height passing
    through the vertex of the acute angle of the rhombus, if the area of the diagonal cross-section
    made through the smaller diagonal is Q, then the volume of the pyramid is
    (d₁/12) * √(16Q² - d₁²d₂²). -/
theorem pyramid_volume (d₁ d₂ Q : ℝ) (h₁ : d₁ > d₂) (h₂ : d₂ > 0) (h₃ : Q > 0) :
  let V := d₁ / 12 * Real.sqrt (16 * Q^2 - d₁^2 * d₂^2)
  ∃ (height : ℝ), height > 0 ∧ 
    (V = (1/3) * (1/2 * d₁ * d₂) * height) ∧
    (Q = (1/2) * d₂ * (2 * Q / d₂)) ∧
    (height = Real.sqrt ((2 * Q / d₂)^2 - (d₁ / 2)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l354_35491


namespace NUMINAMATH_CALUDE_complex_second_quadrant_l354_35499

theorem complex_second_quadrant (a : ℝ) : 
  let z : ℂ := a^2 * (1 + Complex.I) - a * (4 + Complex.I) - 6 * Complex.I
  (z.re < 0 ∧ z.im > 0) → (3 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_second_quadrant_l354_35499


namespace NUMINAMATH_CALUDE_x_twelve_percent_greater_than_seventy_l354_35416

theorem x_twelve_percent_greater_than_seventy (x : ℝ) : 
  x = 70 * (1 + 12 / 100) → x = 78.4 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_percent_greater_than_seventy_l354_35416


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l354_35444

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a = (1, m) and b = (3, 1), prove that if they are parallel, then m = 1/3 -/
theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (1, m) (3, 1) → m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l354_35444


namespace NUMINAMATH_CALUDE_expression_simplification_l354_35497

theorem expression_simplification (x y z w : ℝ) :
  (x - (y - (z - w))) - ((x - y) - (z - w)) = 2*z - 2*w := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l354_35497


namespace NUMINAMATH_CALUDE_man_mass_and_pressure_l354_35474

/-- Given a boat with specified dimensions and conditions, prove the mass and pressure exerted by a man --/
theorem man_mass_and_pressure (boat_length boat_breadth sink_depth : Real)
  (supplies_mass : Real) (water_density : Real) (gravity : Real)
  (h1 : boat_length = 6)
  (h2 : boat_breadth = 3)
  (h3 : sink_depth = 0.01)
  (h4 : supplies_mass = 15)
  (h5 : water_density = 1000)
  (h6 : gravity = 9.81) :
  ∃ (man_mass : Real) (pressure : Real),
    man_mass = 165 ∧
    pressure = 89.925 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_and_pressure_l354_35474


namespace NUMINAMATH_CALUDE_fair_die_weighted_coin_l354_35418

theorem fair_die_weighted_coin (n : ℕ) (p_heads : ℚ) : 
  n ≥ 7 →
  (p_heads = 1/3 ∨ p_heads = 2/3) →
  (1/n) * p_heads = 1/15 →
  n = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_die_weighted_coin_l354_35418


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l354_35409

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l354_35409


namespace NUMINAMATH_CALUDE_cyclic_sum_factorization_l354_35455

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_factorization_l354_35455


namespace NUMINAMATH_CALUDE_cricket_team_size_l354_35438

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let captain_age : ℕ := 25
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 22
  let remaining_average_age : ℝ := team_average_age - 1
  (n : ℝ) * team_average_age = captain_age + wicket_keeper_age + (n - 2 : ℝ) * remaining_average_age →
  n = 11 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l354_35438


namespace NUMINAMATH_CALUDE_quadratic_max_value_l354_35428

theorem quadratic_max_value :
  ∃ (max : ℝ), max = 216 ∧ ∀ (s : ℝ), -3 * s^2 + 54 * s - 27 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l354_35428


namespace NUMINAMATH_CALUDE_preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l354_35490

-- Statement A
theorem preserve_inequality (a b c : ℝ) (h : a < b) (k : ℝ) (hk : k > 0) :
  k * a < k * b ∧ a / k < b / k := by sorry

-- Statement B
theorem arithmetic_harmonic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (a + b) / 2 > 2 * a * b / (a + b) := by sorry

-- Statement C
theorem max_product_fixed_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (s : ℝ) (hs : s = a + b) :
  a * b ≤ (s / 2) * (s / 2) := by sorry

-- Statement D
theorem inequality_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (1 / 3) * (a^2 + b^2) > ((1 / 3) * (a + b))^2 := by sorry

-- Statement E (incorrect)
theorem not_always_max_sum_fixed_product (P : ℝ → ℝ → Prop) :
  (∃ a b k, a > 0 ∧ b > 0 ∧ a * b = k ∧ a + b > 2 * Real.sqrt k) → 
  ¬(∀ x y, x > 0 → y > 0 → x * y = k → x + y ≤ 2 * Real.sqrt k) := by sorry

end NUMINAMATH_CALUDE_preserve_inequality_arithmetic_harmonic_mean_max_product_fixed_sum_inequality_squares_not_always_max_sum_fixed_product_l354_35490


namespace NUMINAMATH_CALUDE_system_solution_l354_35413

theorem system_solution (x y z : ℝ) 
  (eq1 : x^2 + 27 = -8*y + 10*z)
  (eq2 : y^2 + 196 = 18*z + 13*x)
  (eq3 : z^2 + 119 = -3*x + 30*y) :
  x + 3*y + 5*z = 127.5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l354_35413


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l354_35420

/-- The mass of a man who causes a boat to sink by a certain amount -/
def man_mass (boat_length boat_width boat_sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_width * boat_sink_depth * water_density

/-- Theorem stating that the mass of the man is 140 kg -/
theorem man_mass_on_boat :
  let boat_length : ℝ := 7
  let boat_width : ℝ := 2
  let boat_sink_depth : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000    -- kg/m³
  man_mass boat_length boat_width boat_sink_depth water_density = 140 := by
  sorry

#eval man_mass 7 2 0.01 1000

end NUMINAMATH_CALUDE_man_mass_on_boat_l354_35420


namespace NUMINAMATH_CALUDE_rectangle_area_l354_35493

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 6 →
  ratio = 3 →
  (2 * r) * (ratio * 2 * r) = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l354_35493


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l354_35458

theorem x_gt_one_sufficient_not_necessary_for_x_squared_gt_one :
  (∀ x : ℝ, x > 1 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_squared_gt_one_l354_35458


namespace NUMINAMATH_CALUDE_k_value_l354_35403

theorem k_value (x : ℝ) (h1 : x ≠ 0) (h2 : 24 / x = k) : k = 24 / x := by
  sorry

end NUMINAMATH_CALUDE_k_value_l354_35403


namespace NUMINAMATH_CALUDE_problem_solution_l354_35486

theorem problem_solution : 
  let M : ℚ := 2007 / 3
  let N : ℚ := M / 3
  let X : ℚ := M - N
  X = 446 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l354_35486


namespace NUMINAMATH_CALUDE_samsung_start_is_15_l354_35424

/-- The number of Samsung cell phones left at the end of the day -/
def samsung_end : ℕ := 10

/-- The number of iPhones left at the end of the day -/
def iphone_end : ℕ := 5

/-- The number of damaged Samsung cell phones thrown out during the day -/
def samsung_thrown : ℕ := 2

/-- The number of defective iPhones thrown out during the day -/
def iphone_thrown : ℕ := 1

/-- The total number of cell phones sold today -/
def total_sold : ℕ := 4

/-- The number of Samsung cell phones David started the day with -/
def samsung_start : ℕ := samsung_end + samsung_thrown + (total_sold - iphone_thrown)

theorem samsung_start_is_15 : samsung_start = 15 := by
  sorry

end NUMINAMATH_CALUDE_samsung_start_is_15_l354_35424


namespace NUMINAMATH_CALUDE_balloon_comparison_l354_35467

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem balloon_comparison : balloon_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_comparison_l354_35467


namespace NUMINAMATH_CALUDE_ghost_castle_paths_l354_35453

theorem ghost_castle_paths (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_castle_paths_l354_35453


namespace NUMINAMATH_CALUDE_angle_B_value_side_lengths_l354_35484

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths

-- Define the conditions
axiom triangle_condition : 2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2)
axiom side_b : b = 3
axiom angle_relation : Real.sin C = 2 * Real.sin A

-- Theorem 1: Prove that B = π/3
theorem angle_B_value : 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → B = π/3 := by sorry

-- Theorem 2: Prove that a = √3 and c = 2√3
theorem side_lengths : 
  b = 3 → 
  Real.sin C = 2 * Real.sin A → 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → 
  a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_B_value_side_lengths_l354_35484


namespace NUMINAMATH_CALUDE_inequality_proof_l354_35457

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l354_35457


namespace NUMINAMATH_CALUDE_max_min_difference_f_l354_35411

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem max_min_difference_f : 
  (⨆ x ∈ (Set.Icc (-3) 0), f x) - (⨅ x ∈ (Set.Icc (-3) 0), f x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_f_l354_35411


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l354_35478

/-- Calculates the total distance walked given the number of blocks and distance per block -/
def total_distance (blocks_east blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Proves that walking 8 blocks east and 10 blocks north, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem arthur_walk_distance :
  total_distance 8 10 (1/4) = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_arthur_walk_distance_l354_35478


namespace NUMINAMATH_CALUDE_books_bought_equals_difference_l354_35454

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought : ℕ := sorry

/-- Melanie's initial number of books -/
def initial_books : ℕ := 41

/-- Melanie's final number of books after the yard sale -/
def final_books : ℕ := 87

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_equals_difference : 
  books_bought = final_books - initial_books :=
by sorry

end NUMINAMATH_CALUDE_books_bought_equals_difference_l354_35454


namespace NUMINAMATH_CALUDE_square_remainder_l354_35487

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l354_35487


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l354_35463

theorem quadratic_equation_with_given_roots (x y : ℝ) 
  (h : x^2 - 6*x + 9 = -|y - 1|) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ z : ℝ, a*z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ∧
    a = 1 ∧ b = -4 ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l354_35463


namespace NUMINAMATH_CALUDE_equal_distance_point_sum_of_distances_equal_distance_time_l354_35443

def A : ℝ := -1
def B : ℝ := 3

theorem equal_distance_point (x : ℝ) : 
  |x - A| = |x - B| → x = 1 := by sorry

theorem sum_of_distances (x : ℝ) : 
  (|x - A| + |x - B| = 5) ↔ (x = -3/2 ∨ x = 7/2) := by sorry

def P (t : ℝ) : ℝ := -t
def A' (t : ℝ) : ℝ := -1 - 5*t
def B' (t : ℝ) : ℝ := 3 - 20*t

theorem equal_distance_time (t : ℝ) : 
  |P t - A' t| = |P t - B' t| ↔ (t = 4/15 ∨ t = 2/23) := by sorry

end NUMINAMATH_CALUDE_equal_distance_point_sum_of_distances_equal_distance_time_l354_35443


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l354_35417

theorem greatest_three_digit_divisible_by_3_6_5 :
  ∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
  n % 3 = 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧
  n = 990 ∧
  ∀ (m : ℕ), 100 ≤ m ∧ m ≤ 999 ∧ 
  m % 3 = 0 ∧ m % 6 = 0 ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_3_6_5_l354_35417


namespace NUMINAMATH_CALUDE_all_propositions_false_l354_35431

-- Define the basic types
variable (Line Plane : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)

-- Define the "passes through" relation for planes and lines
variable (passes_through : Plane → Line → Prop)

-- Define the "within" relation for lines and planes
variable (within : Line → Plane → Prop)

-- Define the "has common points" relation
variable (has_common_points : Line → Line → Prop)

-- Define a proposition for "countless lines within a plane"
variable (countless_parallel_lines : Line → Plane → Prop)

-- State the theorem
theorem all_propositions_false :
  -- Proposition 1
  (∀ l1 l2 : Line, ∀ p : Plane, 
    parallel l1 l2 → passes_through p l2 → parallelLP l1 p) ∧
  -- Proposition 2
  (∀ l : Line, ∀ p : Plane,
    parallelLP l p → 
    (∀ l2 : Line, within l2 p → ¬(has_common_points l l2)) ∧
    (∀ l2 : Line, within l2 p → parallel l l2)) ∧
  -- Proposition 3
  (∀ l : Line, ∀ p : Plane,
    ¬(parallelLP l p) → ∀ l2 : Line, within l2 p → ¬(parallel l l2)) ∧
  -- Proposition 4
  (∀ l : Line, ∀ p : Plane,
    countless_parallel_lines l p → parallelLP l p)
  → False := by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l354_35431


namespace NUMINAMATH_CALUDE_second_sample_not_23_l354_35425

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total : ℕ  -- Total number of items
  sample_size : ℕ  -- Number of items to be sampled
  first_sample : ℕ  -- The first sampled item

/-- The second sample in a systematic sampling scheme -/
def second_sample (s : SystematicSampling) : ℕ :=
  s.first_sample + (s.total / s.sample_size)

/-- Theorem: The second sample cannot be 23 in the given systematic sampling scheme -/
theorem second_sample_not_23 (s : SystematicSampling) 
  (h1 : s.total > 0)
  (h2 : s.sample_size > 0)
  (h3 : s.sample_size ≤ s.total)
  (h4 : s.first_sample ≤ 10)
  (h5 : s.first_sample > 0)
  (h6 : s.sample_size = s.total / 10) :
  second_sample s ≠ 23 := by
  sorry

end NUMINAMATH_CALUDE_second_sample_not_23_l354_35425


namespace NUMINAMATH_CALUDE_math_quiz_items_l354_35415

theorem math_quiz_items (score_percentage : ℚ) (num_mistakes : ℕ) : 
  score_percentage = 80 → num_mistakes = 5 → 
  (100 : ℚ) * num_mistakes / (100 - score_percentage) = 25 :=
by sorry

end NUMINAMATH_CALUDE_math_quiz_items_l354_35415


namespace NUMINAMATH_CALUDE_equation_solution_l354_35451

theorem equation_solution : 
  {x : ℝ | (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7} = {-5/2, -4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l354_35451


namespace NUMINAMATH_CALUDE_f_two_roots_iff_m_range_f_min_value_on_interval_l354_35494

/-- The function f(x) = x^2 - 4mx + 6m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*m*x + 6*m

theorem f_two_roots_iff_m_range (m : ℝ) :
  (∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < 0 ∨ m > 3/2 := by sorry

theorem f_min_value_on_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f m x ≥ (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) ∧
  (∃ x ∈ Set.Icc 0 3, f m x = (
    if m ≤ 0 then 6*m
    else if m < 3/2 then -4*m^2 + 6*m
    else 9 - 6*m
  )) := by sorry

end NUMINAMATH_CALUDE_f_two_roots_iff_m_range_f_min_value_on_interval_l354_35494


namespace NUMINAMATH_CALUDE_deepak_age_l354_35489

/-- Proves that Deepak's present age is 18 years given the conditions -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l354_35489


namespace NUMINAMATH_CALUDE_cashback_discount_percentage_l354_35456

theorem cashback_discount_percentage
  (iphone_price : ℝ)
  (iwatch_price : ℝ)
  (iphone_discount : ℝ)
  (iwatch_discount : ℝ)
  (total_after_cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : iphone_discount = 0.15)
  (h4 : iwatch_discount = 0.10)
  (h5 : total_after_cashback = 931) :
  let discounted_iphone := iphone_price * (1 - iphone_discount)
  let discounted_iwatch := iwatch_price * (1 - iwatch_discount)
  let total_after_discounts := discounted_iphone + discounted_iwatch
  let cashback_amount := total_after_discounts - total_after_cashback
  let cashback_percentage := cashback_amount / total_after_discounts * 100
  cashback_percentage = 2 := by sorry

end NUMINAMATH_CALUDE_cashback_discount_percentage_l354_35456


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l354_35475

theorem sum_of_powers_of_i_is_zero : Complex.I + Complex.I^2 + Complex.I^3 + Complex.I^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_is_zero_l354_35475


namespace NUMINAMATH_CALUDE_expression_bounds_l354_35434

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2) (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ∧
  Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ≤ 8 ∧
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 2 ∧
    4 * Real.sqrt (t^2 + (2-t)^2) = 4 * Real.sqrt 2 ∨
    4 * Real.sqrt (t^2 + (2-t)^2) = 8 :=
by sorry


end NUMINAMATH_CALUDE_expression_bounds_l354_35434


namespace NUMINAMATH_CALUDE_dogwood_trees_after_planting_l354_35423

/-- The number of dogwood trees in the park after planting is equal to the sum of 
    the initial number of trees and the number of trees planted. -/
theorem dogwood_trees_after_planting (initial_trees planted_trees : ℕ) :
  initial_trees = 34 → planted_trees = 49 → initial_trees + planted_trees = 83 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_after_planting_l354_35423


namespace NUMINAMATH_CALUDE_westward_movement_l354_35407

-- Define the direction as an enumeration
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem westward_movement :
  movement 1000 Direction.West = -1000 :=
by sorry

end NUMINAMATH_CALUDE_westward_movement_l354_35407


namespace NUMINAMATH_CALUDE_train_length_calculation_train2_length_l354_35466

/-- Calculates the length of a train given the conditions of two trains passing each other. -/
theorem train_length_calculation (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 * 1000 / 3600 + speed_train2 * 1000 / 3600
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- The length of Train 2 is approximately 269.95 meters. -/
theorem train2_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 230 120 80 9 - 269.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_train2_length_l354_35466


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l354_35461

theorem quadratic_roots_problem (m : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x : ℝ, x^2 - (2*m - 1)*x + m^2 = 0) →
  (x₁^2 - (2*m - 1)*x₁ + m^2 = 0) →
  (x₂^2 - (2*m - 1)*x₂ + m^2 = 0) →
  (x₁ ≠ x₂) →
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l354_35461


namespace NUMINAMATH_CALUDE_melanie_gave_27_apples_l354_35439

/-- The number of apples Joan picked from the orchard -/
def apples_picked : ℕ := 43

/-- The total number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def apples_from_melanie : ℕ := total_apples - apples_picked

theorem melanie_gave_27_apples : apples_from_melanie = 27 := by
  sorry

end NUMINAMATH_CALUDE_melanie_gave_27_apples_l354_35439


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l354_35459

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := ((Complex.I - 1)^2 + 4) / (Complex.I + 1)
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l354_35459


namespace NUMINAMATH_CALUDE_cost_of_fifty_roses_l354_35480

/-- Represents the cost of a bouquet of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  if roses ≤ 30 then
    (30 : ℚ) / 15 * roses
  else
    (30 : ℚ) / 15 * 30 + (30 : ℚ) / 15 / 2 * (roses - 30)

/-- The theorem stating the cost of a bouquet with 50 roses -/
theorem cost_of_fifty_roses : bouquetCost 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_fifty_roses_l354_35480


namespace NUMINAMATH_CALUDE_simplify_expression_l354_35401

theorem simplify_expression (m n : ℝ) : m - n - (m - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l354_35401


namespace NUMINAMATH_CALUDE_line_slope_one_m_value_l354_35422

/-- Given a line passing through points P (-2, m) and Q (m, 4) with a slope of 1,
    prove that the value of m is 1. -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_one_m_value_l354_35422


namespace NUMINAMATH_CALUDE_divide_ten_with_difference_five_l354_35488

theorem divide_ten_with_difference_five :
  ∀ x y : ℝ, x + y = 10 ∧ y - x = 5 → x = (5 : ℝ) / 2 ∧ y = (15 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_divide_ten_with_difference_five_l354_35488


namespace NUMINAMATH_CALUDE_circle_area_ratio_l354_35498

theorem circle_area_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0) (h : R = 0.2 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l354_35498


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l354_35432

theorem favorite_fruit_strawberries (total students_oranges students_pears students_apples : ℕ) 
  (h_total : total = 450)
  (h_oranges : students_oranges = 70)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147) :
  total - (students_oranges + students_pears + students_apples) = 113 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l354_35432


namespace NUMINAMATH_CALUDE_dave_race_walking_time_l354_35427

theorem dave_race_walking_time 
  (total_time : ℕ) 
  (jogging_ratio : ℕ) 
  (walking_ratio : ℕ) 
  (h1 : total_time = 21)
  (h2 : jogging_ratio = 4)
  (h3 : walking_ratio = 3) :
  (walking_ratio * total_time) / (jogging_ratio + walking_ratio) = 9 := by
sorry


end NUMINAMATH_CALUDE_dave_race_walking_time_l354_35427


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l354_35440

/-- Represents the probability of a team member answering correctly -/
structure TeamMember where
  prob_correct : ℝ
  prob_correct_nonneg : 0 ≤ prob_correct
  prob_correct_le_one : prob_correct ≤ 1

/-- Represents a team in the knowledge competition -/
structure Team where
  member_a : TeamMember
  member_b : TeamMember

/-- The total score of a team in the competition -/
inductive TotalScore
  | zero
  | ten
  | twenty
  | thirty

def prob_first_correct (team : Team) : ℝ :=
  team.member_a.prob_correct + (1 - team.member_a.prob_correct) * team.member_b.prob_correct

def prob_distribution (team : Team) : TotalScore → ℝ
  | TotalScore.zero => (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.ten => prob_first_correct team * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.twenty => (prob_first_correct team)^2 * (1 - team.member_a.prob_correct) * (1 - team.member_b.prob_correct)
  | TotalScore.thirty => (prob_first_correct team)^3

theorem knowledge_competition_probabilities (team : Team)
  (h_a : team.member_a.prob_correct = 2/5)
  (h_b : team.member_b.prob_correct = 2/3) :
  prob_first_correct team = 4/5 ∧
  prob_distribution team TotalScore.zero = 1/5 ∧
  prob_distribution team TotalScore.ten = 4/25 ∧
  prob_distribution team TotalScore.twenty = 16/125 ∧
  prob_distribution team TotalScore.thirty = 64/125 := by
  sorry

#check knowledge_competition_probabilities

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l354_35440


namespace NUMINAMATH_CALUDE_three_fourths_cubed_l354_35483

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_three_fourths_cubed_l354_35483


namespace NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l354_35468

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l354_35468


namespace NUMINAMATH_CALUDE_grace_weekly_charge_l354_35435

/-- Grace's weekly charge given her total earnings and work duration -/
def weekly_charge (total_earnings : ℚ) (weeks : ℕ) : ℚ :=
  total_earnings / weeks

/-- Theorem: Grace's weekly charge is $300 -/
theorem grace_weekly_charge :
  let total_earnings : ℚ := 1800
  let weeks : ℕ := 6
  weekly_charge total_earnings weeks = 300 := by
  sorry

end NUMINAMATH_CALUDE_grace_weekly_charge_l354_35435


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l354_35492

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 1 / x ≥ 4 ∧ ∃ y > 0, 3 * Real.sqrt y + 1 / y = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l354_35492


namespace NUMINAMATH_CALUDE_triangle_properties_l354_35402

/-- Triangle ABC with vertices A(1,3), B(3,1), and C(-1,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle := {
  A := (1, 3)
  B := (3, 1)
  C := (-1, 0)
}

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to get the line equation of side AB -/
def getLineAB (t : Triangle) : LineEquation := sorry

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) (h : t = triangleABC) : 
  getLineAB t = { a := 1, b := 1, c := -4 } ∧ triangleArea t = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l354_35402


namespace NUMINAMATH_CALUDE_inverse_24_mod_53_l354_35470

theorem inverse_24_mod_53 (h : (19⁻¹ : ZMod 53) = 31) : (24⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_24_mod_53_l354_35470


namespace NUMINAMATH_CALUDE_total_markers_l354_35479

theorem total_markers (red_markers blue_markers : ℕ) :
  red_markers = 41 → blue_markers = 64 → red_markers + blue_markers = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_total_markers_l354_35479


namespace NUMINAMATH_CALUDE_sally_initial_peaches_l354_35496

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The final number of peaches at Sally's stand -/
def final_peaches : ℕ := 55

/-- The initial number of peaches at Sally's stand -/
def initial_peaches : ℕ := final_peaches - picked_peaches

theorem sally_initial_peaches : initial_peaches = 13 := by
  sorry

end NUMINAMATH_CALUDE_sally_initial_peaches_l354_35496


namespace NUMINAMATH_CALUDE_bus_ride_difference_l354_35442

def tess_to_noah : ℝ := 0.75
def tess_noah_to_kayla : ℝ := 0.85
def tess_kayla_to_school : ℝ := 1.15

def oscar_to_charlie : ℝ := 0.25
def oscar_charlie_to_school : ℝ := 1.35

theorem bus_ride_difference : 
  (tess_to_noah + tess_noah_to_kayla + tess_kayla_to_school) - 
  (oscar_to_charlie + oscar_charlie_to_school) = 1.15 := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_difference_l354_35442


namespace NUMINAMATH_CALUDE_prob_five_heads_five_tails_l354_35441

/-- Represents the state of the coin after some number of flips. -/
structure CoinState where
  heads : ℕ
  tails : ℕ

/-- The probability of getting heads given the current state of the coin. -/
def prob_heads (state : CoinState) : ℚ :=
  (state.heads + 1) / (state.heads + state.tails + 2)

/-- The probability of a specific sequence of 10 flips resulting in exactly 5 heads and 5 tails. -/
def prob_sequence : ℚ := 1 / 39916800

/-- The number of ways to arrange 5 heads and 5 tails in 10 flips. -/
def num_sequences : ℕ := 252

/-- The theorem stating the probability of getting exactly 5 heads and 5 tails after 10 flips. -/
theorem prob_five_heads_five_tails :
  num_sequences * prob_sequence = 1 / 158760 := by sorry

end NUMINAMATH_CALUDE_prob_five_heads_five_tails_l354_35441
