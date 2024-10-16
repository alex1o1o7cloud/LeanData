import Mathlib

namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1920_192064

theorem greatest_divisor_with_remainders : Nat.gcd (3461 - 23) (4783 - 41) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1920_192064


namespace NUMINAMATH_CALUDE_vector_c_value_l1920_192008

/-- Given vectors a and b, if vector c satisfies the parallel and perpendicular conditions,
    then c equals the specified vector. -/
theorem vector_c_value (a b c : ℝ × ℝ) : 
  a = (1, 2) →
  b = (2, -3) →
  (∃ k : ℝ, c + a = k • b) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  c = (-7/9, -7/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_value_l1920_192008


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1920_192026

-- Problem 1
theorem problem_1 (a : ℝ) (h : a ≠ 1) : 
  a^2 / (a - 1) - a - 1 = 1 / (a - 1) := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  (2 * x * y) / (x^2 - y^2) / ((1 / (x - y)) + (1 / (x + y))) = y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1920_192026


namespace NUMINAMATH_CALUDE_triangle_properties_l1920_192072

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle)
  (h1 : Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B))
  (h2 : t.c = Real.sqrt 3) :
  t.C = π / 3 ∧ 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1920_192072


namespace NUMINAMATH_CALUDE_infinitely_many_squarefree_n_squared_plus_one_l1920_192022

/-- A natural number is squarefree if it has no repeated prime factors -/
def IsSquarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ^ 2 ∣ n → p = 2)

/-- The set of positive integers n for which n^2 + 1 is squarefree -/
def SquarefreeSet : Set ℕ := {n : ℕ | n > 0 ∧ IsSquarefree (n^2 + 1)}

theorem infinitely_many_squarefree_n_squared_plus_one : Set.Infinite SquarefreeSet := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_squarefree_n_squared_plus_one_l1920_192022


namespace NUMINAMATH_CALUDE_solve_equation_l1920_192088

theorem solve_equation (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1920_192088


namespace NUMINAMATH_CALUDE_ball_picking_probabilities_l1920_192073

/-- Represents the probability of selecting ball 3 using strategy 1 -/
def P₁ : ℚ := 1/3

/-- Represents the probability of selecting ball 3 using strategy 2 -/
def P₂ : ℚ := 1/2

/-- Represents the probability of selecting ball 3 using strategy 3 -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities : P₁ < P₂ ∧ P₁ < P₃ ∧ 2 * P₁ = P₃ := by
  sorry

end NUMINAMATH_CALUDE_ball_picking_probabilities_l1920_192073


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1920_192002

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1920_192002


namespace NUMINAMATH_CALUDE_system_solution_l1920_192092

theorem system_solution (x y : ℝ) (m n : ℤ) : 
  (4 * (Real.cos x)^2 * (Real.sin (x/6))^2 + 4 * Real.sin (x/6) - 4 * (Real.sin x)^2 * Real.sin (x/6) + 1 = 0 ∧
   Real.sin (x/4) = Real.sqrt (Real.cos y)) ↔ 
  ((x = 11 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n)) ∨
   (x = -5 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1920_192092


namespace NUMINAMATH_CALUDE_player_positions_satisfy_distances_l1920_192067

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end NUMINAMATH_CALUDE_player_positions_satisfy_distances_l1920_192067


namespace NUMINAMATH_CALUDE_quartic_roots_l1920_192063

theorem quartic_roots (x : ℝ) :
  (7 * x^4 - 50 * x^3 + 94 * x^2 - 50 * x + 7 = 0) ↔
  (x + 1/x = (50 + Real.sqrt 260)/14 ∨ x + 1/x = (50 - Real.sqrt 260)/14) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l1920_192063


namespace NUMINAMATH_CALUDE_certain_number_value_l1920_192003

theorem certain_number_value (x p n : ℕ) (h1 : x > 0) (h2 : Prime p) 
  (h3 : ∃ k : ℕ, Prime k ∧ Even k ∧ x = k * n * p) (h4 : x ≥ 44) 
  (h5 : ∀ y, y > 0 → y < x → ¬∃ k : ℕ, Prime k ∧ Even k ∧ y = k * n * p) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l1920_192003


namespace NUMINAMATH_CALUDE_midpoint_square_area_l1920_192018

theorem midpoint_square_area (large_square_area : ℝ) (large_square_area_pos : large_square_area > 0) :
  let large_side := Real.sqrt large_square_area
  let small_side := large_side / Real.sqrt 2
  let small_square_area := small_side ^ 2
  large_square_area = 100 → small_square_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_square_area_l1920_192018


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1920_192051

def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x + 7)^2 + (y + 2)^2) = 24

def focus1 : ℝ × ℝ := (1, 3)
def focus2 : ℝ × ℝ := (-7, -2)

theorem ellipse_foci_distance :
  let d := Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2)
  d = Real.sqrt 89 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1920_192051


namespace NUMINAMATH_CALUDE_work_completion_time_b_l1920_192094

/-- The number of days it takes for worker b to complete a work alone,
    given that workers a and b together can finish the work in 16 days,
    and worker a alone can do the same work in 32 days. -/
theorem work_completion_time_b (work_rate_a_and_b : ℚ) (work_rate_a : ℚ) :
  work_rate_a_and_b = 1 / 16 →
  work_rate_a = 1 / 32 →
  (1 : ℚ) / (work_rate_a_and_b - work_rate_a) = 32 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_b_l1920_192094


namespace NUMINAMATH_CALUDE_power_comparison_l1920_192069

theorem power_comparison : 3^17 < 8^9 ∧ 8^9 < 4^15 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l1920_192069


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1920_192015

theorem simple_interest_rate (P : ℝ) (h : P > 0) : 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P) → 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P ∧ R = 100 / 15) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1920_192015


namespace NUMINAMATH_CALUDE_device_usage_probability_l1920_192071

theorem device_usage_probability (pA pB pC : ℝ) 
  (hA : pA = 0.4) 
  (hB : pB = 0.5) 
  (hC : pC = 0.7) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) 
  (hpB : 0 ≤ pB ∧ pB ≤ 1) 
  (hpC : 0 ≤ pC ∧ pC ≤ 1) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.91 := by
  sorry

end NUMINAMATH_CALUDE_device_usage_probability_l1920_192071


namespace NUMINAMATH_CALUDE_rain_on_tuesdays_l1920_192098

theorem rain_on_tuesdays 
  (monday_count : ℕ) 
  (tuesday_count : ℕ) 
  (rain_per_monday : ℝ) 
  (total_rain_difference : ℝ) 
  (h1 : monday_count = 7)
  (h2 : tuesday_count = 9)
  (h3 : rain_per_monday = 1.5)
  (h4 : total_rain_difference = 12) :
  (monday_count * rain_per_monday + total_rain_difference) / tuesday_count = 2.5 := by
sorry

end NUMINAMATH_CALUDE_rain_on_tuesdays_l1920_192098


namespace NUMINAMATH_CALUDE_meeting_democrat_ratio_l1920_192080

/-- Given a meeting with participants, prove the ratio of male democrats to total male participants -/
theorem meeting_democrat_ratio 
  (total_participants : ℕ) 
  (female_democrats : ℕ) 
  (h_total : total_participants = 780)
  (h_female_dem : female_democrats = 130)
  (h_half_female : female_democrats * 2 ≤ total_participants)
  (h_third_dem : 3 * female_democrats * 2 = total_participants)
  : (total_participants - female_democrats * 2 - female_democrats) / 
    (total_participants - female_democrats * 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_meeting_democrat_ratio_l1920_192080


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_three_pi_halves_l1920_192082

theorem sum_of_solutions_is_three_pi_halves :
  ∃ (x₁ x₂ : Real),
    0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧
    0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧
    (1 / Real.sin x₁ + 1 / Real.cos x₁ = 4) ∧
    (1 / Real.sin x₂ + 1 / Real.cos x₂ = 4) ∧
    x₁ + x₂ = 3 * Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_three_pi_halves_l1920_192082


namespace NUMINAMATH_CALUDE_cube_volume_problem_l1920_192083

/-- Given a cube with side length a, prove that if we form a rectangular solid
    by increasing one edge by 2, decreasing another by 1, and leaving the third unchanged,
    and if the volume of this new solid is 14 more than the original cube,
    then the volume of the original cube is 64. -/
theorem cube_volume_problem (a : ℕ) : 
  (a + 2) * (a - 1) * a = a^3 + 14 → a^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l1920_192083


namespace NUMINAMATH_CALUDE_distinct_products_count_l1920_192037

def S : Finset ℕ := {2, 3, 5, 7, 13}

def products (s : Finset ℕ) : Finset ℕ :=
  (Finset.powerset s).filter (λ t => t.card ≥ 2) |>.image (λ t => t.prod id)

theorem distinct_products_count : (products S).card = 26 := by
  sorry

end NUMINAMATH_CALUDE_distinct_products_count_l1920_192037


namespace NUMINAMATH_CALUDE_recurrence_solution_l1920_192066

-- Define the recurrence relation
def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => 2 * a n + 2^(n + 1)

-- State the theorem
theorem recurrence_solution (n : ℕ) : a n = (n + 3) * 2^n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_solution_l1920_192066


namespace NUMINAMATH_CALUDE_train_length_l1920_192062

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 56.8) (h2 : time = 18) :
  speed * time = 1022.4 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1920_192062


namespace NUMINAMATH_CALUDE_lens_savings_l1920_192077

/-- The price of the more expensive lens before discount -/
def original_price : ℝ := 300

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.20

/-- The price of the cheaper lens -/
def cheaper_price : ℝ := 220

/-- The discounted price of the more expensive lens -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- The amount saved by buying the cheaper lens -/
def savings : ℝ := discounted_price - cheaper_price

theorem lens_savings : savings = 20 := by
  sorry

end NUMINAMATH_CALUDE_lens_savings_l1920_192077


namespace NUMINAMATH_CALUDE_carson_octopus_legs_l1920_192028

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := 5

/-- The number of legs each octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_octopus_legs : ℕ := num_octopuses * legs_per_octopus

theorem carson_octopus_legs : total_octopus_legs = 40 := by
  sorry

end NUMINAMATH_CALUDE_carson_octopus_legs_l1920_192028


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1920_192076

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x ↔ x > -3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1920_192076


namespace NUMINAMATH_CALUDE_chord_length_l1920_192013

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 9) →  -- Circle equation
  (x + y = 2 * Real.sqrt 2) →  -- Line equation
  ∃ (a b : ℝ), (a - x)^2 + (b - y)^2 = 25 ∧  -- Chord endpoints
               (a^2 + b^2 = 9) ∧  -- Endpoints on circle
               (a + b = 2 * Real.sqrt 2) :=  -- Endpoints on line
by sorry

end NUMINAMATH_CALUDE_chord_length_l1920_192013


namespace NUMINAMATH_CALUDE_average_difference_l1920_192097

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 80 + 15) / 3) + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1920_192097


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1920_192036

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 21| + |x - 17| = |2*x - 38| :=
by
  -- The unique solution is x = 19
  use 19
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1920_192036


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1920_192059

theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) : 
  total - (red + green) = 177 := by
sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1920_192059


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1920_192039

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price = 100)
  (h2 : gain_percent = 15) :
  cost_price * (1 + gain_percent / 100) = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1920_192039


namespace NUMINAMATH_CALUDE_prime_between_50_60_mod_7_l1920_192019

theorem prime_between_50_60_mod_7 :
  ∀ n : ℕ,
  (Prime n) →
  (50 < n) →
  (n < 60) →
  (n % 7 = 4) →
  n = 53 :=
by sorry

end NUMINAMATH_CALUDE_prime_between_50_60_mod_7_l1920_192019


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_is_zero_l1920_192089

theorem real_part_of_one_plus_i_squared_is_zero :
  Complex.re ((1 : ℂ) + Complex.I) ^ 2 = 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_is_zero_l1920_192089


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1920_192095

/-- Given an arithmetic sequence where a₁ = 1 and a₃ = 5, prove that a₁₀ = 19. -/
theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 5)  -- Third term is 5
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tenth_term_l1920_192095


namespace NUMINAMATH_CALUDE_equation_solution_l1920_192044

theorem equation_solution : 
  ∃ x : ℝ, x = (8 * Real.sqrt 2) / 3 ∧ 
  Real.sqrt (9 + Real.sqrt (16 + 3*x)) + Real.sqrt (3 + Real.sqrt (4 + x)) = 3 + 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1920_192044


namespace NUMINAMATH_CALUDE_a_spending_percentage_l1920_192029

/-- Proves that A spends 95% of his salary given the conditions of the problem -/
theorem a_spending_percentage (total_salary : ℝ) (a_salary : ℝ) (b_spending_percentage : ℝ) :
  total_salary = 6000 →
  a_salary = 4500 →
  b_spending_percentage = 0.85 →
  let b_salary := total_salary - a_salary
  let a_savings := a_salary * (1 - (95 / 100))
  let b_savings := b_salary * (1 - b_spending_percentage)
  a_savings = b_savings :=
by sorry

end NUMINAMATH_CALUDE_a_spending_percentage_l1920_192029


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1920_192045

/-- An isosceles triangle with perimeter 60 and two equal sides of length x has a base of length 60 - 2x -/
theorem isosceles_triangle_base_length (x : ℝ) (h : x > 0) : 
  let y := 60 - 2*x
  (2*x + y = 60) ∧ (y = -2*x + 60) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1920_192045


namespace NUMINAMATH_CALUDE_inequality_always_true_l1920_192035

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1920_192035


namespace NUMINAMATH_CALUDE_inequality_proof_l1920_192084

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 2) :
  x^2 * y^2 + |x^2 - y^2| ≤ π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1920_192084


namespace NUMINAMATH_CALUDE_quadratic_minimum_at_positive_l1920_192046

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 6

-- Theorem statement
theorem quadratic_minimum_at_positive (x : ℝ) :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (y : ℝ), f y ≥ f x_min :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_at_positive_l1920_192046


namespace NUMINAMATH_CALUDE_boat_purchase_problem_l1920_192075

theorem boat_purchase_problem (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 40 := by sorry

end NUMINAMATH_CALUDE_boat_purchase_problem_l1920_192075


namespace NUMINAMATH_CALUDE_intersection_condition_l1920_192041

-- Define the sets A and B
def A (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = -p.1^2 + m*p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3 ∧ 0 ≤ p.1 ∧ p.1 ≤ 3}

-- Define the condition for exactly one intersection
def exactly_one_intersection (m : ℝ) : Prop :=
  ∃! p, p ∈ A m ∩ B

-- State the theorem
theorem intersection_condition (m : ℝ) :
  exactly_one_intersection m ↔ (m = 3 ∨ m > 10/3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1920_192041


namespace NUMINAMATH_CALUDE_value_of_b_l1920_192007

theorem value_of_b (x y : ℝ) : 
  x = (1 - Real.sqrt 3) / (1 + Real.sqrt 3) →
  y = (1 + Real.sqrt 3) / (1 - Real.sqrt 3) →
  2 * x^2 - 3 * x * y + 2 * y^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_value_of_b_l1920_192007


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l1920_192006

def total_balls : ℕ := 13
def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5

def prob_same_color : ℚ :=
  (blue_balls * blue_balls + yellow_balls * yellow_balls) / (total_balls * total_balls)

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l1920_192006


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1920_192016

theorem arithmetic_calculations :
  ((-15) + (-23) - 26 - (-15) = -49) ∧
  ((-1/2 + 2/3 - 1/4) * (-24) = 2) ∧
  ((-24) / (-6) * (-1/4) = -1) ∧
  ((-1)^2024 - (-2)^3 - 3^2 + 2 / (2/3) * (3/2) = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1920_192016


namespace NUMINAMATH_CALUDE_proportional_function_two_quadrants_l1920_192099

/-- A proportional function passing through two quadrants -/
theorem proportional_function_two_quadrants (m : ℝ) : 
  let f : ℝ → ℝ := λ x => (m + 3) * x^(m^2 + m - 5)
  m = 2 → (∃ x y, f x = y ∧ x > 0 ∧ y > 0) ∧ 
          (∃ x y, f x = y ∧ x < 0 ∧ y < 0) ∧
          (∀ x y, f x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)) :=
by sorry


end NUMINAMATH_CALUDE_proportional_function_two_quadrants_l1920_192099


namespace NUMINAMATH_CALUDE_largest_cube_minus_smallest_fifth_l1920_192056

theorem largest_cube_minus_smallest_fifth : ∃ (a b : ℕ), 
  (∀ n : ℕ, n^3 < 999 → n ≤ a) ∧ 
  (a^3 < 999) ∧
  (∀ m : ℕ, m^5 > 99 → b ≤ m) ∧ 
  (b^5 > 99) ∧
  (a - b = 6) := by
sorry

end NUMINAMATH_CALUDE_largest_cube_minus_smallest_fifth_l1920_192056


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1920_192023

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(1) = 0 and f(2) = 0, then f(-1) = 6 -/
theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  (f 1 = 0) → (f 2 = 0) → (f (-1) = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1920_192023


namespace NUMINAMATH_CALUDE_jackie_apples_l1920_192030

def adam_apples : ℕ := 10
def difference : ℕ := 8

theorem jackie_apples : ∃ j : ℕ, j = adam_apples - difference :=
  sorry

end NUMINAMATH_CALUDE_jackie_apples_l1920_192030


namespace NUMINAMATH_CALUDE_sequence_properties_l1920_192049

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sequence_properties (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 4 = 24 →
  b 1 = 0 →
  (∀ n : ℕ, b n + b (n + 1) = a n) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^(n - 1) + (-1)^n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1920_192049


namespace NUMINAMATH_CALUDE_unique_function_property_l1920_192093

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l1920_192093


namespace NUMINAMATH_CALUDE_distance_between_circumcenters_l1920_192058

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the side lengths of the triangle
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (dist t.A t.B, dist t.B t.C, dist t.C t.A)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ :=
  sorry

-- Theorem statement
theorem distance_between_circumcenters (t : Triangle) :
  let H := orthocenter t
  let side_len := side_lengths t
  side_len.1 = 13 ∧ side_len.2.1 = 14 ∧ side_len.2.2 = 15 →
  dist (circumcenter ⟨t.A, H, t.B⟩) (circumcenter ⟨t.A, H, t.C⟩) = 14 :=
sorry

end NUMINAMATH_CALUDE_distance_between_circumcenters_l1920_192058


namespace NUMINAMATH_CALUDE_largest_remainder_269_l1920_192011

theorem largest_remainder_269 (n : ℕ) (h : n < 150) :
  ∃ (q r : ℕ), 269 = n * q + r ∧ r < n ∧ r ≤ 133 ∧
  (∀ (q' r' : ℕ), 269 = n * q' + r' ∧ r' < n → r' ≤ r) :=
sorry

end NUMINAMATH_CALUDE_largest_remainder_269_l1920_192011


namespace NUMINAMATH_CALUDE_total_students_agreed_l1920_192053

def third_grade_total : ℕ := 256
def fourth_grade_total : ℕ := 525
def fifth_grade_total : ℕ := 410
def sixth_grade_total : ℕ := 600

def third_grade_percentage : ℚ := 60 / 100
def fourth_grade_percentage : ℚ := 45 / 100
def fifth_grade_percentage : ℚ := 35 / 100
def sixth_grade_percentage : ℚ := 55 / 100

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem total_students_agreed : 
  round_to_nearest (third_grade_percentage * third_grade_total) +
  round_to_nearest (fourth_grade_percentage * fourth_grade_total) +
  round_to_nearest (fifth_grade_percentage * fifth_grade_total) +
  round_to_nearest (sixth_grade_percentage * sixth_grade_total) = 864 := by
  sorry

end NUMINAMATH_CALUDE_total_students_agreed_l1920_192053


namespace NUMINAMATH_CALUDE_sqrt_x_minus_7_real_implies_x_geq_7_l1920_192042

theorem sqrt_x_minus_7_real_implies_x_geq_7 (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 7) → x ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_7_real_implies_x_geq_7_l1920_192042


namespace NUMINAMATH_CALUDE_factors_of_1320_l1920_192032

theorem factors_of_1320 : Nat.card (Nat.divisors 1320) = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l1920_192032


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1920_192031

-- Define the function f
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem solution_set_inequality (x : ℝ) :
  f (2 * x^2 - 1) < -1 ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1920_192031


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1920_192027

/-- A point P in ℝ² is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 < 0 ∧ P.2 > 0

/-- An angle θ is in the fourth quadrant if sin θ < 0 and cos θ > 0 -/
def in_fourth_quadrant (θ : ℝ) : Prop :=
  Real.sin θ < 0 ∧ Real.cos θ > 0

/-- If P(sin θ cos θ, 2cos θ) is in the second quadrant, then θ is in the fourth quadrant -/
theorem angle_in_fourth_quadrant (θ : ℝ) :
  in_second_quadrant (Real.sin θ * Real.cos θ, 2 * Real.cos θ) → in_fourth_quadrant θ :=
by sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1920_192027


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l1920_192050

theorem difference_of_squares_factorization (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l1920_192050


namespace NUMINAMATH_CALUDE_rationalized_factor_simplify_fraction_special_sqrt_l1920_192057

-- Part 1
theorem rationalized_factor (x : ℝ) : 
  (3 + Real.sqrt 11) * (3 - Real.sqrt 11) = -2 :=
sorry

-- Part 2
theorem simplify_fraction (b : ℝ) (h1 : b ≥ 0) (h2 : b ≠ 1) : 
  (1 - b) / (1 - Real.sqrt b) = 1 + Real.sqrt b :=
sorry

-- Part 3
theorem special_sqrt (a b : ℝ) 
  (ha : a = 1 / (Real.sqrt 3 - 2)) 
  (hb : b = 1 / (Real.sqrt 3 + 2)) : 
  Real.sqrt (a^2 + b^2 + 2) = 4 :=
sorry

end NUMINAMATH_CALUDE_rationalized_factor_simplify_fraction_special_sqrt_l1920_192057


namespace NUMINAMATH_CALUDE_binomial_theorem_and_sum_l1920_192005

def binomial_expansion (m : ℝ) : ℕ → ℝ
| 0 => 1
| 1 => 7 * m
| 2 => 21 * m^2
| 3 => 35 * m^3
| 4 => 35 * m^4
| 5 => 21 * m^5
| 6 => 7 * m^6
| 7 => m^7
| _ => 0

def a (m : ℝ) (i : ℕ) : ℝ := binomial_expansion m i

theorem binomial_theorem_and_sum (m : ℝ) :
  a m 3 = -280 →
  (m = -2 ∧ a m 1 + a m 3 + a m 5 + a m 7 = -1094) := by sorry

end NUMINAMATH_CALUDE_binomial_theorem_and_sum_l1920_192005


namespace NUMINAMATH_CALUDE_angle_property_l1920_192000

theorem angle_property (θ : Real) (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin θ - Real.cos θ = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_property_l1920_192000


namespace NUMINAMATH_CALUDE_coordinates_of_B_l1920_192055

-- Define the square OABC
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (4, 3)

-- Define the property that C is in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the property of being a square
def is_square (O A B C : ℝ × ℝ) : Prop :=
  let d₁ := (A.1 - O.1)^2 + (A.2 - O.2)^2
  let d₂ := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let d₃ := (C.1 - B.1)^2 + (C.2 - B.2)^2
  let d₄ := (O.1 - C.1)^2 + (O.2 - C.2)^2
  d₁ = d₂ ∧ d₂ = d₃ ∧ d₃ = d₄

-- Theorem statement
theorem coordinates_of_B :
  ∃ (B C : ℝ × ℝ), is_square O A B C ∧ in_fourth_quadrant C → B = (7, -1) :=
sorry

end NUMINAMATH_CALUDE_coordinates_of_B_l1920_192055


namespace NUMINAMATH_CALUDE_unique_monic_quadratic_l1920_192024

-- Define a monic polynomial of degree 2
def MonicQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, f x = x^2 + b*x + c

theorem unique_monic_quadratic (f : ℝ → ℝ) 
  (monic : MonicQuadratic f) 
  (eval_zero : f 0 = 4)
  (eval_one : f 1 = 10) :
  ∀ x, f x = x^2 + 5*x + 4 := by
sorry

end NUMINAMATH_CALUDE_unique_monic_quadratic_l1920_192024


namespace NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l1920_192020

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 5
def red_balls : ℕ := 6
def purple_balls : ℕ := 9

theorem probability_neither_red_nor_purple :
  (total_balls - (red_balls + purple_balls)) / total_balls = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_red_nor_purple_l1920_192020


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1920_192017

theorem modulus_of_complex_number (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = -2 * Real.cos (α/2) :=
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1920_192017


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1920_192048

/-- An isosceles triangle with given height and median -/
structure IsoscelesTriangle where
  -- Height from the base to the vertex
  height : ℝ
  -- Median from a leg to the midpoint of the base
  median : ℝ

/-- The area of an isosceles triangle given its height and median -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with height 18 and median 15 is 144 -/
theorem isosceles_triangle_area :
  let t : IsoscelesTriangle := { height := 18, median := 15 }
  area t = 144 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1920_192048


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l1920_192090

theorem exponential_equation_solution : 
  ∃ x : ℝ, (3 : ℝ)^x * 9^x = 27^(x - 20) ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l1920_192090


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1920_192081

theorem imaginary_part_of_product : Complex.im ((2 - Complex.I) * (4 + Complex.I)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1920_192081


namespace NUMINAMATH_CALUDE_angle_in_third_quadrant_l1920_192001

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α > 0) 
  (h2 : Real.sin α * Real.tan α < 0) : 
  angle_quadrant α = Quadrant.third :=
sorry

end NUMINAMATH_CALUDE_angle_in_third_quadrant_l1920_192001


namespace NUMINAMATH_CALUDE_smallest_congruent_difference_l1920_192079

/-- The smallest positive four-digit integer congruent to 7 (mod 13) -/
def p : ℕ := sorry

/-- The smallest positive five-digit integer congruent to 7 (mod 13) -/
def q : ℕ := sorry

theorem smallest_congruent_difference : q - p = 8996 := by sorry

end NUMINAMATH_CALUDE_smallest_congruent_difference_l1920_192079


namespace NUMINAMATH_CALUDE_simplified_expression_equals_negative_three_l1920_192086

theorem simplified_expression_equals_negative_three :
  let a : ℚ := -4
  (1 / (a - 1) + 1) / (a / (a^2 - 1)) = -3 := by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_negative_three_l1920_192086


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l1920_192096

theorem joel_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) : 
  joel_current_age = 5 → 
  dad_current_age = 32 → 
  ∃ (years : ℕ), 
    dad_current_age + years = 2 * (joel_current_age + years) ∧ 
    joel_current_age + years = 27 := by
sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l1920_192096


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1920_192070

/-- Given a car traveling for two hours with a speed of 98 km/h in the first hour
    and an average speed of 84 km/h over the two hours,
    prove that the speed of the car in the second hour is 70 km/h. -/
theorem car_speed_second_hour :
  let speed_first_hour : ℝ := 98
  let average_speed : ℝ := 84
  let total_time : ℝ := 2
  let speed_second_hour : ℝ := (average_speed * total_time) - speed_first_hour
  speed_second_hour = 70 := by
sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1920_192070


namespace NUMINAMATH_CALUDE_product_of_four_six_seven_fourteen_l1920_192009

theorem product_of_four_six_seven_fourteen : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_six_seven_fourteen_l1920_192009


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l1920_192065

theorem quadratic_form_k_value (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l1920_192065


namespace NUMINAMATH_CALUDE_trapezoid_angles_l1920_192061

-- Define a trapezoid
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  sum_360 : a + b + c + d = 360
  sum_ab_180 : a + b = 180
  sum_cd_180 : c + d = 180

-- Theorem statement
theorem trapezoid_angles (t : Trapezoid) (h1 : t.a = 60) (h2 : t.b = 130) :
  t.c = 50 ∧ t.d = 120 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_angles_l1920_192061


namespace NUMINAMATH_CALUDE_number_problem_l1920_192033

theorem number_problem (N : ℝ) : 
  (1/2 : ℝ) * ((3/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N) = 45 → 
  (65/100 : ℝ) * N = 585 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1920_192033


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1920_192078

-- Define the circles and their properties
structure CirclePair where
  r : ℝ  -- radius of the smaller circle
  R : ℝ  -- radius of the larger circle
  common_chord : ℝ  -- length of the common chord

-- Define the theorem
theorem circle_centers_distance (c : CirclePair) :
  (c.r > 0) →  -- ensure positive radius
  (c.common_chord = c.r * Real.sqrt 2) →  -- common chord is side of square in smaller circle
  (c.R = c.r * Real.sqrt 2) →  -- radius of larger circle
  (∃ d : ℝ, (d = (c.r * (Real.sqrt 6 + Real.sqrt 2)) / 2) ∨
            (d = (c.r * (Real.sqrt 6 - Real.sqrt 2)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l1920_192078


namespace NUMINAMATH_CALUDE_juice_bottles_theorem_l1920_192034

theorem juice_bottles_theorem (bottle_capacity : ℕ) (required_amount : ℕ) (min_bottles : ℕ) : 
  bottle_capacity = 15 →
  required_amount = 195 →
  min_bottles = 13 →
  (min_bottles * bottle_capacity ≥ required_amount ∧
   ∀ n : ℕ, n * bottle_capacity ≥ required_amount → n ≥ min_bottles) :=
by sorry

end NUMINAMATH_CALUDE_juice_bottles_theorem_l1920_192034


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1920_192052

/-- Trajectory M in the Cartesian plane -/
def trajectory_M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

/-- Line l in the Cartesian plane -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Intersection points of line l and trajectory M -/
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    x₁ ≠ x₂

/-- Angle condition for F₂P and F₂Q -/
def angle_condition (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory_M x₁ y₁ ∧ trajectory_M x₂ y₂ ∧
    line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
    (y₁ - 0) / (x₁ - 1) + (y₂ - 0) / (x₂ - 1) = 0

theorem line_passes_through_fixed_point :
  ∀ k m : ℝ,
    k ≠ 0 →
    intersection_points k m →
    angle_condition k m →
    ∃ x y : ℝ, x = 4 ∧ y = 0 ∧ line_l k m x y :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1920_192052


namespace NUMINAMATH_CALUDE_rain_probability_l1920_192085

theorem rain_probability (p_monday p_tuesday p_neither : ℝ) 
  (h1 : p_monday = 0.7)
  (h2 : p_tuesday = 0.55)
  (h3 : p_neither = 0.35)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1)
  (h5 : 0 ≤ p_tuesday ∧ p_tuesday ≤ 1)
  (h6 : 0 ≤ p_neither ∧ p_neither ≤ 1) :
  p_monday + p_tuesday - (1 - p_neither) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1920_192085


namespace NUMINAMATH_CALUDE_sum_equals_5186_l1920_192010

theorem sum_equals_5186 : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_5186_l1920_192010


namespace NUMINAMATH_CALUDE_flowchart_requirement_l1920_192014

-- Define the structure of a flowchart
structure Flowchart where
  boxes : Set (Operation)
  flowLines : Set (SequenceIndicator)

-- Define operations
inductive Operation
  | process : Operation
  | decision : Operation
  | inputOutput : Operation

-- Define sequence indicators
inductive SequenceIndicator
  | arrow : SequenceIndicator

-- Define the direction of flow
inductive FlowDirection
  | leftToRight : FlowDirection
  | topToBottom : FlowDirection

-- Define the general requirement for drawing a flowchart
def generalRequirement : (FlowDirection × FlowDirection) := (FlowDirection.leftToRight, FlowDirection.topToBottom)

-- Theorem: The general requirement for drawing a flowchart is from left to right, from top to bottom
theorem flowchart_requirement (f : Flowchart) : 
  generalRequirement = (FlowDirection.leftToRight, FlowDirection.topToBottom) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_requirement_l1920_192014


namespace NUMINAMATH_CALUDE_equation_solution_l1920_192068

theorem equation_solution (x : ℚ) : 
  (x + 10) / (x - 4) = (x - 4) / (x + 8) → x = -32 / 13 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1920_192068


namespace NUMINAMATH_CALUDE_power_function_sum_l1920_192025

/-- Given a power function f(x) = kx^α that passes through the point (1/2, √2),
    prove that k + α = 1/2 -/
theorem power_function_sum (k α : ℝ) (h : k * (1/2)^α = Real.sqrt 2) : k + α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l1920_192025


namespace NUMINAMATH_CALUDE_third_anthill_population_l1920_192087

/-- Calculates the number of ants in the next anthill given the current number of ants -/
def next_anthill_population (current_ants : ℕ) : ℕ :=
  (current_ants * 4) / 5

/-- Represents the forest with three anthills -/
structure Forest where
  anthill1 : ℕ
  anthill2 : ℕ
  anthill3 : ℕ

/-- Creates a forest with three anthills, where each subsequent anthill has 20% fewer ants -/
def create_forest (initial_ants : ℕ) : Forest :=
  let anthill2 := next_anthill_population initial_ants
  let anthill3 := next_anthill_population anthill2
  { anthill1 := initial_ants, anthill2 := anthill2, anthill3 := anthill3 }

/-- Theorem stating that in a forest with 100 ants in the first anthill, 
    the third anthill will have 64 ants -/
theorem third_anthill_population : 
  (create_forest 100).anthill3 = 64 := by sorry

end NUMINAMATH_CALUDE_third_anthill_population_l1920_192087


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1920_192012

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-2, -1, 0}

theorem intersection_A_complement_B (x : Int) : 
  x ∈ (A ∩ (U \ B)) ↔ x = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1920_192012


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l1920_192004

theorem extended_morse_code_symbols : 
  (Finset.range 5).sum (fun n => 2^(n+1)) = 62 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l1920_192004


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l1920_192043

theorem fixed_point_power_function (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →
  f 2 = Real.sqrt 2 / 2 →
  f 9 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l1920_192043


namespace NUMINAMATH_CALUDE_at_least_one_product_leq_one_l1920_192091

theorem at_least_one_product_leq_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_product_leq_one_l1920_192091


namespace NUMINAMATH_CALUDE_rosie_pies_from_27_apples_l1920_192021

/-- Represents the number of pies Rosie can make given a certain number of apples -/
def pies_from_apples (apples : ℕ) : ℚ :=
  (2 : ℚ) * apples / 9

theorem rosie_pies_from_27_apples :
  pies_from_apples 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_from_27_apples_l1920_192021


namespace NUMINAMATH_CALUDE_max_cables_for_given_network_l1920_192054

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brand_a : ℕ
  brand_b : ℕ

/-- Represents the number of cables in the network. -/
def cables (n : ComputerNetwork) : ℕ := sorry

/-- Checks if all computers in the network can communicate. -/
def all_communicate (n : ComputerNetwork) : Prop := sorry

/-- The maximum number of cables needed for full communication. -/
def max_cables (n : ComputerNetwork) : ℕ := sorry

/-- Theorem stating the maximum number of cables for the given network. -/
theorem max_cables_for_given_network :
  ∀ (n : ComputerNetwork),
    n.brand_a = 20 ∧ n.brand_b = 20 →
    max_cables n = 20 ∧ all_communicate n := by sorry

end NUMINAMATH_CALUDE_max_cables_for_given_network_l1920_192054


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l1920_192047

/-- The probability of rain on Saturday -/
def prob_rain_saturday : ℝ := 0.4

/-- The probability of rain on Sunday -/
def prob_rain_sunday : ℝ := 0.5

/-- The probabilities are independent -/
axiom independence : True

/-- The probability of rain on at least one day over the weekend -/
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

theorem weekend_rain_probability :
  prob_rain_weekend = 0.7 :=
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l1920_192047


namespace NUMINAMATH_CALUDE_snail_return_time_is_whole_hours_l1920_192038

/-- Represents the movement of a snail on a plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the time taken for the snail to return to its starting point -/
def returnTime (movement : SnailMovement) : ℝ := sorry

/-- Theorem stating that the return time is always an integer multiple of hours -/
theorem snail_return_time_is_whole_hours (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnInterval = 0.25) -- 15 minutes = 0.25 hours
  (h3 : movement.turnAngle = π / 2) -- right angle
  : ∃ n : ℕ, returnTime movement = n := by sorry

end NUMINAMATH_CALUDE_snail_return_time_is_whole_hours_l1920_192038


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1920_192074

/-- A Mersenne number is of the form 2^n - 1 --/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A Mersenne prime is a Mersenne number that is prime --/
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, Prime n ∧ p = mersenne_number n ∧ Prime p

/-- The largest Mersenne prime less than 500 is 127 --/
theorem largest_mersenne_prime_under_500 :
  (∀ p, is_mersenne_prime p → p < 500 → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l1920_192074


namespace NUMINAMATH_CALUDE_prime_9k_plus_1_divides_cubic_l1920_192060

theorem prime_9k_plus_1_divides_cubic (p : Nat) (k : Nat) (h_prime : Nat.Prime p) (h_form : p = 9*k + 1) :
  ∃ n : ℤ, (n^3 - 3*n + 1) % p = 0 := by sorry

end NUMINAMATH_CALUDE_prime_9k_plus_1_divides_cubic_l1920_192060


namespace NUMINAMATH_CALUDE_minimum_sugar_amount_l1920_192040

theorem minimum_sugar_amount (f s : ℝ) : 
  (f ≥ 8 + (3 * s) / 4) → 
  (f ≤ 2 * s) → 
  s ≥ 32 / 5 :=
by
  sorry

#eval (32 : ℚ) / 5  -- To show that 32/5 = 6.4

end NUMINAMATH_CALUDE_minimum_sugar_amount_l1920_192040
