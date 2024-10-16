import Mathlib

namespace NUMINAMATH_CALUDE_f_inverse_property_implies_c_plus_d_eq_nine_halves_l2355_235504

-- Define the piecewise function f
noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

-- State the theorem
theorem f_inverse_property_implies_c_plus_d_eq_nine_halves
  (c d : ℝ)
  (h : ∀ x, f c d (f c d x) = x) :
  c + d = 9/2 := by
sorry

end NUMINAMATH_CALUDE_f_inverse_property_implies_c_plus_d_eq_nine_halves_l2355_235504


namespace NUMINAMATH_CALUDE_factorialLastNonzeroDigitSeq_not_periodic_l2355_235547

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10

/-- The sequence of last nonzero digits of factorials -/
def factorialLastNonzeroDigitSeq : ℕ → ℕ :=
  fun n => lastNonzeroDigit (Nat.factorial n)

/-- The sequence of last nonzero digits of factorials is not periodic -/
theorem factorialLastNonzeroDigitSeq_not_periodic :
  ¬ ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), factorialLastNonzeroDigitSeq (n + p) = factorialLastNonzeroDigitSeq n :=
sorry

end NUMINAMATH_CALUDE_factorialLastNonzeroDigitSeq_not_periodic_l2355_235547


namespace NUMINAMATH_CALUDE_initial_distance_proof_l2355_235578

/-- The initial distance between Tim and Élan -/
def initial_distance : ℝ := 30

/-- Tim's initial speed in mph -/
def tim_speed : ℝ := 10

/-- Élan's initial speed in mph -/
def elan_speed : ℝ := 5

/-- The distance Tim travels until meeting Élan -/
def tim_distance : ℝ := 20

/-- The time it takes for Tim and Élan to meet -/
def meeting_time : ℝ := 1.5

theorem initial_distance_proof :
  initial_distance = 
    tim_speed * 1 + 
    elan_speed * 1 + 
    (tim_speed * 2) * (meeting_time - 1) + 
    (elan_speed * 2) * (meeting_time - 1) :=
sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l2355_235578


namespace NUMINAMATH_CALUDE_agnes_hourly_rate_l2355_235573

/-- Proves that Agnes's hourly rate is $15 given the conditions of the problem -/
theorem agnes_hourly_rate : 
  ∀ (mila_rate : ℝ) (agnes_weekly_hours : ℝ) (mila_equal_hours : ℝ) (weeks_per_month : ℝ),
  mila_rate = 10 →
  agnes_weekly_hours = 8 →
  mila_equal_hours = 48 →
  weeks_per_month = 4 →
  (agnes_weekly_hours * weeks_per_month * (mila_rate * mila_equal_hours / (agnes_weekly_hours * weeks_per_month))) = 15 :=
by sorry

end NUMINAMATH_CALUDE_agnes_hourly_rate_l2355_235573


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2355_235525

theorem binomial_expansion_coefficient (n : ℕ) : 
  (8 * (Nat.choose n 3) * 2^3 = 16 * n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2355_235525


namespace NUMINAMATH_CALUDE_tangent_inequality_tan_pi_12_l2355_235568

theorem tangent_inequality (a : Fin 13 → ℝ) (h : ∀ i j, i ≠ j → a i ≠ a j) : 
  ∃ i j, i ≠ j ∧ 0 < (a i - a j) / (1 + a i * a j) ∧ 
    (a i - a j) / (1 + a i * a j) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

theorem tan_pi_12 : Real.tan (π / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_inequality_tan_pi_12_l2355_235568


namespace NUMINAMATH_CALUDE_george_final_stickers_l2355_235539

/-- The number of stickers each person has --/
structure Stickers where
  bob : ℕ
  tom : ℕ
  dan : ℕ
  george : ℕ

/-- The conditions of the problem --/
def sticker_conditions (s : Stickers) : Prop :=
  s.dan = 2 * s.tom ∧
  s.tom = 3 * s.bob ∧
  s.george = 5 * s.dan ∧
  s.bob = 12

/-- The total number of stickers to be distributed --/
def extra_stickers : ℕ := 100

/-- The number of people --/
def num_people : ℕ := 4

/-- Theorem stating that George will have 505 stickers in total --/
theorem george_final_stickers (s : Stickers) 
  (h : sticker_conditions s) : 
  s.george + (s.bob + s.tom + s.dan + s.george + extra_stickers) / num_people = 505 := by
  sorry


end NUMINAMATH_CALUDE_george_final_stickers_l2355_235539


namespace NUMINAMATH_CALUDE_polyhedron_sum_l2355_235534

/-- Represents a convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ -- number of vertices
  E : ℕ -- number of edges
  F : ℕ -- number of faces
  q : ℕ -- number of quadrilateral faces
  h : ℕ -- number of hexagonal faces
  Q : ℕ -- number of quadrilateral faces meeting at each vertex
  H : ℕ -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 24
  face_types : q + h = F
  edge_count : E = 2*q + 3*h
  vertex_degree : Q = 1 ∧ H = 1

/-- The main theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.Q + p.V = 136 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l2355_235534


namespace NUMINAMATH_CALUDE_car_discount_proof_l2355_235527

/-- Proves that the initial discount on a car's original price was 30%, given specific selling conditions --/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Original price is positive
  0 ≤ D ∧ D < 1 →  -- Discount is between 0 and 1 (exclusive)
  P * (1 - D) * 1.7 = P * 1.18999999999999993 →  -- Selling price equation
  D = 0.3 := by
sorry

end NUMINAMATH_CALUDE_car_discount_proof_l2355_235527


namespace NUMINAMATH_CALUDE_lighter_person_weight_l2355_235542

/-- Given two people with a total weight of 88 kg, where one person is 4 kg heavier than the other,
    prove that the weight of the lighter person is 42 kg. -/
theorem lighter_person_weight (total_weight : ℝ) (weight_difference : ℝ) (lighter_weight : ℝ) : 
  total_weight = 88 → weight_difference = 4 → 
  lighter_weight + (lighter_weight + weight_difference) = total_weight →
  lighter_weight = 42 := by
  sorry

#check lighter_person_weight

end NUMINAMATH_CALUDE_lighter_person_weight_l2355_235542


namespace NUMINAMATH_CALUDE_acute_angle_condition_x_plus_y_value_l2355_235526

-- Define the vectors
def a : Fin 2 → ℝ := ![2, -1]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Theorem 1: Acute angle condition
theorem acute_angle_condition (x : ℝ) :
  (dot_product a (b x) > 0) ↔ (x > 1/2) := by sorry

-- Theorem 2: Value of x + y
theorem x_plus_y_value (x y : ℝ) :
  (3 • a - 2 • (b x) = ![4, y]) → x + y = -4 := by sorry

end NUMINAMATH_CALUDE_acute_angle_condition_x_plus_y_value_l2355_235526


namespace NUMINAMATH_CALUDE_tangent_circles_distance_l2355_235507

/-- The distance between the centers of two tangent circles with radii 1 and 7 is either 6 or 8. -/
theorem tangent_circles_distance (r₁ r₂ d : ℝ) : 
  r₁ = 1 → r₂ = 7 → (d = |r₁ - r₂| ∨ d = r₁ + r₂) → d = 6 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_distance_l2355_235507


namespace NUMINAMATH_CALUDE_jeans_business_weekly_hours_l2355_235549

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of Jean's business for a week -/
structure WeeklyHours where
  weekday_hours : DailyHours
  weekend_hours : DailyHours

/-- Calculates the total number of hours Jean's business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday_hours * 5) + (hours_open wh.weekend_hours * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday_hours := { open_time := 16, close_time := 22 }
  , weekend_hours := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end NUMINAMATH_CALUDE_jeans_business_weekly_hours_l2355_235549


namespace NUMINAMATH_CALUDE_power_division_equality_l2355_235553

theorem power_division_equality : (3 : ℕ)^12 / ((27 : ℕ)^2 * 3^3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2355_235553


namespace NUMINAMATH_CALUDE_golden_section_division_l2355_235538

/-- Given a line segment AB of length a, prove that the point H that divides AB
    such that AH = a(√5 - 1)/2 makes AH the mean proportional between AB and HB. -/
theorem golden_section_division (a : ℝ) (h : a > 0) :
  let x := a * (Real.sqrt 5 - 1) / 2
  x * x = a * (a - x) :=
by sorry

end NUMINAMATH_CALUDE_golden_section_division_l2355_235538


namespace NUMINAMATH_CALUDE_queue_arrangements_l2355_235596

/-- Represents the number of people in each category -/
def num_fathers : ℕ := 2
def num_mothers : ℕ := 2
def num_children : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := num_fathers + num_mothers + num_children

/-- Represents the constraint that fathers must be at the beginning and end -/
def fathers_fixed : ℕ := 2

/-- Represents the number of units to arrange between fathers (2 mothers and 1 children unit) -/
def units_between : ℕ := num_mothers + 1

/-- Represents the number of ways to arrange children within their unit -/
def children_arrangements : ℕ := Nat.factorial num_children

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := fathers_fixed * Nat.factorial units_between * children_arrangements

/-- Theorem stating that the number of possible arrangements is 24 -/
theorem queue_arrangements : total_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_queue_arrangements_l2355_235596


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2355_235583

theorem binomial_coefficient_equality (n s : ℕ) (h : s > 0) :
  (n.choose s) = (n * (n - 1).choose (s - 1)) / s :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2355_235583


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2355_235509

/-- Given vectors a and b in ℝ², prove that the magnitude of 2a + b equals 4. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (ha : a = (3, -3)) (hb : b = (-2, 6)) : 
  ‖(2 : ℝ) • a + b‖ = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2355_235509


namespace NUMINAMATH_CALUDE_moving_circle_theorem_l2355_235537

-- Define the circle and its properties
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_point : center.1^2 + center.2^2 = (center.1 - 1)^2 + center.2^2
  tangent_to_line : (center.1 + 1)^2 = center.1^2 + center.2^2

-- Define the trajectory
def trajectory (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for two points on the trajectory
def trajectory_points_condition (A B : ℝ × ℝ) : Prop :=
  trajectory A ∧ trajectory B ∧ A ≠ B ∧
  (A.2 / A.1) * (B.2 / B.1) = 1

-- The main theorem
theorem moving_circle_theorem (C : MovingCircle) :
  (∀ p : ℝ × ℝ, p = C.center → trajectory p) ∧
  (∀ A B : ℝ × ℝ, trajectory_points_condition A B →
    ∃ k : ℝ, B.2 - A.2 = k * (B.1 - A.1) ∧ A.2 = k * (A.1 + 4)) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_theorem_l2355_235537


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2355_235513

def num_letters : Nat := 26
def num_letter_positions : Nat := 4
def num_digit_positions : Nat := 3

def license_plate_combinations : Nat :=
  Nat.choose num_letters 2 *
  Nat.choose num_letter_positions 2 *
  2 *
  (10 * 9 * 8)

theorem license_plate_theorem :
  license_plate_combinations = 2808000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2355_235513


namespace NUMINAMATH_CALUDE_multiplication_mistake_l2355_235506

theorem multiplication_mistake (x : ℕ) (h : 53 * x - 35 * x = 540) : 53 * x = 1590 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l2355_235506


namespace NUMINAMATH_CALUDE_projection_matrix_condition_l2355_235579

/-- A projection matrix is idempotent (P^2 = P) -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49], ![c, 29/49]]

/-- The theorem stating the conditions for the given matrix to be a projection matrix -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 1 ∧ c = 0 := by
  sorry

#check projection_matrix_condition

end NUMINAMATH_CALUDE_projection_matrix_condition_l2355_235579


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2355_235581

theorem right_triangle_third_side 
  (a b c : ℝ) 
  (ha : a = 10) 
  (hb : b = 24) 
  (hright : a^2 + c^2 = b^2) : 
  c = 2 * Real.sqrt 119 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2355_235581


namespace NUMINAMATH_CALUDE_bounds_of_abs_diff_over_sum_l2355_235510

theorem bounds_of_abs_diff_over_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z, z = |x - y| / (|x| + |y|) → m ≤ z ∧ z ≤ M) ∧
    m = 0 ∧ M = 1 ∧ M - m = 1 :=
by sorry

end NUMINAMATH_CALUDE_bounds_of_abs_diff_over_sum_l2355_235510


namespace NUMINAMATH_CALUDE_other_coin_denomination_l2355_235550

/-- Given the following conditions:
    - There are 344 coins in total
    - The total value of all coins is 7100 paise (Rs. 71)
    - There are 300 coins of 20 paise each
    - There are two types of coins: 20 paise and another unknown denomination
    Prove that the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 344)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 300)
  : (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l2355_235550


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_of_4680_l2355_235570

theorem smaller_two_digit_factor_of_4680 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4680 → min a b = 52 := by
  sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_of_4680_l2355_235570


namespace NUMINAMATH_CALUDE_circus_revenue_l2355_235564

/-- Calculates the total revenue from circus ticket sales -/
theorem circus_revenue (lower_price upper_price : ℕ) (total_tickets lower_tickets : ℕ) :
  lower_price = 30 →
  upper_price = 20 →
  total_tickets = 80 →
  lower_tickets = 50 →
  lower_price * lower_tickets + upper_price * (total_tickets - lower_tickets) = 2100 := by
sorry

end NUMINAMATH_CALUDE_circus_revenue_l2355_235564


namespace NUMINAMATH_CALUDE_store_pricing_l2355_235500

-- Define variables for the prices of individual items
variable (p n e : ℝ)

-- Define the equations based on the given conditions
def equation1 : Prop := 10 * p + 12 * n + 6 * e = 5.50
def equation2 : Prop := 6 * p + 4 * n + 3 * e = 2.40

-- Define the final cost calculation
def final_cost : ℝ := 20 * p + 15 * n + 9 * e

-- Theorem statement
theorem store_pricing (h1 : equation1 p n e) (h2 : equation2 p n e) : 
  final_cost p n e = 8.95 := by
  sorry


end NUMINAMATH_CALUDE_store_pricing_l2355_235500


namespace NUMINAMATH_CALUDE_exists_sequence_iff_N_ge_4_l2355_235586

/-- A sequence of positive integers -/
def PositiveIntegerSequence := ℕ+ → ℕ+

/-- Strictly increasing sequence -/
def StrictlyIncreasing (s : PositiveIntegerSequence) : Prop :=
  ∀ n m : ℕ+, n < m → s n < s m

/-- The property that the sequence satisfies for a given N -/
def SatisfiesProperty (s : PositiveIntegerSequence) (N : ℝ) : Prop :=
  ∀ n : ℕ+, (s (2 * n - 1) + s (2 * n)) / s n = N

/-- The main theorem -/
theorem exists_sequence_iff_N_ge_4 (N : ℝ) : 
  (∃ s : PositiveIntegerSequence, StrictlyIncreasing s ∧ SatisfiesProperty s N) ↔ N ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_sequence_iff_N_ge_4_l2355_235586


namespace NUMINAMATH_CALUDE_equal_product_sequence_characterization_l2355_235597

def is_equal_product_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k

theorem equal_product_sequence_characterization (a : ℕ → ℝ) :
  is_equal_product_sequence a ↔
    ∃ k : ℝ, ∀ n : ℕ, n ≥ 2 → a n * a (n - 1) = k :=
by sorry

end NUMINAMATH_CALUDE_equal_product_sequence_characterization_l2355_235597


namespace NUMINAMATH_CALUDE_probability_at_least_three_white_balls_l2355_235557

theorem probability_at_least_three_white_balls 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (drawn_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 8)
  (h3 : black_balls = 7)
  (h4 : drawn_balls = 5) :
  let favorable_outcomes := Nat.choose white_balls 3 * Nat.choose black_balls 2 +
                            Nat.choose white_balls 4 * Nat.choose black_balls 1 +
                            Nat.choose white_balls 5 * Nat.choose black_balls 0
  let total_outcomes := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 1722 / 3003 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_three_white_balls_l2355_235557


namespace NUMINAMATH_CALUDE_min_value_expression_l2355_235595

theorem min_value_expression :
  ∀ x y : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≥ -19 ∧
  ∃ x₀ y₀ : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x₀))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x₀ + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y₀ - Real.cos (2 * y₀)) = -19 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2355_235595


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2355_235503

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2355_235503


namespace NUMINAMATH_CALUDE_nine_chapters_equal_distribution_l2355_235565

theorem nine_chapters_equal_distribution :
  ∀ (a : ℚ) (d : ℚ),
    (5 * a + 10 * d = 5) →  -- Sum of 5 terms is 5
    (2 * a + d = 3 * a + 9 * d) →  -- Sum of first two terms equals sum of last three terms
    a = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_nine_chapters_equal_distribution_l2355_235565


namespace NUMINAMATH_CALUDE_no_solution_exists_l2355_235545

theorem no_solution_exists : ¬∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧
  ∀ (n : ℕ), n > 0 → ((n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2355_235545


namespace NUMINAMATH_CALUDE_unique_sequence_solution_l2355_235575

/-- Represents a solution to the sequence problem -/
structure SequenceSolution where
  n : ℕ
  q : ℚ
  d : ℚ

/-- Checks if a given solution satisfies all conditions of the problem -/
def is_valid_solution (sol : SequenceSolution) : Prop :=
  sol.n > 1 ∧
  1 + (sol.n - 1) * sol.d = 81 ∧
  1 * sol.q^(sol.n - 1) = 81 ∧
  sol.q / sol.d = 0.15

/-- The unique solution to the sequence problem -/
def unique_solution : SequenceSolution :=
  { n := 5, q := 3, d := 20 }

/-- Theorem stating that the unique_solution is the only valid solution -/
theorem unique_sequence_solution :
  is_valid_solution unique_solution ∧
  ∀ (sol : SequenceSolution), is_valid_solution sol → sol = unique_solution :=
sorry

end NUMINAMATH_CALUDE_unique_sequence_solution_l2355_235575


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l2355_235524

theorem smallest_addition_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n + x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n + y) % d ≠ 0 :=
by sorry

theorem smallest_addition_for_27452_div_9 :
  ∃ (x : ℕ), x < 9 ∧ (27452 + x) % 9 = 0 ∧ ∀ (y : ℕ), y < x → (27452 + y) % 9 ≠ 0 :=
by
  apply smallest_addition_for_divisibility 27452 9
  norm_num

theorem smallest_addition_is_7 :
  7 < 9 ∧ (27452 + 7) % 9 = 0 ∧ ∀ (y : ℕ), y < 7 → (27452 + y) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_smallest_addition_for_27452_div_9_smallest_addition_is_7_l2355_235524


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2355_235554

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  ∀ x : ℝ, x^3 - 2*x^2 - 5*x + 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2355_235554


namespace NUMINAMATH_CALUDE_no_solution_exists_l2355_235535

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the properties of the function f
def StrictlyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem no_solution_exists (f : ℝ → ℝ) 
  (h1 : StrictlyDecreasing f)
  (h2 : ∀ x ∈ PositiveReals, f x * f (f x + 3 / (2 * x)) = 1/4) :
  ¬ ∃ x ∈ PositiveReals, f x + 3 * x = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2355_235535


namespace NUMINAMATH_CALUDE_abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l2355_235541

theorem abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely :
  ∃ (a b : ℝ), (abs b + a < 0 → b^2 < a^2) ∧
  ¬(∀ (a b : ℝ), b^2 < a^2 → abs b + a < 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_plus_a_neg_implies_b_sq_lt_a_sq_but_not_conversely_l2355_235541


namespace NUMINAMATH_CALUDE_prime_difference_divisibility_l2355_235560

theorem prime_difference_divisibility 
  (p₁ p₂ p₃ p₄ q₁ q₂ q₃ q₄ : ℕ) 
  (hp : p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄)
  (hq : q₁ < q₂ ∧ q₂ < q₃ ∧ q₃ < q₄)
  (hp_prime : Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄)
  (hq_prime : Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄)
  (hp_diff : p₄ - p₁ = 8)
  (hq_diff : q₄ - q₁ = 8)
  (hp_gt_5 : p₁ > 5)
  (hq_gt_5 : q₁ > 5) :
  30 ∣ (p₁ - q₁) := by
sorry

end NUMINAMATH_CALUDE_prime_difference_divisibility_l2355_235560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l2355_235502

theorem arithmetic_sequence_sum_mod_15 (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 2 →
  d = 5 →
  aₙ = 102 →
  n * (a₁ + aₙ) / 2 ≡ 12 [MOD 15] :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l2355_235502


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l2355_235518

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x => 9 * x^2 - 45 * x + 50
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l2355_235518


namespace NUMINAMATH_CALUDE_medical_supply_transport_l2355_235516

/-- Given two locations A and B that are 360 kilometers apart, a truck carrying 6 boxes of medical supplies
    traveling from A to B at 40 km/h, and a motorcycle departing from B towards the truck at 80 km/h,
    this theorem proves that the total time needed to transport all 6 boxes to location B is 26/3 hours
    and the total distance traveled by the motorcycle is 2080/3 kilometers. -/
theorem medical_supply_transport (distance_AB : ℝ) (truck_speed : ℝ) (motorcycle_speed : ℝ) 
  (boxes : ℕ) (boxes_per_trip : ℕ) :
  distance_AB = 360 →
  truck_speed = 40 →
  motorcycle_speed = 80 →
  boxes = 6 →
  boxes_per_trip = 2 →
  ∃ (total_time : ℝ) (total_distance : ℝ),
    total_time = 26/3 ∧
    total_distance = 2080/3 :=
by sorry


end NUMINAMATH_CALUDE_medical_supply_transport_l2355_235516


namespace NUMINAMATH_CALUDE_extra_boxes_calculation_l2355_235561

/-- Proves that given an order of 3 dozen boxes with extra free boxes equivalent to a 25% discount, the number of extra boxes received is 9 -/
theorem extra_boxes_calculation (dozen : ℕ) (order_size : ℕ) (discount_percent : ℚ) : 
  dozen = 12 →
  order_size = 3 →
  discount_percent = 25 / 100 →
  (dozen * order_size : ℚ) * (1 - discount_percent) = dozen * order_size - 9 :=
by sorry

end NUMINAMATH_CALUDE_extra_boxes_calculation_l2355_235561


namespace NUMINAMATH_CALUDE_five_digit_sum_l2355_235517

def is_valid_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

def sum_of_digits (x : ℕ) : ℕ := 120 * (1 + 3 + 4 + 6 + x)

theorem five_digit_sum (x : ℕ) (h1 : is_valid_digit x) (h2 : sum_of_digits x = 2640) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l2355_235517


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2355_235536

theorem solve_linear_equation (x : ℚ) (h : x + 2*x + 3*x + 4*x = 5) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2355_235536


namespace NUMINAMATH_CALUDE_grandfathers_age_is_79_l2355_235571

/-- The age of Caleb's grandfather based on the number of candles on the cake -/
def grandfathers_age (yellow_candles red_candles blue_candles : ℕ) : ℕ :=
  yellow_candles + red_candles + blue_candles

/-- Theorem stating that Caleb's grandfather's age is 79 given the number of candles -/
theorem grandfathers_age_is_79 :
  grandfathers_age 27 14 38 = 79 := by
  sorry

end NUMINAMATH_CALUDE_grandfathers_age_is_79_l2355_235571


namespace NUMINAMATH_CALUDE_jason_gave_nine_cards_l2355_235574

/-- The number of Pokemon cards Jason gave to his friends -/
def cards_given_to_friends (initial_cards : ℕ) (remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem: Jason gave 9 Pokemon cards to his friends -/
theorem jason_gave_nine_cards : cards_given_to_friends 13 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jason_gave_nine_cards_l2355_235574


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2355_235523

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 1 ∧ x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2 ∧
    2*y₁^2 - 2*y₁ - 1 = 0 ∧ 2*y₂^2 - 2*y₂ - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2355_235523


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l2355_235567

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (hf : QuadraticFunction f)
  (h1 : f 0 = 1)
  (h2 : ∀ x, f (x + 1) - f x = 2 * x) :
  ∀ x, f x = x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l2355_235567


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2355_235556

theorem repeating_decimal_sum (c d : ℕ) : 
  (c < 10 ∧ d < 10) →  -- Ensuring c and d are single digits
  (5 : ℚ) / 13 = (c * 10 + d : ℚ) / 99 →  -- Representing the repeating decimal
  c + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2355_235556


namespace NUMINAMATH_CALUDE_cold_drink_recipe_l2355_235520

theorem cold_drink_recipe (tea_per_drink : ℚ) (total_mixture : ℚ) (total_lemonade : ℚ)
  (h1 : tea_per_drink = 1/4)
  (h2 : total_mixture = 18)
  (h3 : total_lemonade = 15) :
  (total_lemonade / ((total_mixture - total_lemonade) / tea_per_drink)) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cold_drink_recipe_l2355_235520


namespace NUMINAMATH_CALUDE_angle_CBO_is_20_l2355_235585

-- Define the triangle ABC and point O
variable (A B C O : Point)

-- Define the angles as real numbers (in degrees)
variable (angle_BAO angle_CAO angle_CBO angle_ABO angle_ACO angle_BCO angle_AOC : ℝ)

-- State the theorem
theorem angle_CBO_is_20 
  (h1 : angle_BAO = angle_CAO)
  (h2 : angle_CBO = angle_ABO)
  (h3 : angle_ACO = angle_BCO)
  (h4 : angle_AOC = 110) :
  angle_CBO = 20 := by
    sorry

end NUMINAMATH_CALUDE_angle_CBO_is_20_l2355_235585


namespace NUMINAMATH_CALUDE_volume_removed_percent_l2355_235558

def box_length : ℝ := 15
def box_width : ℝ := 10
def box_height : ℝ := 8
def cube_side : ℝ := 3
def num_corners : ℕ := 8

def box_volume : ℝ := box_length * box_width * box_height
def removed_cube_volume : ℝ := cube_side ^ 3
def total_removed_volume : ℝ := num_corners * removed_cube_volume

theorem volume_removed_percent :
  (total_removed_volume / box_volume) * 100 = 18 := by sorry

end NUMINAMATH_CALUDE_volume_removed_percent_l2355_235558


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l2355_235532

/-- The number of bottle caps Marilyn shared -/
def shared_caps : ℕ := 36

/-- The number of bottle caps Marilyn ended up with -/
def remaining_caps : ℕ := 15

/-- The initial number of bottle caps Marilyn had -/
def initial_caps : ℕ := shared_caps + remaining_caps

theorem marilyn_bottle_caps : initial_caps = 51 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l2355_235532


namespace NUMINAMATH_CALUDE_initial_fish_count_l2355_235508

def days_in_three_weeks : ℕ := 21

def koi_added_per_day : ℕ := 2
def goldfish_added_per_day : ℕ := 5

def final_koi_count : ℕ := 227
def final_goldfish_count : ℕ := 200

def total_koi_added : ℕ := days_in_three_weeks * koi_added_per_day
def total_goldfish_added : ℕ := days_in_three_weeks * goldfish_added_per_day

def initial_koi_count : ℕ := final_koi_count - total_koi_added
def initial_goldfish_count : ℕ := final_goldfish_count - total_goldfish_added

theorem initial_fish_count :
  initial_koi_count + initial_goldfish_count = 280 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l2355_235508


namespace NUMINAMATH_CALUDE_conic_section_k_range_l2355_235555

/-- Represents a conic section of the form x^2/2 + y^2/k = 1 -/
structure ConicSection (k : ℝ) where
  equation : ∀ (x y : ℝ), x^2/2 + y^2/k = 1

/-- Proposition p: The conic section is an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- Proposition q: The eccentricity of the conic section is in the interval (√2, √3) -/
def eccentricity_in_range (k : ℝ) : Prop :=
  let e := Real.sqrt ((2 - k) / 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 3

/-- The main theorem -/
theorem conic_section_k_range (k : ℝ) (E : ConicSection k) :
  (¬is_ellipse_x_foci k) ∧ eccentricity_in_range k → -4 < k ∧ k < -2 :=
sorry

end NUMINAMATH_CALUDE_conic_section_k_range_l2355_235555


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2355_235552

theorem proposition_equivalence (x : ℝ) :
  (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1) ↔ (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2355_235552


namespace NUMINAMATH_CALUDE_small_boxes_count_l2355_235593

theorem small_boxes_count (chocolate_per_small_box : ℕ) (total_chocolate : ℕ) : 
  chocolate_per_small_box = 25 → total_chocolate = 475 → 
  total_chocolate / chocolate_per_small_box = 19 := by
  sorry

end NUMINAMATH_CALUDE_small_boxes_count_l2355_235593


namespace NUMINAMATH_CALUDE_certain_amount_proof_l2355_235589

/-- The interest rate per annum -/
def interest_rate : ℚ := 8 / 100

/-- The time period for the first amount in years -/
def time1 : ℚ := 25 / 2

/-- The time period for the second amount in years -/
def time2 : ℚ := 4

/-- The first principal amount in Rs -/
def principal1 : ℚ := 160

/-- The second principal amount (the certain amount) in Rs -/
def principal2 : ℚ := 500

/-- Simple interest formula -/
def simple_interest (p r t : ℚ) : ℚ := p * r * t

theorem certain_amount_proof :
  simple_interest principal1 interest_rate time1 = simple_interest principal2 interest_rate time2 :=
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l2355_235589


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2355_235531

theorem sum_of_roots_equation (x : ℝ) :
  (x ≠ 3 ∧ x ≠ -3) →
  ((-6 * x) / (x^2 - 9) = (3 * x) / (x + 3) - 2 / (x - 3) + 1) →
  ∃ (y : ℝ), (y ≠ 3 ∧ y ≠ -3) ∧
             ((-6 * y) / (y^2 - 9) = (3 * y) / (y + 3) - 2 / (y - 3) + 1) ∧
             x + y = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2355_235531


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l2355_235546

theorem line_through_parabola_vertex :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b : ℝ, b ∈ s ↔ 
    (∃ x y : ℝ, y = 2*x + b ∧ 
               y = x^2 + b^2 - 1 ∧ 
               ∀ x' : ℝ, x'^2 + b^2 - 1 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l2355_235546


namespace NUMINAMATH_CALUDE_negative_two_a_cubed_l2355_235588

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_a_cubed_l2355_235588


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2355_235563

theorem shirt_price_proof (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 105)
  (h2 : discount1 = 19.954259576901087)
  (h3 : discount2 = 12.55) :
  ∃ (list_price : ℝ), 
    list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price ∧ 
    list_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2355_235563


namespace NUMINAMATH_CALUDE_function_difference_bound_l2355_235544

/-- Given a function f(x) = x^2 - x + c and a real number a such that |x - a| < 1,
    prove that |f(x) - f(a)| < 2(|a| + 1) -/
theorem function_difference_bound (c a x : ℝ) (h : |x - a| < 1) :
  let f := fun (t : ℝ) => t^2 - t + c
  |f x - f a| < 2 * (|a| + 1) := by
sorry


end NUMINAMATH_CALUDE_function_difference_bound_l2355_235544


namespace NUMINAMATH_CALUDE_cos_product_eighth_and_five_eighths_pi_l2355_235577

theorem cos_product_eighth_and_five_eighths_pi :
  Real.cos (π / 8) * Real.cos (5 * π / 8) = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_product_eighth_and_five_eighths_pi_l2355_235577


namespace NUMINAMATH_CALUDE_linear_inequality_solution_l2355_235590

theorem linear_inequality_solution (x : ℝ) : 3 * (x + 1) > 9 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_l2355_235590


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2355_235572

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 3) :
  (1/x + 1/y : ℝ) ≥ 1 + 2*Real.sqrt 2/3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2355_235572


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l2355_235566

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ 72542 ≡ n [ZMOD 25] ∧ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l2355_235566


namespace NUMINAMATH_CALUDE_fair_spending_theorem_l2355_235551

/-- Calculates the remaining amount after spending at the fair -/
def remaining_amount (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ) : ℕ :=
  initial - (snacks + rides_multiplier * snacks + games)

/-- Theorem stating that the remaining amount is 10 dollars -/
theorem fair_spending_theorem (initial : ℕ) (snacks : ℕ) (rides_multiplier : ℕ) (games : ℕ)
  (h1 : initial = 80)
  (h2 : snacks = 15)
  (h3 : rides_multiplier = 3)
  (h4 : games = 10) :
  remaining_amount initial snacks rides_multiplier games = 10 := by
  sorry

end NUMINAMATH_CALUDE_fair_spending_theorem_l2355_235551


namespace NUMINAMATH_CALUDE_sports_club_total_members_l2355_235528

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 42 members -/
theorem sports_club_total_members :
  sports_club_members 20 23 7 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_total_members_l2355_235528


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2355_235599

/-- The stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 50 1.5 0.3 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l2355_235599


namespace NUMINAMATH_CALUDE_sector_area_l2355_235529

theorem sector_area (α : Real) (perimeter : Real) (h1 : α = 1/3) (h2 : perimeter = 7) :
  let r := perimeter / (2 + α)
  (1/2) * α * r^2 = 3/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2355_235529


namespace NUMINAMATH_CALUDE_modulus_of_z_l2355_235521

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2355_235521


namespace NUMINAMATH_CALUDE_harold_catch_up_distance_l2355_235580

/-- The distance from X to Y in miles -/
def total_distance : ℝ := 60

/-- Adrienne's walking speed in miles per hour -/
def adrienne_speed : ℝ := 3

/-- Harold's walking speed in miles per hour -/
def harold_speed : ℝ := adrienne_speed + 1

/-- Time difference between Adrienne's and Harold's start in hours -/
def time_difference : ℝ := 1

/-- The distance Harold will have traveled when he catches up to Adrienne -/
def catch_up_distance : ℝ := 12

theorem harold_catch_up_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  adrienne_speed * (t + time_difference) = harold_speed * t ∧
  catch_up_distance = harold_speed * t :=
by sorry

end NUMINAMATH_CALUDE_harold_catch_up_distance_l2355_235580


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2355_235584

theorem quadratic_equation_solution : 
  {x : ℝ | 2 * x^2 + 5 * x = 0} = {0, -5/2} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2355_235584


namespace NUMINAMATH_CALUDE_least_whole_number_ratio_l2355_235533

theorem least_whole_number_ratio (x : ℕ) : x ≥ 3 ↔ (6 - x : ℚ) / (7 - x) < 16 / 21 :=
sorry

end NUMINAMATH_CALUDE_least_whole_number_ratio_l2355_235533


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_today_l2355_235511

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final total number of trees -/
def final_total_trees : ℕ := 100

theorem dogwood_trees_planted_today :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_total_trees :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_today_l2355_235511


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_3_l2355_235543

theorem smallest_four_digit_mod_8_3 : ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_3_l2355_235543


namespace NUMINAMATH_CALUDE_largest_y_coordinate_l2355_235576

theorem largest_y_coordinate (x y : ℝ) : 
  (x - 3)^2 / 25 + (y - 2)^2 / 9 = 0 → y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_l2355_235576


namespace NUMINAMATH_CALUDE_exists_k_for_circle_through_E_l2355_235501

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line equation -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

/-- The fixed point E -/
def point_E : ℝ × ℝ := (-1, 0)

/-- Predicate to check if a circle with CD as diameter passes through E -/
def circle_passes_through_E (C D : ℝ × ℝ) : Prop :=
  let (x1, y1) := C
  let (x2, y2) := D
  y1 / (x1 + 1) * y2 / (x2 + 1) = -1

/-- The main theorem -/
theorem exists_k_for_circle_through_E :
  ∃ k : ℝ, k ≠ 0 ∧ k = 7/6 ∧
  ∃ C D : ℝ × ℝ,
    ellipse C.1 C.2 ∧
    ellipse D.1 D.2 ∧
    line k C.1 C.2 ∧
    line k D.1 D.2 ∧
    circle_passes_through_E C D :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_circle_through_E_l2355_235501


namespace NUMINAMATH_CALUDE_abe_age_sum_l2355_235559

/-- The sum of Abe's present age and his age 7 years ago is 31, given that Abe's present age is 19. -/
theorem abe_age_sum : 
  let present_age : ℕ := 19
  let years_ago : ℕ := 7
  present_age + (present_age - years_ago) = 31 := by sorry

end NUMINAMATH_CALUDE_abe_age_sum_l2355_235559


namespace NUMINAMATH_CALUDE_pump_out_time_l2355_235530

/-- Calculates the time needed to pump out water from a flooded basement -/
theorem pump_out_time (length width depth : ℝ) (num_pumps pump_rate : ℝ) (conversion_rate : ℝ) :
  length = 20 ∧ 
  width = 40 ∧ 
  depth = 2 ∧ 
  num_pumps = 5 ∧ 
  pump_rate = 10 ∧ 
  conversion_rate = 7.5 →
  (length * width * depth * conversion_rate) / (num_pumps * pump_rate) = 240 := by
  sorry

end NUMINAMATH_CALUDE_pump_out_time_l2355_235530


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2355_235591

theorem square_perimeter_problem (perimeterA : ℝ) (sideA sideB sideC : ℝ) : 
  perimeterA = 30 →
  sideB = 2 * sideA →
  sideC = sideA + sideB →
  4 * sideC = 90 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2355_235591


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2355_235598

theorem least_number_for_divisibility (n m : ℕ) (hn : n = 1056) (hm : m = 23) :
  ∃ k : ℕ, k > 0 ∧ k ≤ m ∧ (n + k) % m = 0 ∧ ∀ j : ℕ, 0 < j ∧ j < k → (n + j) % m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2355_235598


namespace NUMINAMATH_CALUDE_wooden_toy_price_is_20_l2355_235522

/-- The price of a wooden toy at the Craftee And Best store -/
def wooden_toy_price : ℕ := sorry

/-- The price of a hat at the Craftee And Best store -/
def hat_price : ℕ := 10

/-- The amount Kendra initially had -/
def initial_amount : ℕ := 100

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra received in change -/
def change_received : ℕ := 30

theorem wooden_toy_price_is_20 :
  wooden_toy_price = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_wooden_toy_price_is_20_l2355_235522


namespace NUMINAMATH_CALUDE_inscribed_circle_equation_l2355_235592

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, 3)

-- Define origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the inscribed circle equation
def is_inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Theorem statement
theorem inscribed_circle_equation :
  ∀ x y : ℝ,
  is_inscribed_circle x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - r)^2 + (y - r)^2 = r^2 ∧
    (x - 4)^2 + y^2 = r^2 ∧
    x^2 + (y - 3)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_equation_l2355_235592


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2355_235505

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

/-- The theorem stating the relationship between the maximum value of f and the value of a -/
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) ↔
  a = Real.sqrt 6 + 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2355_235505


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2355_235587

theorem quadratic_equation_roots (m : ℚ) :
  (∃ x : ℚ, x^2 + 2*x + 3*m - 4 = 0) ∧ 
  (2^2 + 2*2 + 3*m - 4 = 0) →
  ((-4)^2 + 2*(-4) + 3*m - 4 = 0) ∧ 
  m = -4/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2355_235587


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2355_235548

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) :
  S n = 48 ∧ S (2 * n) = 60 →
  S (3 * n) = 63 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2355_235548


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2355_235582

/-- A regular polygon with perimeter 49 and side length 7 has 7 sides. -/
theorem regular_polygon_sides (p : ℕ) (s : ℕ) (h1 : p = 49) (h2 : s = 7) :
  p / s = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2355_235582


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2355_235569

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers without any zero -/
def six_digit_numbers_without_zero : ℕ := 531441

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero :
  total_six_digit_numbers - six_digit_numbers_without_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2355_235569


namespace NUMINAMATH_CALUDE_doodads_for_thingamabobs_l2355_235515

/-- The number of doodads required to make one widget -/
def doodads_per_widget : ℚ := 18 / 5

/-- The number of widgets required to make one thingamabob -/
def widgets_per_thingamabob : ℚ := 11 / 4

/-- The number of thingamabobs we want to make -/
def target_thingamabobs : ℕ := 80

/-- Theorem stating that 792 doodads are required to make 80 thingamabobs -/
theorem doodads_for_thingamabobs : 
  ⌈(target_thingamabobs : ℚ) * widgets_per_thingamabob * doodads_per_widget⌉ = 792 := by
  sorry

end NUMINAMATH_CALUDE_doodads_for_thingamabobs_l2355_235515


namespace NUMINAMATH_CALUDE_arc_length_for_150_degrees_l2355_235519

theorem arc_length_for_150_degrees (r : ℝ) (θ : ℝ) :
  r = 6 →
  θ = 150 →
  (θ / 180) * π * r = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_150_degrees_l2355_235519


namespace NUMINAMATH_CALUDE_daisy_monday_toys_l2355_235562

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := sorry

/-- The number of dog toys Daisy had on Tuesday after losing some -/
def tuesday_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Tuesday -/
def tuesday_new_toys : ℕ := 3

/-- The number of new toys Daisy's owner bought on Wednesday -/
def wednesday_new_toys : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys : ℕ := 13

theorem daisy_monday_toys : 
  monday_toys = 5 :=
by sorry

end NUMINAMATH_CALUDE_daisy_monday_toys_l2355_235562


namespace NUMINAMATH_CALUDE_square_rotation_lateral_area_l2355_235540

/-- The lateral surface area of a cylinder formed by rotating a square around one of its sides -/
theorem square_rotation_lateral_area (side_length : ℝ) (h : side_length = 2) :
  2 * side_length * Real.pi = 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_rotation_lateral_area_l2355_235540


namespace NUMINAMATH_CALUDE_angle_with_same_terminal_side_l2355_235514

/-- The angle (in degrees) that has the same terminal side as 1303° -/
def equivalent_angle : ℝ := -137

/-- Theorem stating that the angle with the same terminal side as 1303° is -137° -/
theorem angle_with_same_terminal_side :
  ∃ (k : ℤ), 1303 = 360 * k + equivalent_angle := by
  sorry

end NUMINAMATH_CALUDE_angle_with_same_terminal_side_l2355_235514


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2355_235512

theorem quadratic_real_roots_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2355_235512


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2355_235594

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) + a * i = (b * i - 1) * i) : 
  (a : ℂ) + b * i = -1 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2355_235594
