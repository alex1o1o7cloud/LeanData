import Mathlib

namespace NUMINAMATH_GPT_age_ratio_l796_79604

variable (A B : ℕ)
variable (k : ℕ)

-- Define the conditions
def sum_of_ages : Prop := A + B = 60
def multiple_of_age : Prop := A = k * B

-- Theorem to prove the ratio of ages
theorem age_ratio (h_sum : sum_of_ages A B) (h_multiple : multiple_of_age A B k) : A = 12 * B :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l796_79604


namespace NUMINAMATH_GPT_senior_ticket_cost_is_13_l796_79608

theorem senior_ticket_cost_is_13
    (adult_ticket_cost : ℕ)
    (child_ticket_cost : ℕ)
    (senior_ticket_cost : ℕ)
    (total_cost : ℕ)
    (num_adults : ℕ)
    (num_children : ℕ)
    (num_senior_citizens : ℕ)
    (age_child1 : ℕ)
    (age_child2 : ℕ)
    (age_child3 : ℕ) :
    adult_ticket_cost = 11 → 
    child_ticket_cost = 8 →
    total_cost = 64 →
    num_adults = 2 →
    num_children = 2 → -- children with discount tickets
    num_senior_citizens = 2 →
    age_child1 = 7 → 
    age_child2 = 10 → 
    age_child3 = 14 → -- this child does not get discount
    senior_ticket_cost * num_senior_citizens = total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) →
    senior_ticket_cost = 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_senior_ticket_cost_is_13_l796_79608


namespace NUMINAMATH_GPT_bob_age_l796_79638

variable {b j : ℝ}

theorem bob_age (h1 : b = 3 * j - 20) (h2 : b + j = 75) : b = 51 := by
  sorry

end NUMINAMATH_GPT_bob_age_l796_79638


namespace NUMINAMATH_GPT_distance_from_center_to_line_l796_79669

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_center_to_line_l796_79669


namespace NUMINAMATH_GPT_inequality_proof_l796_79699

variable (a b : ℝ)

theorem inequality_proof (h : a < b) : 1 - a > 1 - b :=
sorry

end NUMINAMATH_GPT_inequality_proof_l796_79699


namespace NUMINAMATH_GPT_extended_morse_code_symbols_l796_79629

def symbol_count (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 2
  else if n = 3 then 1
  else if n = 4 then 1 + 4 + 1
  else if n = 5 then 1 + 8
  else 0

theorem extended_morse_code_symbols : 
  (symbol_count 1 + symbol_count 2 + symbol_count 3 + symbol_count 4 + symbol_count 5) = 20 :=
by sorry

end NUMINAMATH_GPT_extended_morse_code_symbols_l796_79629


namespace NUMINAMATH_GPT_carlos_jogged_distance_l796_79654

def carlos_speed := 4 -- Carlos's speed in miles per hour
def jogging_time := 2 -- Time in hours

theorem carlos_jogged_distance : carlos_speed * jogging_time = 8 :=
by
  sorry

end NUMINAMATH_GPT_carlos_jogged_distance_l796_79654


namespace NUMINAMATH_GPT_ellipse_equation_and_m_value_l796_79657

variable {a b : ℝ}
variable (e : ℝ) (F : ℝ × ℝ) (h1 : e = Real.sqrt 2 / 2) (h2 : F = (1, 0))

theorem ellipse_equation_and_m_value (h3 : a > b) (h4 : b > 0) 
  (h5 : (x y : ℝ) → (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x - 1) ^ 2 + y ^ 2 = 1) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ m : ℝ, (y = x + m) → 
  ((∃ A B : ℝ × ℝ, A = (x₁, x₁ + m) ∧ B = (x₂, x₂ + m) ∧
  (x₁ ^ 2) / 2 + (x₁ + m) ^ 2 = 1 ∧ (x₂ ^ 2) / 2 + (x₂ + m) ^ 2 = 1 ∧
  x₁ * x₂ + (x₁ + m) * (x₂ + m) = -1) ↔ m = Real.sqrt 3 / 3 ∨ m = - Real.sqrt 3 / 3))
  :=
sorry

end NUMINAMATH_GPT_ellipse_equation_and_m_value_l796_79657


namespace NUMINAMATH_GPT_least_clock_equiv_square_l796_79645

def clock_equiv (h k : ℕ) : Prop := (h - k) % 24 = 0

theorem least_clock_equiv_square : ∃ (h : ℕ), h > 6 ∧ (h^2) % 24 = h % 24 ∧ (∀ (k : ℕ), k > 6 ∧ clock_equiv k (k^2) → h ≤ k) :=
sorry

end NUMINAMATH_GPT_least_clock_equiv_square_l796_79645


namespace NUMINAMATH_GPT_number_of_solutions_decrease_l796_79659

-- Define the conditions and the main theorem
theorem number_of_solutions_decrease (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∀ x y : ℝ, x^2 - x^2 = 0 ∧ (x - a)^2 + x^2 = 1) →
  a = 1 ∨ a = -1 := 
sorry

end NUMINAMATH_GPT_number_of_solutions_decrease_l796_79659


namespace NUMINAMATH_GPT_chess_tournament_participants_l796_79634

theorem chess_tournament_participants (n : ℕ) 
  (h : (n * (n - 1)) / 2 = 15) : n = 6 :=
sorry

end NUMINAMATH_GPT_chess_tournament_participants_l796_79634


namespace NUMINAMATH_GPT_friend_spent_seven_l796_79685

/-- You and your friend spent a total of $11 for lunch.
    Your friend spent $3 more than you.
    Prove that your friend spent $7 on their lunch. -/
theorem friend_spent_seven (you friend : ℝ) 
  (h1: you + friend = 11) 
  (h2: friend = you + 3) : 
  friend = 7 := 
by 
  sorry

end NUMINAMATH_GPT_friend_spent_seven_l796_79685


namespace NUMINAMATH_GPT_line_equation_l796_79684

theorem line_equation (m b : ℝ) (h_slope : m = 3) (h_intercept : b = 4) :
  3 * x - y + 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l796_79684


namespace NUMINAMATH_GPT_initial_value_l796_79640

theorem initial_value (x k : ℤ) (h : x + 335 = k * 456) : x = 121 := sorry

end NUMINAMATH_GPT_initial_value_l796_79640


namespace NUMINAMATH_GPT_ellipse_standard_equation_midpoint_trajectory_equation_l796_79650

theorem ellipse_standard_equation :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ (∀ x y, (x, y) = (2, 0) → x^2 / a^2 + y^2 / b^2 = 1) → (a = 2 ∧ b = 1) :=
sorry

theorem midpoint_trajectory_equation :
  ∀ x y : ℝ,
  (∃ x0 y0 : ℝ, x0 = 2 * x - 1 ∧ y0 = 2 * y - 1 / 2 ∧ (x0^2 / 4 + y0^2 = 1)) →
  (x - 1 / 2)^2 + 4 * (y - 1 / 4)^2 = 1 :=
sorry

end NUMINAMATH_GPT_ellipse_standard_equation_midpoint_trajectory_equation_l796_79650


namespace NUMINAMATH_GPT_cos_A_value_compare_angles_l796_79622

variable (A B C : ℝ) (a b c : ℝ)

-- Given conditions
variable (h1 : a = 3) (h2 : b = 2 * Real.sqrt 6) (h3 : B = 2 * A)

-- Problem (I) statement
theorem cos_A_value (hcosA : Real.cos A = Real.sqrt 6 / 3) : 
  Real.cos A = Real.sqrt 6 / 3 :=
by 
  sorry

-- Problem (II) statement
theorem compare_angles (hcosA : Real.cos A = Real.sqrt 6 / 3) (hcosC : Real.cos C = Real.sqrt 6 / 9) :
  B < C :=
by
  sorry

end NUMINAMATH_GPT_cos_A_value_compare_angles_l796_79622


namespace NUMINAMATH_GPT_topsoil_cost_proof_l796_79601

-- Definitions
def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def amount_in_cubic_yards : ℕ := 7

-- Theorem
theorem topsoil_cost_proof : cost_per_cubic_foot * cubic_feet_per_cubic_yard * amount_in_cubic_yards = 1512 := by
  -- proof logic goes here
  sorry

end NUMINAMATH_GPT_topsoil_cost_proof_l796_79601


namespace NUMINAMATH_GPT_algebra_problem_l796_79665

theorem algebra_problem (a b c d x : ℝ) (h1 : a = -b) (h2 : c * d = 1) (h3 : |x| = 3) : 
  (a + b) / 2023 + c * d - x^2 = -8 := by
  sorry

end NUMINAMATH_GPT_algebra_problem_l796_79665


namespace NUMINAMATH_GPT_gym_monthly_revenue_l796_79602

theorem gym_monthly_revenue (members_per_month_fee : ℕ) (num_members : ℕ) 
  (h1 : members_per_month_fee = 18 * 2) 
  (h2 : num_members = 300) : 
  num_members * members_per_month_fee = 10800 := 
by 
  -- calculation rationale goes here
  sorry

end NUMINAMATH_GPT_gym_monthly_revenue_l796_79602


namespace NUMINAMATH_GPT_log_expression_l796_79641

variable (a : ℝ) (log3 : ℝ → ℝ)
axiom h_a : a = log3 2
axiom log3_8_eq : log3 8 = 3 * log3 2
axiom log3_6_eq : log3 6 = log3 2 + 1

theorem log_expression (log_def : log3 8 - 2 * log3 6 = a - 2) :
  log3 8 - 2 * log3 6 = a - 2 := by
  sorry

end NUMINAMATH_GPT_log_expression_l796_79641


namespace NUMINAMATH_GPT_digit_after_decimal_l796_79619

theorem digit_after_decimal (n : ℕ) : 
  (Nat.floor (10 * (Real.sqrt (n^2 + n) - Nat.floor (Real.sqrt (n^2 + n))))) = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_after_decimal_l796_79619


namespace NUMINAMATH_GPT_sin_range_l796_79696

theorem sin_range :
  ∀ x, (-Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4) → (∃ y, y = Real.sin x ∧ -Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1) := by
  sorry

end NUMINAMATH_GPT_sin_range_l796_79696


namespace NUMINAMATH_GPT_max_value_of_expression_eq_two_l796_79620

noncomputable def max_value_of_expression (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) : ℝ :=
  (a^2 + b^2 + c^2) / c^2

theorem max_value_of_expression_eq_two (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_a : a = 3) :
  max_value_of_expression a b c h_right_triangle h_a = 2 := by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_eq_two_l796_79620


namespace NUMINAMATH_GPT_geometric_sequence_value_a3_l796_79632

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Conditions given in the problem
variable (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 2)
variable (h₂ : (geometric_sequence a₁ q 4) * (geometric_sequence a₁ q 6) = 4 * (geometric_sequence a₁ q 7) ^ 2)

-- The goal is to prove that a₃ = 1
theorem geometric_sequence_value_a3 : geometric_sequence a₁ q 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_a3_l796_79632


namespace NUMINAMATH_GPT_trigonometric_identity_l796_79674

open Real

theorem trigonometric_identity (α : ℝ) (hα : sin (2 * π - α) = 4 / 5) (hα_range : 3 * π / 2 < α ∧ α < 2 * π) : 
  (sin α + cos α) / (sin α - cos α) = 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l796_79674


namespace NUMINAMATH_GPT_problem1_problem2_l796_79614

open Real

theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
sorry

theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + b^3) * (a^3 + b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l796_79614


namespace NUMINAMATH_GPT_sin_cos_sum_eq_one_or_neg_one_l796_79600

theorem sin_cos_sum_eq_one_or_neg_one (α : ℝ) (h : (Real.sin α)^4 + (Real.cos α)^4 = 1) : (Real.sin α + Real.cos α) = 1 ∨ (Real.sin α + Real.cos α) = -1 :=
sorry

end NUMINAMATH_GPT_sin_cos_sum_eq_one_or_neg_one_l796_79600


namespace NUMINAMATH_GPT_part_I_part_II_l796_79680

theorem part_I (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_adbc: a * d = b * c) (h_ineq1: a + d > b + c): |a - d| > |b - c| :=
sorry

theorem part_II (a b c d t: ℝ) 
(h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(h_eq: t * (Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2)) = Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)):
t >= Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l796_79680


namespace NUMINAMATH_GPT_circle_area_l796_79656

-- Definition of the given circle equation
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Prove the area of the circle defined by circle_eq (x y) is 25/4 * π
theorem circle_area (x y : ℝ) (h : circle_eq x y) : ∃ r : ℝ, r = 5 / 2 ∧ π * r^2 = 25 / 4 * π :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l796_79656


namespace NUMINAMATH_GPT_angle_measure_triple_complement_l796_79661

variable (x : ℝ)

theorem angle_measure_triple_complement (h : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end NUMINAMATH_GPT_angle_measure_triple_complement_l796_79661


namespace NUMINAMATH_GPT_hyperbola_eccentricity_b_value_l796_79628

theorem hyperbola_eccentricity_b_value (b : ℝ) (a : ℝ) (e : ℝ) 
  (h1 : a^2 = 1) (h2 : e = 2) 
  (h3 : b > 0) (h4 : b^2 = 4 - 1) : 
  b = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_b_value_l796_79628


namespace NUMINAMATH_GPT_arithmetic_sequence_n_2005_l796_79626

/-- Define an arithmetic sequence with first term a₁ = 1 and common difference d = 3. -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + (n - 1) * 3

/-- Statement of the proof problem. -/
theorem arithmetic_sequence_n_2005 : 
  ∃ n : ℕ, arithmetic_sequence n = 2005 ∧ n = 669 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_2005_l796_79626


namespace NUMINAMATH_GPT_initial_notebooks_is_10_l796_79603

-- Define the conditions
def ordered_notebooks := 6
def lost_notebooks := 2
def current_notebooks := 14

-- Define the initial number of notebooks
def initial_notebooks (N : ℕ) :=
  N + ordered_notebooks - lost_notebooks = current_notebooks

-- The proof statement
theorem initial_notebooks_is_10 : initial_notebooks 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_notebooks_is_10_l796_79603


namespace NUMINAMATH_GPT_factorization_of_polynomial_l796_79643

theorem factorization_of_polynomial :
  (x : ℤ) → x^10 + x^5 + 1 = (x^2 + x + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l796_79643


namespace NUMINAMATH_GPT_hypotenuse_length_l796_79636

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l796_79636


namespace NUMINAMATH_GPT_symmetric_line_equation_l796_79644

-- Definitions of the given conditions.
def original_line_equation (x y : ℝ) : Prop := 2 * x + 3 * y + 6 = 0
def line_of_symmetry (x y : ℝ) : Prop := y = x

-- The theorem statement to prove:
theorem symmetric_line_equation (x y : ℝ) : original_line_equation y x ↔ (3 * x + 2 * y + 6 = 0) :=
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l796_79644


namespace NUMINAMATH_GPT_sqrt7_problem_l796_79642

theorem sqrt7_problem (x y : ℝ) (h1 : 2 < Real.sqrt 7) (h2 : Real.sqrt 7 < 3) (hx : x = 2) (hy : y = Real.sqrt 7 - 2) :
  (x + Real.sqrt 7) * y = 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt7_problem_l796_79642


namespace NUMINAMATH_GPT_train_length_l796_79637

namespace TrainProblem

def speed_kmh : ℤ := 60
def time_sec : ℤ := 18
def speed_ms : ℚ := (speed_kmh : ℚ) * (1000 / 1) * (1 / 3600)
def length_meter := speed_ms * (time_sec : ℚ)

theorem train_length :
  length_meter = 300.06 := by
  sorry

end TrainProblem

end NUMINAMATH_GPT_train_length_l796_79637


namespace NUMINAMATH_GPT_xyz_problem_l796_79612

variables {x y z : ℝ}

theorem xyz_problem
  (h1 : y + z = 10 - 4 * x)
  (h2 : x + z = -16 - 4 * y)
  (h3 : x + y = 9 - 4 * z) :
  3 * x + 3 * y + 3 * z = 1.5 :=
by 
  sorry

end NUMINAMATH_GPT_xyz_problem_l796_79612


namespace NUMINAMATH_GPT_problem_statement_l796_79672

theorem problem_statement (a b c d : ℤ) (h1 : a - b = -3) (h2 : c + d = 2) : (b + c) - (a - d) = 5 :=
by
  -- Proof steps skipped.
  sorry

end NUMINAMATH_GPT_problem_statement_l796_79672


namespace NUMINAMATH_GPT_find_base_l796_79631

theorem find_base (b x y : ℝ) (h₁ : b^x * 4^y = 59049) (h₂ : x = 10) (h₃ : x - y = 10) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_base_l796_79631


namespace NUMINAMATH_GPT_simplify_and_evaluate_l796_79673

-- Define the expression as a function of a and b
def expr (a b : ℚ) : ℚ := 5 * a * b - 2 * (3 * a * b - (4 * a * b^2 + (1/2) * a * b)) - 5 * a * b^2

-- State the condition and the target result
theorem simplify_and_evaluate : 
  let a : ℚ := -1
  let b : ℚ := 1 / 2
  expr a b = -3 / 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l796_79673


namespace NUMINAMATH_GPT_train_speed_is_42_point_3_km_per_h_l796_79627

-- Definitions for the conditions.
def train_length : ℝ := 150
def bridge_length : ℝ := 320
def crossing_time : ℝ := 40
def meter_per_sec_to_km_per_hour : ℝ := 3.6
def total_distance : ℝ := train_length + bridge_length

-- The theorem we want to prove
theorem train_speed_is_42_point_3_km_per_h : 
    (total_distance / crossing_time) * meter_per_sec_to_km_per_hour = 42.3 :=
by 
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_train_speed_is_42_point_3_km_per_h_l796_79627


namespace NUMINAMATH_GPT_store_profit_l796_79658

variables (m n : ℝ)

def total_profit (m n : ℝ) : ℝ :=
  110 * m - 50 * n

theorem store_profit (m n : ℝ) : total_profit m n = 110 * m - 50 * n :=
  by
  -- sorry indicates that the proof is skipped
  sorry

end NUMINAMATH_GPT_store_profit_l796_79658


namespace NUMINAMATH_GPT_count_four_digit_numbers_l796_79653

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_l796_79653


namespace NUMINAMATH_GPT_smallest_lcm_of_4digit_gcd_5_l796_79663

theorem smallest_lcm_of_4digit_gcd_5 :
  ∃ (m n : ℕ), (1000 ≤ m ∧ m < 10000) ∧ (1000 ≤ n ∧ n < 10000) ∧ 
               m.gcd n = 5 ∧ m.lcm n = 203010 :=
by sorry

end NUMINAMATH_GPT_smallest_lcm_of_4digit_gcd_5_l796_79663


namespace NUMINAMATH_GPT_arithmetic_sequence_twentieth_term_l796_79666

theorem arithmetic_sequence_twentieth_term
  (a1 : ℤ) (a13 : ℤ) (a20 : ℤ) (d : ℤ)
  (h1 : a1 = 3)
  (h2 : a13 = 27)
  (h3 : a13 = a1 + 12 * d)
  (h4 : a20 = a1 + 19 * d) : 
  a20 = 41 :=
by
  --  We assume a20 and prove it equals 41 instead of solving it in steps
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_twentieth_term_l796_79666


namespace NUMINAMATH_GPT_find_incorrect_statements_l796_79683

-- Definitions of the statements based on their mathematical meanings
def is_regular_tetrahedron (shape : Type) : Prop := 
  -- assume some definition for regular tetrahedron
  sorry 

def is_cube (shape : Type) : Prop :=
  -- assume some definition for cube
  sorry

def is_generatrix_parallel (cylinder : Type) : Prop :=
  -- assume definition stating that generatrix of a cylinder is parallel to its axis
  sorry

def is_lateral_faces_isosceles (pyramid : Type) : Prop :=
  -- assume definition that in a regular pyramid, lateral faces are congruent isosceles triangles
  sorry

def forms_cone_on_rotation (triangle : Type) (axis : Type) : Prop :=
  -- assume definition that a right triangle forms a cone when rotated around one of its legs (other than hypotenuse)
  sorry

-- Given conditions as definitions
def statement_A : Prop := ∀ (shape : Type), is_regular_tetrahedron shape → is_cube shape = false
def statement_B : Prop := ∀ (cylinder : Type), is_generatrix_parallel cylinder = true
def statement_C : Prop := ∀ (pyramid : Type), is_lateral_faces_isosceles pyramid = true
def statement_D : Prop := ∀ (triangle : Type) (axis : Type), forms_cone_on_rotation triangle axis = false

-- The proof problem equivalent to incorrectness of statements A, B, and D
theorem find_incorrect_statements : 
  (statement_A = true) ∧ -- statement A is indeed incorrect
  (statement_B = true) ∧ -- statement B is indeed incorrect
  (statement_C = false) ∧ -- statement C is correct
  (statement_D = true)    -- statement D is indeed incorrect
:= 
sorry

end NUMINAMATH_GPT_find_incorrect_statements_l796_79683


namespace NUMINAMATH_GPT_three_digit_numbers_excluding_adjacent_same_digits_is_correct_l796_79690

def num_valid_three_digit_numbers_exclude_adjacent_same_digits : Nat :=
  let total_numbers := 900
  let excluded_numbers_AAB := 81
  let excluded_numbers_BAA := 81
  total_numbers - (excluded_numbers_AAB + excluded_numbers_BAA)

theorem three_digit_numbers_excluding_adjacent_same_digits_is_correct :
  num_valid_three_digit_numbers_exclude_adjacent_same_digits = 738 := by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_excluding_adjacent_same_digits_is_correct_l796_79690


namespace NUMINAMATH_GPT_ratio_of_x_and_y_l796_79648

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_and_y_l796_79648


namespace NUMINAMATH_GPT_solve_for_x_l796_79686

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (7 * x) ^ 5 = (14 * x) ^ 4 → x = 16 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l796_79686


namespace NUMINAMATH_GPT_average_marks_l796_79691

theorem average_marks (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) 
                      (chemistry_marks : ℕ) (biology_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 →
  chemistry_marks = 87 → biology_marks = 81 → 
  (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / 5 = 85 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_marks_l796_79691


namespace NUMINAMATH_GPT_three_pairs_exist_l796_79649

theorem three_pairs_exist :
  ∃! S P : ℕ, 5 * S + 7 * P = 90 :=
by
  sorry

end NUMINAMATH_GPT_three_pairs_exist_l796_79649


namespace NUMINAMATH_GPT_max_tension_of_pendulum_l796_79635

theorem max_tension_of_pendulum 
  (m g L θ₀ : ℝ) 
  (h₀ : θ₀ < π / 2) 
  (T₀ : ℝ) 
  (no_air_resistance : true) 
  (no_friction : true) : 
  ∃ T_max, T_max = m * g * (3 - 2 * Real.cos θ₀) := 
by 
  sorry

end NUMINAMATH_GPT_max_tension_of_pendulum_l796_79635


namespace NUMINAMATH_GPT_final_length_of_movie_l796_79624

theorem final_length_of_movie :
  let original_length := 3600 -- original movie length in seconds
  let cut_1 := 3 * 60 -- first scene cut in seconds
  let cut_2 := (5 * 60) + 30 -- second scene cut in seconds
  let cut_3 := (2 * 60) + 15 -- third scene cut in seconds
  let total_cut := cut_1 + cut_2 + cut_3 -- total cut time in seconds
  let final_length_seconds := original_length - total_cut -- final length in seconds
  final_length_seconds = 2955 ∧ final_length_seconds / 60 = 49 ∧ final_length_seconds % 60 = 15
:= by
  sorry

end NUMINAMATH_GPT_final_length_of_movie_l796_79624


namespace NUMINAMATH_GPT_day50_previous_year_is_Wednesday_l796_79652

-- Given conditions
variable (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)

-- Provided conditions stating specific days are Fridays
def day250_is_Friday : Prop := dayOfWeek 250 N = 5
def day150_is_Friday_next_year : Prop := dayOfWeek 150 (N+1) = 5

-- Proving the day of week for the 50th day of year N-1
def day50_previous_year : Prop := dayOfWeek 50 (N-1) = 3

-- Main theorem tying it together
theorem day50_previous_year_is_Wednesday (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)
  (h1 : day250_is_Friday N dayOfWeek)
  (h2 : day150_is_Friday_next_year N dayOfWeek) :
  day50_previous_year N dayOfWeek :=
sorry -- Placeholder for actual proof

end NUMINAMATH_GPT_day50_previous_year_is_Wednesday_l796_79652


namespace NUMINAMATH_GPT_concert_people_count_l796_79647

variable {W M : ℕ}

theorem concert_people_count (h1 : W * 2 = M) (h2 : (W - 12) * 3 = M - 29) : W + M = 21 := 
sorry

end NUMINAMATH_GPT_concert_people_count_l796_79647


namespace NUMINAMATH_GPT_collinear_vectors_value_m_l796_79694

theorem collinear_vectors_value_m (m : ℝ) : 
  (∃ k : ℝ, (2*m = k * (m - 1)) ∧ (3 = k)) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_collinear_vectors_value_m_l796_79694


namespace NUMINAMATH_GPT_ninja_star_ratio_l796_79625

-- Define variables for the conditions
variables (Eric_stars Chad_stars Jeff_stars Total_stars : ℕ) (Jeff_bought : ℕ)

/-- Given the following conditions:
1. Eric has 4 ninja throwing stars.
2. Jeff now has 6 throwing stars.
3. Jeff bought 2 ninja stars from Chad.
4. Altogether, they have 16 ninja throwing stars.

We want to prove that the ratio of the number of ninja throwing stars Chad has to the number Eric has is 2:1. --/
theorem ninja_star_ratio
  (h1 : Eric_stars = 4)
  (h2 : Jeff_stars = 6)
  (h3 : Jeff_bought = 2)
  (h4 : Total_stars = 16)
  (h5 : Eric_stars + Jeff_stars - Jeff_bought + Chad_stars = Total_stars) :
  Chad_stars / Eric_stars = 2 :=
by
  sorry

end NUMINAMATH_GPT_ninja_star_ratio_l796_79625


namespace NUMINAMATH_GPT_find_blue_sea_glass_pieces_l796_79695

-- Define all required conditions and the proof problem.
theorem find_blue_sea_glass_pieces (B : ℕ) : 
  let BlancheRed := 3
  let RoseRed := 9
  let DorothyRed := 2 * (BlancheRed + RoseRed)
  let DorothyBlue := 3 * B
  let DorothyTotal := 57
  DorothyTotal = DorothyRed + DorothyBlue → B = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_blue_sea_glass_pieces_l796_79695


namespace NUMINAMATH_GPT_percentage_of_part_of_whole_l796_79660

theorem percentage_of_part_of_whole :
  let part := 375.2
  let whole := 12546.8
  (part / whole) * 100 = 2.99 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_part_of_whole_l796_79660


namespace NUMINAMATH_GPT_am_gm_inequality_l796_79613

theorem am_gm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a^3 + b^3 + a + b ≥ 4 * a * b :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l796_79613


namespace NUMINAMATH_GPT_find_a_l796_79618

theorem find_a (a : ℝ) (h : 1 / Real.log 5 / Real.log a + 1 / Real.log 6 / Real.log a + 1 / Real.log 10 / Real.log a = 1) : a = 300 :=
sorry

end NUMINAMATH_GPT_find_a_l796_79618


namespace NUMINAMATH_GPT_ways_A_to_C_via_B_l796_79605

def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 3

theorem ways_A_to_C_via_B : ways_A_to_B * ways_B_to_C = 6 := by
  sorry

end NUMINAMATH_GPT_ways_A_to_C_via_B_l796_79605


namespace NUMINAMATH_GPT_tangent_condition_l796_79639

def curve1 (x y : ℝ) : Prop := y = x ^ 3 + 2
def curve2 (x y m : ℝ) : Prop := y^2 - m * x = 1

theorem tangent_condition (m : ℝ) (h : ∃ x y : ℝ, curve1 x y ∧ curve2 x y m) :
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tangent_condition_l796_79639


namespace NUMINAMATH_GPT_find_BG_l796_79689

-- Define given lengths and the required proof
def BC : ℝ := 5
def BF : ℝ := 12

theorem find_BG : BG = 13 := by
  -- Formal proof would go here
  sorry

end NUMINAMATH_GPT_find_BG_l796_79689


namespace NUMINAMATH_GPT_simplify_expression_evaluate_expression_with_values_l796_79677

-- Problem 1: Simplify the expression to -xy
theorem simplify_expression (x y : ℤ) : 
  3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = - x * y :=
  sorry

-- Problem 2: Evaluate the expression with given values
theorem evaluate_expression_with_values (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_evaluate_expression_with_values_l796_79677


namespace NUMINAMATH_GPT_calculate_total_cost_l796_79662

def num_chicken_nuggets := 100
def num_per_box := 20
def cost_per_box := 4

theorem calculate_total_cost :
  (num_chicken_nuggets / num_per_box) * cost_per_box = 20 := by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l796_79662


namespace NUMINAMATH_GPT_S_when_R_is_16_and_T_is_1_div_4_l796_79687

theorem S_when_R_is_16_and_T_is_1_div_4 :
  ∃ (S : ℝ), (∀ (R S T : ℝ) (c : ℝ), (R = c * S / T) →
  (2 = c * 8 / (1/2)) → c = 1 / 8) ∧
  (16 = (1/8) * S / (1/4)) → S = 32 :=
sorry

end NUMINAMATH_GPT_S_when_R_is_16_and_T_is_1_div_4_l796_79687


namespace NUMINAMATH_GPT_fraction_identity_l796_79630

variables {a b c x : ℝ}

theorem fraction_identity (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : c ≠ a) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + 2 * b + 3 * c) / (a - b - 3 * c) = (b * (x + 2) + 3 * c) / (b * (x - 1) - 3 * c) :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_identity_l796_79630


namespace NUMINAMATH_GPT_speed_in_kmph_l796_79646

noncomputable def speed_conversion (speed_mps: ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_in_kmph : speed_conversion 18.334799999999998 = 66.00528 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_speed_in_kmph_l796_79646


namespace NUMINAMATH_GPT_sqrt_31_minus_2_in_range_l796_79670

-- Defining the conditions based on the problem statements
def five_squared : ℤ := 5 * 5
def six_squared : ℤ := 6 * 6
def thirty_one : ℤ := 31

theorem sqrt_31_minus_2_in_range : 
  (5 * 5 < thirty_one) ∧ (thirty_one < 6 * 6) →
  3 < (Real.sqrt thirty_one) - 2 ∧ (Real.sqrt thirty_one) - 2 < 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_31_minus_2_in_range_l796_79670


namespace NUMINAMATH_GPT_min_value_l796_79681

noncomputable def min_value_of_expression (a b: ℝ) :=
    a > 0 ∧ b > 0 ∧ a + b = 1 → (∃ (m : ℝ), (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2)

theorem min_value (a b: ℝ) (h₀: a > 0) (h₁: b > 0) (h₂: a + b = 1) :
    ∃ m, (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) ∧ m = 3 + 2 * Real.sqrt 2 := 
by
    sorry

end NUMINAMATH_GPT_min_value_l796_79681


namespace NUMINAMATH_GPT_distribution_of_tickets_l796_79609

-- Define the number of total people and the number of tickets
def n : ℕ := 10
def k : ℕ := 3

-- Define the permutation function P(n, k)
def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Main theorem statement
theorem distribution_of_tickets : P n k = 720 := by
  unfold P
  sorry

end NUMINAMATH_GPT_distribution_of_tickets_l796_79609


namespace NUMINAMATH_GPT_smaller_number_of_ratio_4_5_lcm_180_l796_79692

theorem smaller_number_of_ratio_4_5_lcm_180 {a b : ℕ} (h_ratio : 4 * b = 5 * a) (h_lcm : Nat.lcm a b = 180) : a = 144 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_of_ratio_4_5_lcm_180_l796_79692


namespace NUMINAMATH_GPT_max_value_of_E_l796_79697

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  ∀ (a b c d : ℝ),
    (-8.5 ≤ a ∧ a ≤ 8.5) →
    (-8.5 ≤ b ∧ b ≤ 8.5) →
    (-8.5 ≤ c ∧ c ≤ 8.5) →
    (-8.5 ≤ d ∧ d ≤ 8.5) →
    E a b c d ≤ 306 := sorry

end NUMINAMATH_GPT_max_value_of_E_l796_79697


namespace NUMINAMATH_GPT_inequality_always_holds_l796_79693

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l796_79693


namespace NUMINAMATH_GPT_Lagrange_interpolation_poly_l796_79610

noncomputable def Lagrange_interpolation (P : ℝ → ℝ) : Prop :=
  P (-1) = -11 ∧ P (1) = -3 ∧ P (2) = 1 ∧ P (3) = 13

theorem Lagrange_interpolation_poly :
  ∃ P : ℝ → ℝ, Lagrange_interpolation P ∧ ∀ x, P x = x^3 - 2*x^2 + 3*x - 5 :=
by
  sorry

end NUMINAMATH_GPT_Lagrange_interpolation_poly_l796_79610


namespace NUMINAMATH_GPT_money_distribution_l796_79679

theorem money_distribution (A B C : ℝ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 340) : 
  C = 40 := 
sorry

end NUMINAMATH_GPT_money_distribution_l796_79679


namespace NUMINAMATH_GPT_compute_expression_l796_79616

theorem compute_expression : (7^2 - 2 * 5 + 2^3) = 47 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l796_79616


namespace NUMINAMATH_GPT_find_angle_QPR_l796_79615

-- Define the angles and line segment
variables (R S Q T P : Type) 
variables (line_RT : R ≠ S)
variables (x : ℝ) 
variables (angle_PTQ : ℝ := 62)
variables (angle_RPS : ℝ := 34)

-- Hypothesis that PQ = PT, making triangle PQT isosceles
axiom eq_PQ_PT : ℝ

-- Conditions
axiom lie_on_RT : ∀ {R S Q T : Type}, R ≠ S 
axiom angle_PTQ_eq : angle_PTQ = 62
axiom angle_RPS_eq : angle_RPS = 34

-- Hypothesis that defines the problem structure
theorem find_angle_QPR : x = 11 := by
sorry

end NUMINAMATH_GPT_find_angle_QPR_l796_79615


namespace NUMINAMATH_GPT_new_car_travel_distance_l796_79623

theorem new_car_travel_distance
  (old_distance : ℝ)
  (new_distance : ℝ)
  (h1 : old_distance = 150)
  (h2 : new_distance = 1.30 * old_distance) : 
  new_distance = 195 := 
by 
  /- include required assumptions and skip the proof. -/
  sorry

end NUMINAMATH_GPT_new_car_travel_distance_l796_79623


namespace NUMINAMATH_GPT_correct_factorization_l796_79607

-- Definitions from conditions
def A: Prop := ∀ x y: ℝ, x^2 - 4*y^2 = (x + y) * (x - 4*y)
def B: Prop := ∀ x: ℝ, (x + 4) * (x - 4) = x^2 - 16
def C: Prop := ∀ x: ℝ, x^2 - 2*x + 1 = (x - 1)^2
def D: Prop := ∀ x: ℝ, x^2 - 8*x + 9 = (x - 4)^2 - 7

-- Goal is to prove that C is a correct factorization
theorem correct_factorization: C := by
  sorry

end NUMINAMATH_GPT_correct_factorization_l796_79607


namespace NUMINAMATH_GPT_proof_problem_l796_79682

variables {m n : ℝ}

theorem proof_problem (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 2 * m * n) :
  (mn : ℝ) ≥ 1 ∧ (m^2 + n^2 ≥ 2) :=
  sorry

end NUMINAMATH_GPT_proof_problem_l796_79682


namespace NUMINAMATH_GPT_diana_took_six_candies_l796_79651

-- Define the initial number of candies in the box
def initial_candies : ℕ := 88

-- Define the number of candies left in the box after Diana took some
def remaining_candies : ℕ := 82

-- Define the number of candies taken by Diana
def candies_taken : ℕ := initial_candies - remaining_candies

-- The theorem we need to prove
theorem diana_took_six_candies : candies_taken = 6 := by
  sorry

end NUMINAMATH_GPT_diana_took_six_candies_l796_79651


namespace NUMINAMATH_GPT_parametric_curve_intersects_itself_l796_79621

-- Given parametric equations
def param_x (t : ℝ) : ℝ := t^2 + 3
def param_y (t : ℝ) : ℝ := t^3 - 6 * t + 4

-- Existential statement for self-intersection
theorem parametric_curve_intersects_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = param_x t2 ∧ param_y t1 = param_y t2 ∧ param_x t1 = 9 ∧ param_y t1 = 4 :=
sorry

end NUMINAMATH_GPT_parametric_curve_intersects_itself_l796_79621


namespace NUMINAMATH_GPT_popcorn_probability_l796_79698

theorem popcorn_probability {w y b : ℝ} (hw : w = 3/5) (hy : y = 1/5) (hb : b = 1/5)
  {pw py pb : ℝ} (hpw : pw = 1/3) (hpy : py = 3/4) (hpb : pb = 1/2) :
  (y * py) / (w * pw + y * py + b * pb) = 1/3 := 
sorry

end NUMINAMATH_GPT_popcorn_probability_l796_79698


namespace NUMINAMATH_GPT_find_g_neg_three_l796_79667

namespace ProofProblem

def g (d e f x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

theorem find_g_neg_three (d e f : ℝ) (h : g d e f 3 = -9) : g d e f (-3) = 21 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_find_g_neg_three_l796_79667


namespace NUMINAMATH_GPT_distance_traveled_l796_79688

-- Let T be the time in hours taken to travel the actual distance D at 10 km/hr.
-- Let D be the actual distance traveled by the person.
-- Given: D = 10 * T and D + 40 = 20 * T prove that D = 40.

theorem distance_traveled (T : ℝ) (D : ℝ) 
  (h1 : D = 10 * T)
  (h2 : D + 40 = 20 * T) : 
  D = 40 := by
  sorry

end NUMINAMATH_GPT_distance_traveled_l796_79688


namespace NUMINAMATH_GPT_geometric_series_smallest_b_l796_79617

theorem geometric_series_smallest_b (a b c : ℝ) (h_geometric : a * c = b^2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 216) : b = 6 :=
sorry

end NUMINAMATH_GPT_geometric_series_smallest_b_l796_79617


namespace NUMINAMATH_GPT_greatest_sundays_in_56_days_l796_79611

theorem greatest_sundays_in_56_days (days_in_first: ℕ) (days_in_week: ℕ) (sundays_in_week: ℕ) : ℕ :=
by 
  -- Given conditions
  have days_in_first := 56
  have days_in_week := 7
  have sundays_in_week := 1

  -- Conclusion
  let num_weeks := days_in_first / days_in_week

  -- Answer
  exact num_weeks * sundays_in_week

-- This theorem establishes that the greatest number of Sundays in 56 days is indeed 8.
-- Proof: The number of Sundays in 56 days is given by the number of weeks (which is 8) times the number of Sundays per week (which is 1).

example : greatest_sundays_in_56_days 56 7 1 = 8 := 
by 
  unfold greatest_sundays_in_56_days
  exact rfl

end NUMINAMATH_GPT_greatest_sundays_in_56_days_l796_79611


namespace NUMINAMATH_GPT_intersection_P_Q_l796_79675

def P : Set ℝ := { x | x^2 - x = 0 }
def Q : Set ℝ := { x | x^2 + x = 0 }

theorem intersection_P_Q : (P ∩ Q) = {0} := 
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l796_79675


namespace NUMINAMATH_GPT_more_girls_than_boys_l796_79676

theorem more_girls_than_boys (total_kids girls boys : ℕ) (h1 : total_kids = 34) (h2 : girls = 28) (h3 : total_kids = girls + boys) : girls - boys = 22 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_more_girls_than_boys_l796_79676


namespace NUMINAMATH_GPT_least_three_digit_divisible_by_2_3_5_7_l796_79606

theorem least_three_digit_divisible_by_2_3_5_7 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∀ k, 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → n ≤ k) ∧
  (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n) ∧ n = 210 :=
by sorry

end NUMINAMATH_GPT_least_three_digit_divisible_by_2_3_5_7_l796_79606


namespace NUMINAMATH_GPT_line_through_circle_center_slope_one_eq_l796_79678

theorem line_through_circle_center_slope_one_eq (x y : ℝ) :
  (∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 ∧ y = 2) →
  (∃ m : ℝ, m = 1 ∧ (x + 1) = m * (y - 2)) →
  (x - y + 3 = 0) :=
sorry

end NUMINAMATH_GPT_line_through_circle_center_slope_one_eq_l796_79678


namespace NUMINAMATH_GPT_round_to_nearest_hundredth_l796_79668

noncomputable def recurring_decimal (n : ℕ) : ℝ :=
  if n = 87 then 87 + 36 / 99 else 0 -- Defines 87.3636... for n = 87

theorem round_to_nearest_hundredth : recurring_decimal 87 = 87.36 :=
by sorry

end NUMINAMATH_GPT_round_to_nearest_hundredth_l796_79668


namespace NUMINAMATH_GPT_problem1_l796_79664

theorem problem1 (α β : ℝ) 
  (tan_sum : Real.tan (α + β) = 2 / 5) 
  (tan_diff : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
sorry

end NUMINAMATH_GPT_problem1_l796_79664


namespace NUMINAMATH_GPT_parabola_has_one_x_intercept_l796_79671

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- State the theorem that proves the number of x-intercepts
theorem parabola_has_one_x_intercept : ∃! x, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
by
  -- Proof goes here, but it's omitted
  sorry

end NUMINAMATH_GPT_parabola_has_one_x_intercept_l796_79671


namespace NUMINAMATH_GPT_find_base_of_numeral_system_l796_79655

def base_of_numeral_system (x : ℕ) : Prop :=
  (3 * x + 4)^2 = x^3 + 5 * x^2 + 5 * x + 2

theorem find_base_of_numeral_system :
  ∃ x : ℕ, base_of_numeral_system x ∧ x = 7 := sorry

end NUMINAMATH_GPT_find_base_of_numeral_system_l796_79655


namespace NUMINAMATH_GPT_increasing_function_condition_l796_79633

variable {x : ℝ} {a : ℝ}

theorem increasing_function_condition (h : 0 < a) :
  (∀ x ≥ 1, deriv (λ x => x^3 - a * x) x ≥ 0) ↔ (0 < a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_condition_l796_79633
