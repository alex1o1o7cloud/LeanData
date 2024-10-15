import Mathlib

namespace NUMINAMATH_GPT_quadratic_function_passes_through_origin_l959_95913

theorem quadratic_function_passes_through_origin (a : ℝ) :
  ((a - 1) * 0^2 - 0 + a^2 - 1 = 0) → a = -1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_quadratic_function_passes_through_origin_l959_95913


namespace NUMINAMATH_GPT_smallest_integer_divisibility_l959_95931

def smallest_integer (a : ℕ) : Prop :=
  a > 0 ∧ ¬ ∀ b, a = b + 1

theorem smallest_integer_divisibility :
  ∃ a, smallest_integer a ∧ gcd a 63 > 1 ∧ gcd a 66 > 1 ∧ ∀ b, smallest_integer b → b < a → gcd b 63 ≤ 1 ∨ gcd b 66 ≤ 1 :=
sorry

end NUMINAMATH_GPT_smallest_integer_divisibility_l959_95931


namespace NUMINAMATH_GPT_melanie_gave_8_dimes_l959_95911

theorem melanie_gave_8_dimes
  (initial_dimes : ℕ)
  (additional_dimes : ℕ)
  (current_dimes : ℕ)
  (given_away_dimes : ℕ) :
  initial_dimes = 7 →
  additional_dimes = 4 →
  current_dimes = 3 →
  given_away_dimes = (initial_dimes + additional_dimes - current_dimes) →
  given_away_dimes = 8 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_melanie_gave_8_dimes_l959_95911


namespace NUMINAMATH_GPT_max_visible_unit_cubes_from_corner_l959_95922

theorem max_visible_unit_cubes_from_corner :
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  faces_visible - edges_shared + corner_cube = 331 := by
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  have result : faces_visible - edges_shared + corner_cube = 331 := by
    sorry
  exact result

end NUMINAMATH_GPT_max_visible_unit_cubes_from_corner_l959_95922


namespace NUMINAMATH_GPT_triangle_XYZ_median_inequalities_l959_95996

theorem triangle_XYZ_median_inequalities :
  ∀ (XY XZ : ℝ), 
  (∀ (YZ : ℝ), YZ = 10 → 
  ∀ (XM : ℝ), XM = 6 → 
  ∃ (x : ℝ), x = (XY + XZ - 20)/4 → 
  ∃ (N n : ℝ), 
  N = 192 ∧ n = 92 → 
  N - n = 100) :=
by sorry

end NUMINAMATH_GPT_triangle_XYZ_median_inequalities_l959_95996


namespace NUMINAMATH_GPT_problem_intersection_union_complement_l959_95948

open Set Real

noncomputable def A : Set ℝ := {x | x ≥ 2}
noncomputable def B : Set ℝ := {y | y ≤ 3}

theorem problem_intersection_union_complement :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧ 
  (A ∪ B = univ) ∧ 
  (compl A ∩ compl B = ∅) :=
by
  sorry

end NUMINAMATH_GPT_problem_intersection_union_complement_l959_95948


namespace NUMINAMATH_GPT_converse_of_x_eq_one_implies_x_squared_eq_one_l959_95969

theorem converse_of_x_eq_one_implies_x_squared_eq_one (x : ℝ) : x^2 = 1 → x = 1 := 
sorry

end NUMINAMATH_GPT_converse_of_x_eq_one_implies_x_squared_eq_one_l959_95969


namespace NUMINAMATH_GPT_correct_statement_D_l959_95954

theorem correct_statement_D (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
sorry

end NUMINAMATH_GPT_correct_statement_D_l959_95954


namespace NUMINAMATH_GPT_evaluate_f_at_5_l959_95937

def f (x : ℕ) : ℕ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem evaluate_f_at_5 : f 5 = 4881 :=
by
-- proof
sorry

end NUMINAMATH_GPT_evaluate_f_at_5_l959_95937


namespace NUMINAMATH_GPT_maximize_pasture_area_l959_95902

theorem maximize_pasture_area
  (barn_length fence_cost budget : ℕ)
  (barn_length_eq : barn_length = 400)
  (fence_cost_eq : fence_cost = 5)
  (budget_eq : budget = 1500) :
  ∃ x y : ℕ, y = 150 ∧
  x > 0 ∧
  2 * x + y = budget / fence_cost ∧
  y = barn_length - 2 * x ∧
  (x * y) = (75 * 150) :=
by
  sorry

end NUMINAMATH_GPT_maximize_pasture_area_l959_95902


namespace NUMINAMATH_GPT_factorization_example_l959_95938

theorem factorization_example (x : ℝ) : (x^2 - 4 * x + 4) = (x - 2)^2 :=
by sorry

end NUMINAMATH_GPT_factorization_example_l959_95938


namespace NUMINAMATH_GPT_find_root_of_polynomial_l959_95966

theorem find_root_of_polynomial (a c x : ℝ)
  (h1 : a + c = -3)
  (h2 : 64 * a + c = 60)
  (h3 : x = 2) :
  a * x^3 - 2 * x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_root_of_polynomial_l959_95966


namespace NUMINAMATH_GPT_simplify_expression_l959_95993

theorem simplify_expression (c : ℤ) : (3 * c + 6 - 6 * c) / 3 = -c + 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l959_95993


namespace NUMINAMATH_GPT_truck_driver_needs_more_gallons_l959_95910

-- Define the conditions
def miles_per_gallon : ℕ := 3
def total_distance : ℕ := 90
def current_gallons : ℕ := 12
def can_cover_distance : ℕ := miles_per_gallon * current_gallons
def additional_distance_needed : ℕ := total_distance - can_cover_distance

-- Define the main theorem
theorem truck_driver_needs_more_gallons :
  additional_distance_needed / miles_per_gallon = 18 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_truck_driver_needs_more_gallons_l959_95910


namespace NUMINAMATH_GPT_inequality_D_no_solution_l959_95967

theorem inequality_D_no_solution :
  ¬ ∃ x : ℝ, 2 - 3 * x + 2 * x^2 ≤ 0 := 
sorry

end NUMINAMATH_GPT_inequality_D_no_solution_l959_95967


namespace NUMINAMATH_GPT_find_b_l959_95930

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: (-(a / 3) = -c)) (h2 : (-(a / 3) = 1 + a + b + c)) (h3: c = 2) : b = -11 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l959_95930


namespace NUMINAMATH_GPT_abs_diff_squares_105_95_l959_95904

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_105_95_l959_95904


namespace NUMINAMATH_GPT_part1_part2_l959_95914

noncomputable def A (x : ℝ) : Prop := x < 0 ∨ x > 2
noncomputable def B (a x : ℝ) : Prop := a ≤ x ∧ x ≤ 3 - 2 * a

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, A x ∨ B a x) ↔ (a ≤ 0) := 
sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, B a x → (0 ≤ x ∧ x ≤ 2)) ↔ (1 / 2 ≤ a) :=
sorry

end NUMINAMATH_GPT_part1_part2_l959_95914


namespace NUMINAMATH_GPT_trapezoid_base_ratio_l959_95987

-- Define the context of the problem
variables (AB CD : ℝ) (h : AB < CD)

-- Define the main theorem to be proved
theorem trapezoid_base_ratio (h : AB / CD = 1 / 2) :
  ∃ (E F G H I J : ℝ), 
    EJ - EI = FI - FH / 5 ∧ -- These points create segments that divide equally as per the conditions 
    FI - FH = GH / 5 ∧
    GH - GI = HI / 5 ∧
    HI - HJ = JI / 5 ∧
    JI - JE = EJ / 5 :=
sorry

end NUMINAMATH_GPT_trapezoid_base_ratio_l959_95987


namespace NUMINAMATH_GPT_work_problem_l959_95908

theorem work_problem (B_rate : ℝ) (C_rate : ℝ) (A_rate : ℝ) :
  (B_rate = 1/12) →
  (B_rate + C_rate = 1/3) →
  (A_rate + C_rate = 1/2) →
  (A_rate = 1/4) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_work_problem_l959_95908


namespace NUMINAMATH_GPT_none_of_these_true_l959_95999

variable (s r p q : ℝ)
variable (hs : s > 0) (hr : r > 0) (hpq : p * q ≠ 0) (h : s * (p * r) > s * (q * r))

theorem none_of_these_true : ¬ (-p > -q) ∧ ¬ (-p > q) ∧ ¬ (1 > -q / p) ∧ ¬ (1 < q / p) :=
by
  -- The hypothetical theorem to be proven would continue here
  sorry

end NUMINAMATH_GPT_none_of_these_true_l959_95999


namespace NUMINAMATH_GPT_at_least_one_greater_l959_95926

theorem at_least_one_greater (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = a * b * c) :
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
sorry

end NUMINAMATH_GPT_at_least_one_greater_l959_95926


namespace NUMINAMATH_GPT_kelvin_can_win_l959_95933

-- Defining the game conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Game Strategy
def kelvin_always_wins : Prop :=
  ∀ (n : ℕ), ∀ (d : ℕ), (d ∈ (List.range 10)) → 
    ∃ (k : ℕ), k ∈ [3, 7] ∧ ¬is_perfect_square (10 * n + k)

theorem kelvin_can_win : kelvin_always_wins :=
by {
  sorry -- Proof based on strategy of adding 3 or 7 modulo 10 and modulo 100 analysis
}

end NUMINAMATH_GPT_kelvin_can_win_l959_95933


namespace NUMINAMATH_GPT_interval_a_b_l959_95991

noncomputable def f (x : ℝ) : ℝ := |Real.log (x - 1)|

theorem interval_a_b (a b : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < x2) (h3 : x2 < b) (h4 : f x1 > f x2) :
  a < 2 := 
sorry

end NUMINAMATH_GPT_interval_a_b_l959_95991


namespace NUMINAMATH_GPT_breakfast_calories_l959_95949

theorem breakfast_calories : ∀ (planned_calories : ℕ) (B : ℕ),
  planned_calories < 1800 →
  B + 900 + 1100 = planned_calories + 600 →
  B = 400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_breakfast_calories_l959_95949


namespace NUMINAMATH_GPT_general_term_formula_sum_formula_and_max_value_l959_95941

-- Definitions for the conditions
def tenth_term : ℕ → ℤ := λ n => 24
def twenty_fifth_term : ℕ → ℤ := λ n => -21

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℤ) (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) :
  ∀ n : ℕ, a n = -3 * n + 54 := sorry

-- Prove the sum formula and its maximum value
theorem sum_formula_and_max_value (a : ℕ → ℤ) (S : ℕ → ℤ)
  (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) 
  (sum_formula : ∀ n : ℕ, S n = -3 * n^2 / 2 + 51 * n) :
  ∃ max_n : ℕ, S max_n = 578 := sorry

end NUMINAMATH_GPT_general_term_formula_sum_formula_and_max_value_l959_95941


namespace NUMINAMATH_GPT_p_n_divisible_by_5_l959_95900

noncomputable def p_n (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

theorem p_n_divisible_by_5 (n : ℕ) (h : n ≠ 0) : p_n n % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_p_n_divisible_by_5_l959_95900


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_l959_95994

noncomputable def consecutive_integers_sum (n m k : ℤ) : ℤ :=
  n + m + k

theorem sum_of_consecutive_integers_product (n m k : ℤ)
  (h1 : n = m - 1)
  (h2 : k = m + 1)
  (h3 : n * m * k = 990) :
  consecutive_integers_sum n m k = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_l959_95994


namespace NUMINAMATH_GPT_sunil_interest_l959_95927

-- Condition definitions
def A : ℝ := 3370.80
def r : ℝ := 0.06
def n : ℕ := 1
def t : ℕ := 2

-- Derived definition for principal P
noncomputable def P : ℝ := A / (1 + r/n)^(n * t)

-- Interest I calculation
noncomputable def I : ℝ := A - P

-- Proof statement
theorem sunil_interest : I = 370.80 :=
by
  -- Insert the mathematical proof steps here.
  sorry

end NUMINAMATH_GPT_sunil_interest_l959_95927


namespace NUMINAMATH_GPT_min_value_of_f_l959_95942

def f (x : ℝ) (a : ℝ) := - x^3 + a * x^2 - 4

def f_deriv (x : ℝ) (a : ℝ) := - 3 * x^2 + 2 * a * x

theorem min_value_of_f (h : f_deriv (2) a = 0)
  (hm : ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f m a + f_deriv m a ≥ f 0 3 + f_deriv (-1) 3) :
  f 0 3 + f_deriv (-1) 3 = -13 :=
by sorry

end NUMINAMATH_GPT_min_value_of_f_l959_95942


namespace NUMINAMATH_GPT_roots_of_polynomial_l959_95952

theorem roots_of_polynomial (c d : ℝ) (h1 : Polynomial.eval c (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0)
    (h2 : Polynomial.eval d (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0) :
    c * d + c + d = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l959_95952


namespace NUMINAMATH_GPT_means_imply_sum_of_squares_l959_95992

noncomputable def arithmetic_mean (x y z : ℝ) : ℝ :=
(x + y + z) / 3

noncomputable def geometric_mean (x y z : ℝ) : ℝ :=
(x * y * z) ^ (1/3)

noncomputable def harmonic_mean (x y z : ℝ) : ℝ :=
3 / ((1/x) + (1/y) + (1/z))

theorem means_imply_sum_of_squares (x y z : ℝ) :
  arithmetic_mean x y z = 10 →
  geometric_mean x y z = 6 →
  harmonic_mean x y z = 4 →
  x^2 + y^2 + z^2 = 576 :=
by
  -- Proof is omitted for now
  exact sorry

end NUMINAMATH_GPT_means_imply_sum_of_squares_l959_95992


namespace NUMINAMATH_GPT_practice_time_for_Friday_l959_95963

variables (M T W Th F : ℕ)

def conditions : Prop :=
  (M = 2 * T) ∧
  (T = W - 10) ∧
  (W = Th + 5) ∧
  (Th = 50) ∧
  (M + T + W + Th + F = 300)

theorem practice_time_for_Friday (h : conditions M T W Th F) : F = 60 :=
sorry

end NUMINAMATH_GPT_practice_time_for_Friday_l959_95963


namespace NUMINAMATH_GPT_volume_of_pyramid_l959_95958

-- Define conditions
variables (x h : ℝ)
axiom x_pos : x > 0
axiom h_pos : h > 0

-- Define the main theorem/problem statement
theorem volume_of_pyramid (x h : ℝ) (x_pos : x > 0) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (1 / 6) * x^2 * h :=
by sorry

end NUMINAMATH_GPT_volume_of_pyramid_l959_95958


namespace NUMINAMATH_GPT_count_false_propositions_l959_95944

def prop (a : ℝ) := a > 1 → a > 2
def converse (a : ℝ) := a > 2 → a > 1
def inverse (a : ℝ) := a ≤ 1 → a ≤ 2
def contrapositive (a : ℝ) := a ≤ 2 → a ≤ 1

theorem count_false_propositions (a : ℝ) (h : ¬(prop a)) : 
  (¬(prop a) ∧ ¬(contrapositive a)) ∧ (converse a ∧ inverse a) ↔ 2 = 2 := 
  by
    sorry

end NUMINAMATH_GPT_count_false_propositions_l959_95944


namespace NUMINAMATH_GPT_minimum_value_l959_95943

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.cos x)^2 - 2 * (Real.sin x) + 9 / 2

theorem minimum_value :
  ∃ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)), f x = 2 :=
by
  use Real.pi / 6
  sorry

end NUMINAMATH_GPT_minimum_value_l959_95943


namespace NUMINAMATH_GPT_solve_fractional_equation_l959_95915

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  1 / x = 2 / (x + 1) → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l959_95915


namespace NUMINAMATH_GPT_ones_digit_of_8_pow_50_l959_95935

theorem ones_digit_of_8_pow_50 : (8 ^ 50) % 10 = 4 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_8_pow_50_l959_95935


namespace NUMINAMATH_GPT_zack_initial_marbles_l959_95920

theorem zack_initial_marbles :
  ∃ M : ℕ, (∃ k : ℕ, M = 3 * k + 5) ∧ (M - 5 - 60 = 5) ∧ M = 70 := by
sorry

end NUMINAMATH_GPT_zack_initial_marbles_l959_95920


namespace NUMINAMATH_GPT_highest_power_of_2_divides_n_highest_power_of_3_divides_n_l959_95959

noncomputable def n : ℕ := 15^4 - 11^4

theorem highest_power_of_2_divides_n : ∃ k : ℕ, 2^4 = 16 ∧ 2^(k) ∣ n :=
by
  sorry

theorem highest_power_of_3_divides_n : ∃ m : ℕ, 3^0 = 1 ∧ 3^(m) ∣ n :=
by
  sorry

end NUMINAMATH_GPT_highest_power_of_2_divides_n_highest_power_of_3_divides_n_l959_95959


namespace NUMINAMATH_GPT_range_of_m_l959_95989

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + m * x + 2 * m - 3 < 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l959_95989


namespace NUMINAMATH_GPT_cube_surface_area_l959_95929

-- Define the edge length of the cube
def edge_length : ℝ := 4

-- Define the formula for the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Prove that given the edge length is 4 cm, the surface area is 96 cm²
theorem cube_surface_area : surface_area edge_length = 96 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cube_surface_area_l959_95929


namespace NUMINAMATH_GPT_no_such_ab_l959_95962

theorem no_such_ab (a b : ℤ) : ¬ (2006^2 ∣ a^2006 + b^2006 + 1) :=
sorry

end NUMINAMATH_GPT_no_such_ab_l959_95962


namespace NUMINAMATH_GPT_max_lessons_l959_95988

-- Declaring noncomputable variables for the number of shirts, pairs of pants, and pairs of shoes.
noncomputable def s : ℕ := sorry
noncomputable def p : ℕ := sorry
noncomputable def b : ℕ := sorry

lemma conditions_satisfied :
  2 * (s + 1) * p * b = 2 * s * p * b + 36 ∧
  2 * s * (p + 1) * b = 2 * s * p * b + 72 ∧
  2 * s * p * (b + 1) = 2 * s * p * b + 54 ∧
  s * p * b = 27 ∧
  s * b = 36 ∧
  p * b = 18 := by
  sorry

theorem max_lessons : (2 * s * p * b) = 216 :=
by
  have h := conditions_satisfied
  sorry

end NUMINAMATH_GPT_max_lessons_l959_95988


namespace NUMINAMATH_GPT_solution_y_amount_l959_95985

theorem solution_y_amount :
  ∀ (y : ℝ) (volume_x volume_y : ℝ),
    volume_x = 200 ∧
    volume_y = y ∧
    10 / 100 * volume_x = 20 ∧
    30 / 100 * volume_y = 0.3 * y ∧
    (20 + 0.3 * y) / (volume_x + y) = 0.25 →
    y = 600 :=
by 
  intros y volume_x volume_y
  intros H
  sorry

end NUMINAMATH_GPT_solution_y_amount_l959_95985


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l959_95907

-- Define the arithmetic sequence condition and sum of given terms
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n d : ℕ, a n = a 1 + (n - 1) * d

def given_sum_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_arithmetic_sequence (a : ℕ → ℕ) (h_arith_seq : arithmetic_sequence a) 
  (h_sum_cond : given_sum_condition a) : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry  -- Proof of the theorem

end NUMINAMATH_GPT_sum_arithmetic_sequence_l959_95907


namespace NUMINAMATH_GPT_product_consecutive_even_div_48_l959_95979

theorem product_consecutive_even_div_48 (k : ℤ) : 
  (2 * k) * (2 * k + 2) * (2 * k + 4) % 48 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_consecutive_even_div_48_l959_95979


namespace NUMINAMATH_GPT_units_digit_2_pow_10_l959_95971

theorem units_digit_2_pow_10 : (2 ^ 10) % 10 = 4 := 
sorry

end NUMINAMATH_GPT_units_digit_2_pow_10_l959_95971


namespace NUMINAMATH_GPT_min_value_proof_l959_95950

noncomputable def min_value (m n : ℝ) : ℝ := 
  if 4 * m + n = 1 ∧ (m > 0 ∧ n > 0) then (4 / m + 1 / n) else 0

theorem min_value_proof : ∃ m n : ℝ, 4 * m + n = 1 ∧ m > 0 ∧ n > 0 ∧ min_value m n = 25 :=
by
  -- stating the theorem conditionally 
  -- and expressing that there exists values of m and n
  sorry

end NUMINAMATH_GPT_min_value_proof_l959_95950


namespace NUMINAMATH_GPT_six_digit_perfect_square_l959_95961

theorem six_digit_perfect_square :
  ∃ n : ℕ, ∃ x : ℕ, (n ^ 2 = 763876) ∧ (n ^ 2 >= 100000) ∧ (n ^ 2 < 1000000) ∧ (5 ≤ x) ∧ (x < 50) ∧ (76 * 10000 + 38 * 100 + 76 = 763876) ∧ (38 = 76 / 2) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_perfect_square_l959_95961


namespace NUMINAMATH_GPT_nate_ratio_is_four_to_one_l959_95925

def nate_exercise : Prop :=
  ∃ (D T L : ℕ), 
    T = D + 500 ∧ 
    T = 1172 ∧ 
    L = 168 ∧ 
    D / L = 4

theorem nate_ratio_is_four_to_one : nate_exercise := 
  sorry

end NUMINAMATH_GPT_nate_ratio_is_four_to_one_l959_95925


namespace NUMINAMATH_GPT_convex_polygon_longest_sides_convex_polygon_shortest_sides_l959_95917

noncomputable def convex_polygon : Type := sorry

-- Definitions for the properties and functions used in conditions
def is_convex (P : convex_polygon) : Prop := sorry
def equal_perimeters (A B : convex_polygon) : Prop := sorry
def longest_side (P : convex_polygon) : ℝ := sorry
def shortest_side (P : convex_polygon) : ℝ := sorry

-- Problem part a
theorem convex_polygon_longest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ∃ (A B : convex_polygon), equal_perimeters A B ∧ longest_side A = longest_side B :=
sorry

-- Problem part b
theorem convex_polygon_shortest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ¬(∀ (A B : convex_polygon), equal_perimeters A B → shortest_side A = shortest_side B) :=
sorry

end NUMINAMATH_GPT_convex_polygon_longest_sides_convex_polygon_shortest_sides_l959_95917


namespace NUMINAMATH_GPT_lance_pennies_saved_l959_95939

theorem lance_pennies_saved :
  let a := 5
  let d := 2
  let n := 20
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n = 480 :=
by
  sorry

end NUMINAMATH_GPT_lance_pennies_saved_l959_95939


namespace NUMINAMATH_GPT_max_colors_404_max_colors_406_l959_95995

theorem max_colors_404 (n k : ℕ) (h1 : n = 404) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

theorem max_colors_406 (n k : ℕ) (h1 : n = 406) 
  (h2 : ∃ (houses : ℕ → ℕ), (∀ c : ℕ, ∃ i : ℕ, (∀ j : ℕ, j < 100 → houses (i + j) = c) 
  ∧ ∀ c' : ℕ, c' ≠ c → (∃ j : ℕ, j < 100 → houses (i + j) ≠ c'))) : 
  k ≤ 202 :=
sorry

end NUMINAMATH_GPT_max_colors_404_max_colors_406_l959_95995


namespace NUMINAMATH_GPT_percentage_of_180_equation_l959_95998

theorem percentage_of_180_equation (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * ((P / 100) * 180) = 36) : P = 30 :=
sorry

end NUMINAMATH_GPT_percentage_of_180_equation_l959_95998


namespace NUMINAMATH_GPT_how_much_money_per_tshirt_l959_95982

def money_made_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) : Prop :=
  total_money_tshirts / number_tshirts = 62

theorem how_much_money_per_tshirt 
  (total_money_tshirts : ℕ) 
  (number_tshirts : ℕ) 
  (h1 : total_money_tshirts = 11346) 
  (h2 : number_tshirts = 183) : 
  money_made_per_tshirt total_money_tshirts number_tshirts := 
by 
  sorry

end NUMINAMATH_GPT_how_much_money_per_tshirt_l959_95982


namespace NUMINAMATH_GPT_distribute_balls_into_boxes_l959_95928

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end NUMINAMATH_GPT_distribute_balls_into_boxes_l959_95928


namespace NUMINAMATH_GPT_arithmetic_geometric_condition_l959_95983

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n-1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_arith_seq (a₁ d n : ℕ) : ℕ := n * a₁ + (n * (n-1) / 2) * d

-- Given conditions and required proofs
theorem arithmetic_geometric_condition {d a₁ : ℕ} (h : d ≠ 0) (S₃ : sum_arith_seq a₁ d 3 = 9)
  (geometric_seq : (arithmetic_seq a₁ d 5)^2 = (arithmetic_seq a₁ d 3) * (arithmetic_seq a₁ d 8)) :
  d = 1 ∧ ∀ n, sum_arith_seq 2 1 n = (n^2 + 3 * n) / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_condition_l959_95983


namespace NUMINAMATH_GPT_avg_bc_eq_28_l959_95968

variable (A B C : ℝ)

-- Conditions
def avg_abc_eq_30 : Prop := (A + B + C) / 3 = 30
def avg_ab_eq_25 : Prop := (A + B) / 2 = 25
def b_eq_16 : Prop := B = 16

-- The Proved Statement
theorem avg_bc_eq_28 (h1 : avg_abc_eq_30 A B C) (h2 : avg_ab_eq_25 A B) (h3 : b_eq_16 B) : (B + C) / 2 = 28 := 
by
  sorry

end NUMINAMATH_GPT_avg_bc_eq_28_l959_95968


namespace NUMINAMATH_GPT_full_price_shoes_l959_95960

variable (P : ℝ)

def full_price (P : ℝ) : ℝ := P
def discount_1_year (P : ℝ) : ℝ := 0.80 * P
def discount_3_years (P : ℝ) : ℝ := 0.75 * discount_1_year P
def price_after_discounts (P : ℝ) : ℝ := 0.60 * P

theorem full_price_shoes : price_after_discounts P = 51 → full_price P = 85 :=
by
  -- Placeholder for proof steps,
  sorry

end NUMINAMATH_GPT_full_price_shoes_l959_95960


namespace NUMINAMATH_GPT_angle_B_of_right_triangle_l959_95984

theorem angle_B_of_right_triangle (B C : ℝ) (hA : A = 90) (hC : C = 3 * B) (h_sum : A + B + C = 180) : B = 22.5 :=
sorry

end NUMINAMATH_GPT_angle_B_of_right_triangle_l959_95984


namespace NUMINAMATH_GPT_length_of_segment_GH_l959_95946

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_GH_l959_95946


namespace NUMINAMATH_GPT_product_even_permutation_l959_95909

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_even_permutation_l959_95909


namespace NUMINAMATH_GPT_gcm_less_than_90_l959_95986

theorem gcm_less_than_90 (a b : ℕ) (h1 : a = 8) (h2 : b = 12) : 
  ∃ x : ℕ, x < 90 ∧ ∀ y : ℕ, y < 90 → (a ∣ y) ∧ (b ∣ y) → y ≤ x → x = 72 :=
sorry

end NUMINAMATH_GPT_gcm_less_than_90_l959_95986


namespace NUMINAMATH_GPT_binomial_coeff_sum_l959_95990

theorem binomial_coeff_sum 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h1 : (1 - 2 * 0 : ℝ)^(7) = a_0 + a_1 * 0 + a_2 * 0^2 + a_3 * 0^3 + a_4 * 0^4 + a_5 * 0^5 + a_6 * 0^6 + a_7 * 0^7)
  (h2 : (1 - 2 * 1 : ℝ)^(7) = a_0 + a_1 * 1 + a_2 * 1^2 + a_3 * 1^3 + a_4 * 1^4 + a_5 * 1^5 + a_6 * 1^6 + a_7 * 1^7) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 := 
sorry

end NUMINAMATH_GPT_binomial_coeff_sum_l959_95990


namespace NUMINAMATH_GPT_problem_a2_minus_b2_problem_a3_minus_b3_l959_95955

variable (a b : ℝ)
variable (h1 : a + b = 8)
variable (h2 : a - b = 4)

theorem problem_a2_minus_b2 :
  a^2 - b^2 = 32 := 
by
sorry

theorem problem_a3_minus_b3 :
  a^3 - b^3 = 208 := 
by
sorry

end NUMINAMATH_GPT_problem_a2_minus_b2_problem_a3_minus_b3_l959_95955


namespace NUMINAMATH_GPT_taehyung_mom_age_l959_95947

variables (taehyung_age_diff_mom : ℕ) (taehyung_age_diff_brother : ℕ) (brother_age : ℕ)

theorem taehyung_mom_age 
  (h1 : taehyung_age_diff_mom = 31) 
  (h2 : taehyung_age_diff_brother = 5) 
  (h3 : brother_age = 7) 
  : 43 = brother_age + taehyung_age_diff_brother + taehyung_age_diff_mom := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_taehyung_mom_age_l959_95947


namespace NUMINAMATH_GPT_true_inverse_negation_l959_95957

theorem true_inverse_negation : ∀ (α β : ℝ),
  (α = β) ↔ (α = β) := 
sorry

end NUMINAMATH_GPT_true_inverse_negation_l959_95957


namespace NUMINAMATH_GPT_min_value_l959_95916

theorem min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ x, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_l959_95916


namespace NUMINAMATH_GPT_mass_percentage_Ba_in_BaI2_l959_95965

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 :
  (molar_mass_Ba / molar_mass_BaI2 * 100) = 35.11 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_Ba_in_BaI2_l959_95965


namespace NUMINAMATH_GPT_propositions_correct_l959_95923

def f (x : Real) (b c : Real) : Real := x * abs x + b * x + c

-- Define proposition P1: When c = 0, y = f(x) is an odd function.
def P1 (b : Real) : Prop :=
  ∀ x : Real, f x b 0 = - f (-x) b 0

-- Define proposition P2: When b = 0 and c > 0, the equation f(x) = 0 has only one real root.
def P2 (c : Real) : Prop :=
  c > 0 → ∃! x : Real, f x 0 c = 0

-- Define proposition P3: The graph of y = f(x) is symmetric about the point (0, c).
def P3 (b c : Real) : Prop :=
  ∀ x : Real, f x b c = 2 * c - f x b c

-- Define the final theorem statement
theorem propositions_correct (b c : Real) : P1 b ∧ P2 c ∧ P3 b c := sorry

end NUMINAMATH_GPT_propositions_correct_l959_95923


namespace NUMINAMATH_GPT_initial_oranges_correct_l959_95945

-- Define constants for the conditions
def oranges_shared : ℕ := 4
def oranges_left : ℕ := 42

-- Define the initial number of oranges
def initial_oranges : ℕ := oranges_left + oranges_shared

-- The theorem to prove
theorem initial_oranges_correct : initial_oranges = 46 :=
by 
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_initial_oranges_correct_l959_95945


namespace NUMINAMATH_GPT_bridge_supports_88_ounces_l959_95918

-- Define the conditions
def weight_of_soda_per_can : ℕ := 12
def number_of_soda_cans : ℕ := 6
def weight_of_empty_can : ℕ := 2
def additional_empty_cans : ℕ := 2

-- Define the total weight the bridge must hold up
def total_weight_bridge_support : ℕ :=
  (number_of_soda_cans * weight_of_soda_per_can) + ((number_of_soda_cans + additional_empty_cans) * weight_of_empty_can)

-- Prove that the total weight is 88 ounces
theorem bridge_supports_88_ounces : total_weight_bridge_support = 88 := by
  sorry

end NUMINAMATH_GPT_bridge_supports_88_ounces_l959_95918


namespace NUMINAMATH_GPT_wendy_percentage_accounting_related_jobs_l959_95978

noncomputable def wendy_accountant_years : ℝ := 25.5
noncomputable def wendy_accounting_manager_years : ℝ := 15.5 -- Including 6 months as 0.5 years
noncomputable def wendy_financial_consultant_years : ℝ := 10.25 -- Including 3 months as 0.25 years
noncomputable def wendy_tax_advisor_years : ℝ := 4
noncomputable def wendy_lifespan : ℝ := 80

theorem wendy_percentage_accounting_related_jobs :
  ((wendy_accountant_years + wendy_accounting_manager_years + wendy_financial_consultant_years + wendy_tax_advisor_years) / wendy_lifespan) * 100 = 69.0625 :=
by
  sorry

end NUMINAMATH_GPT_wendy_percentage_accounting_related_jobs_l959_95978


namespace NUMINAMATH_GPT_speed_of_train_is_20_l959_95912

def length_of_train := 120 -- in meters
def time_to_cross := 6 -- in seconds

def speed_of_train := length_of_train / time_to_cross -- Speed formula

theorem speed_of_train_is_20 :
  speed_of_train = 20 := by
  sorry

end NUMINAMATH_GPT_speed_of_train_is_20_l959_95912


namespace NUMINAMATH_GPT_find_real_numbers_l959_95997

theorem find_real_numbers (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_GPT_find_real_numbers_l959_95997


namespace NUMINAMATH_GPT_purely_imaginary_complex_number_l959_95976

theorem purely_imaginary_complex_number (a : ℝ) :
  (∃ b : ℝ, (a^2 - 3 * a + 2) = 0 ∧ a ≠ 1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_number_l959_95976


namespace NUMINAMATH_GPT_cubes_sum_eq_ten_squared_l959_95972

theorem cubes_sum_eq_ten_squared : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end NUMINAMATH_GPT_cubes_sum_eq_ten_squared_l959_95972


namespace NUMINAMATH_GPT_number_of_green_balls_l959_95970

-- Define the problem statement and conditions
def total_balls : ℕ := 12
def probability_both_green (g : ℕ) : ℚ := (g / 12) * ((g - 1) / 11)

-- The main theorem statement
theorem number_of_green_balls (g : ℕ) (h : probability_both_green g = 1 / 22) : g = 3 :=
sorry

end NUMINAMATH_GPT_number_of_green_balls_l959_95970


namespace NUMINAMATH_GPT_composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l959_95924

theorem composite_10201_base_gt_2 (x : ℕ) (hx : x > 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + 2*x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base (x : ℕ) (hx : x ≥ 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base_any_x (x : ℕ) (hx : x ≥ 1) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

end NUMINAMATH_GPT_composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l959_95924


namespace NUMINAMATH_GPT_Willy_more_crayons_l959_95906

theorem Willy_more_crayons (Willy Lucy : ℕ) (h1 : Willy = 1400) (h2 : Lucy = 290) : (Willy - Lucy) = 1110 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Willy_more_crayons_l959_95906


namespace NUMINAMATH_GPT_investment_return_formula_l959_95932

noncomputable def investment_return (x : ℕ) (x_pos : x > 0) : ℝ :=
  if x = 1 then 0.5
  else 2 ^ (x - 2)

theorem investment_return_formula (x : ℕ) (x_pos : x > 0) : investment_return x x_pos = 2 ^ (x - 2) := 
by
  sorry

end NUMINAMATH_GPT_investment_return_formula_l959_95932


namespace NUMINAMATH_GPT_length_of_bridge_l959_95956

theorem length_of_bridge
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (conversion_factor : ℝ)
  (bridge_length : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  train_speed_kmph = 120 →
  conversion_factor = 1 / 3.6 →
  bridge_length = 299.96 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l959_95956


namespace NUMINAMATH_GPT_nina_money_l959_95977

variable (C : ℝ)

theorem nina_money (h1: 6 * C = 8 * (C - 1.15)) : 6 * C = 27.6 := by
  have h2: C = 4.6 := sorry
  rw [h2]
  norm_num
  done

end NUMINAMATH_GPT_nina_money_l959_95977


namespace NUMINAMATH_GPT_apples_used_l959_95953

def initial_apples : ℕ := 43
def apples_left : ℕ := 2

theorem apples_used : initial_apples - apples_left = 41 :=
by sorry

end NUMINAMATH_GPT_apples_used_l959_95953


namespace NUMINAMATH_GPT_max_sum_of_multiplication_table_l959_95974

theorem max_sum_of_multiplication_table :
  let numbers := [3, 5, 7, 11, 17, 19]
  let repeated_num := 19
  ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧
  3 * repeated_num * (d + e + f) = 1995 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_sum_of_multiplication_table_l959_95974


namespace NUMINAMATH_GPT_sin_C_of_right_triangle_l959_95901

theorem sin_C_of_right_triangle (A B C: ℝ) (sinA: ℝ) (sinB: ℝ) (sinC: ℝ) :
  (sinA = 8/17) →
  (sinB = 1) →
  (A + B + C = π) →
  (B = π / 2) →
  (sinC = 15/17) :=
  
by
  intro h_sinA h_sinB h_triangle h_B
  sorry -- Proof is not required

end NUMINAMATH_GPT_sin_C_of_right_triangle_l959_95901


namespace NUMINAMATH_GPT_eggs_per_basket_l959_95905

-- Lucas places a total of 30 blue Easter eggs in several yellow baskets
-- Lucas places a total of 42 green Easter eggs in some purple baskets
-- Each basket contains the same number of eggs
-- There are at least 5 eggs in each basket

theorem eggs_per_basket (n : ℕ) (h1 : n ∣ 30) (h2 : n ∣ 42) (h3 : n ≥ 5) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_eggs_per_basket_l959_95905


namespace NUMINAMATH_GPT_David_squats_l959_95980

theorem David_squats (h1: ∀ d z: ℕ, d = 3 * 58) : d = 174 :=
by
  sorry

end NUMINAMATH_GPT_David_squats_l959_95980


namespace NUMINAMATH_GPT_constant_term_expansion_l959_95936

-- Defining the binomial theorem term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The general term of the binomial expansion (2sqrt(x) - 1/x)^6
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  binomial_coeff 6 r * (-1)^r * (2^(6-r)) * x^((6 - 3 * r) / 2)

-- Problem statement: Show that the constant term in the expansion is 240
theorem constant_term_expansion :
  (∃ r : ℕ, (6 - 3 * r) / 2 = 0 ∧ 
            general_term r arbitrary = 240) :=
sorry

end NUMINAMATH_GPT_constant_term_expansion_l959_95936


namespace NUMINAMATH_GPT_least_value_is_one_l959_95964

noncomputable def least_possible_value (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x^2 + y)^2

theorem least_value_is_one : ∀ x y : ℝ, (least_possible_value x y) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_least_value_is_one_l959_95964


namespace NUMINAMATH_GPT_weight_of_banana_l959_95903

theorem weight_of_banana (A B G : ℝ) (h1 : 3 * A = G) (h2 : 4 * B = 2 * A) (h3 : G = 576) : B = 96 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_banana_l959_95903


namespace NUMINAMATH_GPT_triangle_area_ratio_l959_95951

theorem triangle_area_ratio 
  (AB BC CA : ℝ)
  (p q r : ℝ)
  (ABC_area DEF_area : ℝ)
  (hAB : AB = 12)
  (hBC : BC = 16)
  (hCA : CA = 20)
  (h1 : p + q + r = 3 / 4)
  (h2 : p^2 + q^2 + r^2 = 1 / 2)
  (area_DEF_to_ABC : DEF_area / ABC_area = 385 / 512)
  : 897 = 385 + 512 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l959_95951


namespace NUMINAMATH_GPT_determine_n_l959_95940

theorem determine_n 
    (n : ℕ) (h2 : n ≥ 2) 
    (a : ℕ) (ha_div_n : a ∣ n) 
    (ha_min : ∀ d : ℕ, d ∣ n → d > 1 → d ≥ a) 
    (b : ℕ) (hb_div_n : b ∣ n)
    (h_eq : n = a^2 + b^2) : 
    n = 8 ∨ n = 20 :=
sorry

end NUMINAMATH_GPT_determine_n_l959_95940


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l959_95975

theorem largest_angle_in_triangle (y : ℝ) (h : 60 + 70 + y = 180) :
    70 > 60 ∧ 70 > y :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_angle_in_triangle_l959_95975


namespace NUMINAMATH_GPT_compute_xy_l959_95973

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56 / 9 :=
by
  sorry

end NUMINAMATH_GPT_compute_xy_l959_95973


namespace NUMINAMATH_GPT_longer_diagonal_of_rhombus_l959_95934

theorem longer_diagonal_of_rhombus
  (A : ℝ) (r1 r2 : ℝ) (x : ℝ)
  (hA : A = 135)
  (h_ratio : r1 = 5) (h_ratio2 : r2 = 3)
  (h_area : (1/2) * (r1 * x) * (r2 * x) = A) :
  r1 * x = 15 :=
by
  sorry

end NUMINAMATH_GPT_longer_diagonal_of_rhombus_l959_95934


namespace NUMINAMATH_GPT_balance_the_scale_l959_95921

theorem balance_the_scale (w1 : ℝ) (w2 : ℝ) (book_weight : ℝ) (h1 : w1 = 0.5) (h2 : w2 = 0.3) :
  book_weight = w1 + 2 * w2 :=
by
  sorry

end NUMINAMATH_GPT_balance_the_scale_l959_95921


namespace NUMINAMATH_GPT_money_inequality_l959_95919

-- Definitions and conditions
variables (a b : ℝ)
axiom cond1 : 6 * a + b > 78
axiom cond2 : 4 * a - b = 42

-- Theorem that encapsulates the problem and required proof
theorem money_inequality (a b : ℝ) (h1: 6 * a + b > 78) (h2: 4 * a - b = 42) : a > 12 ∧ b > 6 :=
  sorry

end NUMINAMATH_GPT_money_inequality_l959_95919


namespace NUMINAMATH_GPT_coefficients_verification_l959_95981

theorem coefficients_verification :
  let a0 := -3
  let a1 := -13 -- Not required as part of the proof but shown for completeness
  let a2 := 6
  let a3 := 0 -- Filler value to ensure there is a6 value
  let a4 := 0 -- Filler value to ensure there is a6 value
  let a5 := 0 -- Filler value to ensure there is a6 value
  let a6 := 0 -- Filler value to ensure there is a6 value
  (1 + 2*x) * (x - 2)^5 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5 + a6 * (1 - x)^6 ->
  a0 = -3 ∧
  a0 + a1 + a2 + a3 + a4 + a5 + a6 = -32 :=
by
  intro a0 a1 a2 a3 a4 a5 a6 h
  exact ⟨rfl, sorry⟩

end NUMINAMATH_GPT_coefficients_verification_l959_95981
